import json
import logging
import os
import sqlite3
import subprocess
from fractions import Fraction
from pathlib import Path

logger = logging.getLogger("prefect-api")

APPS_DIR = Path(__file__).resolve().parent.parent
PROGRAMS_DIR = APPS_DIR / "programs"
SCENEDETECT_SCRIPT = PROGRAMS_DIR / "mf-scenedetect-microservice" / "scenedetect_service.sh"
AUDITOK_SCRIPT = PROGRAMS_DIR / "mf-auditok-microservice" / "auditok_service.sh"
WHISPERX_SCRIPT = PROGRAMS_DIR / "mf-whisperx-microservice" / "whisperx_service.sh"
SAMPLE_MEDIA_DIR = Path("/home/cursor/sample_media")
MEDIA_TYPES = {"VID", "AUD"}


def sqlite_db_path() -> str:
    db_path_env = os.getenv("PATH_SQLITE_DB")
    if not db_path_env:
        raise RuntimeError("PATH_SQLITE_DB must be set (for example in .env)")
    db_path = Path(db_path_env).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def get_media_duration(file_path: str) -> float | None:
    """Return media duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _run_ffprobe_json(cmd: list[str]) -> dict | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def _parse_fps(raw_fps: str | None) -> float | None:
    if not raw_fps:
        return None
    try:
        return float(Fraction(raw_fps))
    except Exception:
        try:
            return float(raw_fps)
        except Exception:
            return None


def get_media_stream_metadata(file_path: str, media_type: str) -> dict[str, int | float | str] | None:
    if media_type == "VID":
        payload = _run_ffprobe_json(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate",
                "-select_streams",
                "v:0",
                file_path,
            ]
        )
        stream = (payload or {}).get("streams", [{}])[0]
        width = stream.get("width")
        height = stream.get("height")
        codec = stream.get("codec_name")
        fps = _parse_fps(stream.get("r_frame_rate"))
        metadata: dict[str, int | float | str] = {}
        if isinstance(width, int):
            metadata["width"] = width
        if isinstance(height, int):
            metadata["height"] = height
        if isinstance(codec, str) and codec:
            metadata["codec"] = codec
        if isinstance(fps, float):
            metadata["fps"] = fps
        return metadata or None

    if media_type == "AUD":
        payload = _run_ffprobe_json(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-select_streams",
                "a:0",
                file_path,
            ]
        )
        stream = (payload or {}).get("streams", [{}])[0]
        codec = stream.get("codec_name")
        sample_rate_raw = stream.get("sample_rate")
        channels = stream.get("channels")
        metadata = {}
        if isinstance(codec, str) and codec:
            metadata["codec"] = codec
        if isinstance(channels, int):
            metadata["channels"] = channels
        try:
            sample_rate = int(sample_rate_raw) if sample_rate_raw is not None else None
        except (TypeError, ValueError):
            sample_rate = None
        if sample_rate is not None:
            metadata["sample_rate"] = sample_rate
        return metadata or None

    return None


def _upsert_video_metadata(conn: sqlite3.Connection, media_file_id: int, metadata: dict[str, int | float | str]) -> None:
    conn.execute(
        """
        INSERT INTO video_metadata (media_file_id, width, height, fps, codec)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(media_file_id) DO UPDATE SET
            width = COALESCE(excluded.width, video_metadata.width),
            height = COALESCE(excluded.height, video_metadata.height),
            fps = COALESCE(excluded.fps, video_metadata.fps),
            codec = COALESCE(excluded.codec, video_metadata.codec)
    """,
        (
            media_file_id,
            metadata.get("width"),
            metadata.get("height"),
            metadata.get("fps"),
            metadata.get("codec"),
        ),
    )
    conn.execute("DELETE FROM audio_metadata WHERE media_file_id = ?", (media_file_id,))


def _upsert_audio_metadata(conn: sqlite3.Connection, media_file_id: int, metadata: dict[str, int | float | str]) -> None:
    conn.execute(
        """
        INSERT INTO audio_metadata (media_file_id, sample_rate, channels, codec)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(media_file_id) DO UPDATE SET
            sample_rate = COALESCE(excluded.sample_rate, audio_metadata.sample_rate),
            channels = COALESCE(excluded.channels, audio_metadata.channels),
            codec = COALESCE(excluded.codec, audio_metadata.codec)
    """,
        (
            media_file_id,
            metadata.get("sample_rate"),
            metadata.get("channels"),
            metadata.get("codec"),
        ),
    )
    conn.execute("DELETE FROM video_metadata WHERE media_file_id = ?", (media_file_id,))


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DROP TABLE IF EXISTS whisperx_transcripts")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_files (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path    TEXT UNIQUE NOT NULL,
            media_type   TEXT NOT NULL CHECK (media_type IN ('VID', 'AUD')),
            duration     REAL,
            file_size    INTEGER,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    media_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(media_files)").fetchall()
    }
    if "media_type" not in media_columns:
        conn.execute(
            """
            ALTER TABLE media_files
            ADD COLUMN media_type TEXT CHECK (media_type IN ('VID', 'AUD'))
        """
        )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS media_files_media_type_insert_guard
        BEFORE INSERT ON media_files
        FOR EACH ROW
        WHEN NEW.media_type IS NULL OR NEW.media_type NOT IN ('VID', 'AUD')
        BEGIN
            SELECT RAISE(ABORT, 'media_type must be VID or AUD');
        END;
    """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS media_files_media_type_update_guard
        BEFORE UPDATE OF media_type ON media_files
        FOR EACH ROW
        WHEN NEW.media_type IS NULL OR NEW.media_type NOT IN ('VID', 'AUD')
        BEGIN
            SELECT RAISE(ABORT, 'media_type must be VID or AUD');
        END;
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auditok_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id INTEGER NOT NULL REFERENCES media_files(id),
            event_id      INTEGER NOT NULL,
            start         REAL NOT NULL,
            end           REAL NOT NULL
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scenedetect_scenes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id  INTEGER NOT NULL REFERENCES media_files(id),
            scene_id       INTEGER NOT NULL,
            start          REAL NOT NULL,
            end            REAL NOT NULL,
            start_timecode TEXT,
            end_timecode   TEXT,
            start_frame    INTEGER,
            end_frame      INTEGER
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS whisperx_segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id INTEGER NOT NULL REFERENCES media_files(id),
            segment_id    INTEGER NOT NULL,
            start         REAL NOT NULL,
            end           REAL NOT NULL,
            text          TEXT,
            speaker       TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_metadata (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id  INTEGER UNIQUE NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
            width          INTEGER,
            height         INTEGER,
            fps            REAL,
            codec          TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_metadata (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id  INTEGER UNIQUE NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
            sample_rate    INTEGER,
            channels       INTEGER,
            codec          TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )


def upsert_media_file(conn: sqlite3.Connection, file_path: str, media_type: str) -> int:
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"media_type must be one of {sorted(MEDIA_TYPES)}")

    file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else None
    duration = get_media_duration(file_path)
    conn.execute(
        """
        INSERT INTO media_files (file_path, media_type, duration, file_size)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            media_type   = excluded.media_type,
            duration = COALESCE(excluded.duration, duration),
            file_size    = COALESCE(excluded.file_size, file_size)
    """,
        (file_path, media_type, duration, file_size),
    )
    media_file_id = conn.execute(
        "SELECT id FROM media_files WHERE file_path = ?",
        (file_path,),
    ).fetchone()[0]
    metadata = get_media_stream_metadata(file_path, media_type)
    if metadata:
        if media_type == "VID":
            _upsert_video_metadata(conn, media_file_id, metadata)
        else:
            _upsert_audio_metadata(conn, media_file_id, metadata)
    return media_file_id
