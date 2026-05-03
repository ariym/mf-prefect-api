import logging
import os
import sqlite3
import subprocess
from pathlib import Path

logger = logging.getLogger("prefect-api")

APPS_DIR = Path(__file__).resolve().parent.parent
PROGRAMS_DIR = APPS_DIR / "programs"
SCENEDETECT_SCRIPT = PROGRAMS_DIR / "mf-scenedetect-microservice" / "scenedetect_service.sh"
AUDITOK_SCRIPT = PROGRAMS_DIR / "mf-auditok-microservice" / "auditok_service.sh"
WHISPERX_SCRIPT = PROGRAMS_DIR / "mf-whisperx-microservice" / "whisperx_service.sh"
SAMPLE_MEDIA_DIR = Path("/home/cursor/sample_media")


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


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_files (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path    TEXT UNIQUE NOT NULL,
            video_length REAL,
            file_size    INTEGER,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
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
        CREATE TABLE IF NOT EXISTS whisperx_transcripts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id   INTEGER NOT NULL UNIQUE REFERENCES media_files(id)
                                ON DELETE CASCADE,
            language        TEXT,
            transcript_text TEXT,
            raw_json        TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_whisperx_transcripts_media_file_id
        ON whisperx_transcripts(media_file_id)
    """
    )


def upsert_media_file(conn: sqlite3.Connection, file_path: str) -> int:
    file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else None
    duration = get_media_duration(file_path)
    conn.execute(
        """
        INSERT INTO media_files (file_path, video_length, file_size)
        VALUES (?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            video_length = COALESCE(excluded.video_length, video_length),
            file_size    = COALESCE(excluded.file_size, file_size)
    """,
        (file_path, duration, file_size),
    )
    return conn.execute(
        "SELECT id FROM media_files WHERE file_path = ?",
        (file_path,),
    ).fetchone()[0]
