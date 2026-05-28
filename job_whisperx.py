import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from prefect import flow, task

from job_shared import WHISPERX_SCRIPT, init_db, logger, sqlite_db_path, upsert_media_file

router = APIRouter()


class WhisperXRequest(BaseModel):
    audio: list[str] = Field(..., description="Audio file path(s) to transcribe", min_length=1)
    model: str = Field("base.en", description="Whisper model name (base.en for speed, small.en for accuracy)")
    model_dir: str | None = Field(None, description="Model cache directory")
    device: str | None = Field(
        None,
        description="Inference device (cuda/cpu). Null uses service defaults.",
    )
    device_index: int = Field(0, description="GPU device index")
    batch_size: int = Field(16, description="Batch size for transcription")
    compute_type: str | None = Field(
        None,
        description="float16, float32, int8, or int8_float16. Null uses service defaults.",
    )
    output_dir: str | None = Field(None, description="Output directory")
    output_format: str = Field("all", description="Output format: all, srt, vtt, txt, tsv, json, aud")
    task: str = Field("transcribe", description="transcribe or translate")
    language: str | None = Field("en", description="Language code (auto-detect if null)")
    align_model: str | None = Field(None, description="Alignment model name")
    interpolate_method: str = Field("nearest", description="nearest, linear, or ignore")
    no_align: bool = Field(False, description="Skip phoneme alignment")
    return_char_alignments: bool = Field(False, description="Include char-level alignments")
    vad_method: str = Field("pyannote", description="VAD method: pyannote or silero")
    vad_onset: float = Field(0.500, description="VAD onset threshold")
    vad_offset: float = Field(0.363, description="VAD offset threshold")
    chunk_size: int = Field(30, description="VAD chunk size in seconds")
    diarize: bool = Field(True, description="Enable speaker diarization")
    min_speakers: int | None = Field(None, description="Minimum number of speakers")
    max_speakers: int | None = Field(None, description="Maximum number of speakers")
    hf_token: str | None = Field(None, description="Hugging Face token for gated models")
    verbose: bool = Field(True, description="Print progress/debug output")


class WhisperXResponse(BaseModel):
    audio_files: list[str]
    output_dir: str
    output_files: list[str]
    raw_output: str


# ---------------------------------------------------------------------------
# Subprocess/program execution
# ---------------------------------------------------------------------------


@task(name="run-whisperx", retries=1, log_prints=True)
def run_whisperx(request: WhisperXRequest) -> WhisperXResponse:
    """Run WhisperX on audio file(s) and return transcription results."""
    script = str(WHISPERX_SCRIPT)

    out_dir = request.output_dir or tempfile.mkdtemp(prefix="whisperx_")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [script] + request.audio
    cmd += ["--model", request.model]
    cmd += ["-o", out_dir]
    cmd += ["-f", request.output_format]
    cmd += ["--task", request.task]
    cmd += ["--batch_size", str(request.batch_size)]
    cmd += ["--device_index", str(request.device_index)]
    cmd += ["--vad_method", request.vad_method]
    cmd += ["--vad_onset", str(request.vad_onset)]
    cmd += ["--vad_offset", str(request.vad_offset)]
    cmd += ["--chunk_size", str(request.chunk_size)]
    cmd += ["--interpolate_method", request.interpolate_method]

    if request.device:
        cmd += ["--device", request.device]
    if request.compute_type:
        cmd += ["--compute_type", request.compute_type]
    if request.model_dir:
        cmd += ["--model_dir", request.model_dir]
    if request.language:
        cmd += ["--language", request.language]
    if request.align_model:
        cmd += ["--align_model", request.align_model]
    if request.no_align:
        cmd.append("--no_align")
    if request.return_char_alignments:
        cmd.append("--return_char_alignments")
    if request.diarize:
        cmd.append("--diarize")
    if request.min_speakers is not None:
        cmd += ["--min_speakers", str(request.min_speakers)]
    if request.max_speakers is not None:
        cmd += ["--max_speakers", str(request.max_speakers)]
    if request.hf_token:
        cmd += ["--hf_token", request.hf_token]
    if not request.verbose:
        cmd += ["--verbose", "False"]

    # The wrapper script (`whisperx_service.sh`) prepends its own default flags
    # before the args we pass. For `--diarize` (argparse `store_true`) this
    # means the wrapper's default wins over the request and the flag cannot be
    # turned off from the CLI side. Override via env, which the wrapper reads
    # before composing its arg list.
    env = os.environ.copy()
    env["WHISPERX_DIARIZE"] = "true" if request.diarize else "false"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"whisperx exited with code {result.returncode}: {result.stderr.strip()}"
        )

    output_files = [str(p) for p in Path(out_dir).iterdir() if p.is_file()]

    for audio_path in request.audio:
        json_path = Path(out_dir) / f"{Path(audio_path).stem}.json"
        _save_whisperx_to_db(audio_path, json_path)

    return WhisperXResponse(
        audio_files=request.audio,
        output_dir=out_dir,
        output_files=sorted(output_files),
        raw_output=result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
    )


# ---------------------------------------------------------------------------
# Prefect flow + API endpoint
# ---------------------------------------------------------------------------


@flow(name="whisperx-flow", log_prints=True)
def whisperx_flow(request: WhisperXRequest) -> WhisperXResponse:
    return run_whisperx(request)


@router.post("/api/whisperx", response_model=WhisperXResponse)
async def api_whisperx(request: WhisperXRequest):
    """Transcribe audio file(s) using WhisperX."""
    try:
        return whisperx_flow(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------


def _save_whisperx_to_db(audio_path: str, json_path: Path) -> None:
    """Persist WhisperX output JSON to SQLite (best-effort)."""
    conn = None
    try:
        if not json_path.is_file():
            logger.warning(
                "WhisperX JSON not found for DB save: %s (audio: %s)",
                json_path,
                audio_path,
            )
            return

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        conn = sqlite3.connect(sqlite_db_path())
        init_db(conn)
        media_file_id = upsert_media_file(conn, audio_path, "AUD")
        conn.execute("DELETE FROM whisperx_segments WHERE media_file_id = ?", (media_file_id,))
        conn.executemany(
            """
            INSERT INTO whisperx_segments
            (media_file_id, segment_id, start, end, text, speaker)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    media_file_id,
                    idx + 1,
                    seg.get("start", 0),
                    seg.get("end", 0),
                    seg.get("text", ""),
                    seg.get("speaker"),
                )
                for idx, seg in enumerate(segments)
            ],
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to persist WhisperX output for %s: %s", audio_path, exc)
    finally:
        if conn is not None:
            conn.close()
