import sqlite3
import subprocess

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from prefect import flow, task

from job_shared import AUDITOK_SCRIPT, init_db, logger, sqlite_db_path, upsert_media_file

router = APIRouter()


class AuditokRequest(BaseModel):
    input: str = Field(..., description="Path to input audio file")
    save_detections_as: str | None = Field(
        None,
        description="Output path template with {id}, {start}, {end}, {duration} placeholders",
    )
    min_dur: float = Field(0.2, description="Minimum event duration in seconds")
    max_dur: float = Field(5.0, description="Maximum event duration in seconds")
    max_silence: float = Field(0.3, description="Max silence within event in seconds")
    energy_threshold: float = Field(50.0, description="Log energy threshold")
    quiet: bool = Field(False, description="Suppress stdout output")


class AuditokEvent(BaseModel):
    id: int
    start: float
    end: float


class AuditokResponse(BaseModel):
    audio: str
    events: list[AuditokEvent]
    count: int
    raw_output: str


# ---------------------------------------------------------------------------
# Subprocess/program execution
# ---------------------------------------------------------------------------


@task(name="run-auditok", retries=1, log_prints=True)
def run_auditok(request: AuditokRequest) -> AuditokResponse:
    """Run auditok on an audio file and return detected events."""
    script = str(AUDITOK_SCRIPT)
    cmd = [
        script,
        request.input,
        "-n",
        str(request.min_dur),
        "-m",
        str(request.max_dur),
        "-s",
        str(request.max_silence),
        "-e",
        str(request.energy_threshold),
    ]

    if request.save_detections_as:
        cmd += ["-o", request.save_detections_as]
    if request.quiet:
        cmd.append("-q")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"auditok exited with code {result.returncode}: {result.stderr.strip()}"
        )

    events = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and not line.strip().startswith("Saved:"):
            try:
                events.append(
                    AuditokEvent(
                        id=int(parts[0]),
                        start=float(parts[1]),
                        end=float(parts[2]),
                    )
                )
            except (ValueError, IndexError):
                continue

    response = AuditokResponse(
        audio=request.input,
        events=events,
        count=len(events),
        raw_output=result.stdout,
    )
    _save_auditok_to_db(response.audio, response.events)
    return response


# ---------------------------------------------------------------------------
# Prefect flow + API endpoint
# ---------------------------------------------------------------------------


@flow(name="auditok-flow", log_prints=True)
def auditok_flow(request: AuditokRequest) -> AuditokResponse:
    return run_auditok(request)


@router.post("/api/auditok", response_model=AuditokResponse)
async def api_auditok(request: AuditokRequest):
    """Detect audio activity regions in an audio file using auditok."""
    try:
        return auditok_flow(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------


def _save_auditok_to_db(audio_path: str, events: list[AuditokEvent]) -> None:
    """Persist Auditok output to SQLite (best-effort)."""
    conn = None
    try:
        conn = sqlite3.connect(sqlite_db_path())
        init_db(conn)
        media_file_id = upsert_media_file(conn, audio_path, "AUD")
        conn.execute("DELETE FROM auditok_events WHERE media_file_id = ?", (media_file_id,))
        conn.executemany(
            """
            INSERT INTO auditok_events (media_file_id, event_id, start, end)
            VALUES (?, ?, ?, ?)
        """,
            [(media_file_id, event.id, event.start, event.end) for event in events],
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to persist Auditok output for %s: %s", audio_path, exc)
    finally:
        if conn is not None:
            conn.close()
