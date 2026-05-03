"""
Prefect REST API for microservices orchestration.

Wraps whisperx, auditok, and scenedetect microservices as Prefect flows
exposed via FastAPI endpoints.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prefect import flow, task
from prefect.context import get_run_context
from job_auditok import router as auditok_router
from job_scenedetect import router as scenedetect_router
from job_shared import APPS_DIR, AUDITOK_SCRIPT, SCENEDETECT_SCRIPT, WHISPERX_SCRIPT
from job_test import router as test_router
from job_whisperx import router as whisperx_router

logger = logging.getLogger("prefect-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
    )
    logger.addHandler(_handler)

BATCH_STATE_FILE = Path(__file__).resolve().parent / ".batch_state.json"


def _load_repo_env() -> None:
    """Load key=value pairs from repo .env into process env if unset."""
    env_path = APPS_DIR / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_repo_env()

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".vob",
}

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class BatchRequest(BaseModel):
    filePath: str = Field(..., description="Directory to recursively scan for video files")


class VideoJobInfo(BaseModel):
    job_id: str
    video_path: str
    status: JobStatus
    output_file: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    flow_run_id: Optional[str] = None


class BatchCreateResponse(BaseModel):
    batch_id: str
    directory: str
    total: int
    video_files: list[str]


class BatchSummary(BaseModel):
    batch_id: str
    directory: str
    status: JobStatus
    total: int
    pending: int
    running: int
    completed: int
    failed: int
    created_at: str


class BatchDetail(BatchSummary):
    jobs: list[VideoJobInfo]


# ---------------------------------------------------------------------------
# Batch processing — in-memory state & background executor
# ---------------------------------------------------------------------------

_batches: dict[str, dict] = {}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def _find_video_files(directory: str) -> list[str]:
    """Recursively find all video files under *directory*."""
    return sorted(
        str(p) for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _batch_counts(batch_id: str) -> dict[str, int]:
    job_ids = _batches[batch_id]["job_ids"]
    statuses = [_jobs[jid]["status"] for jid in job_ids]
    return {
        "total": len(statuses),
        "pending": statuses.count(JobStatus.pending),
        "running": statuses.count(JobStatus.running),
        "completed": statuses.count(JobStatus.completed),
        "failed": statuses.count(JobStatus.failed),
    }


def _refresh_batch_status(batch_id: str):
    with _lock:
        batch = _batches.get(batch_id)
        if not batch:
            return
        statuses = [_jobs[jid]["status"] for jid in batch["job_ids"]]
        if all(s in (JobStatus.completed, JobStatus.failed) for s in statuses):
            batch["status"] = (
                JobStatus.completed
                if all(s == JobStatus.completed for s in statuses)
                else JobStatus.failed
            )
        elif any(s == JobStatus.running for s in statuses):
            batch["status"] = JobStatus.running


def _save_batch_state():
    """Persist batch/job metadata to disk for recovery across restarts."""
    state: dict = {"batches": {}, "jobs": {}}
    with _lock:
        for bid, batch in _batches.items():
            state["batches"][bid] = {
                "batch_id": batch["batch_id"],
                "directory": batch["directory"],
                "created_at": batch["created_at"],
                "job_ids": batch["job_ids"],
            }
        for jid, job in _jobs.items():
            state["jobs"][jid] = {
                "job_id": job["job_id"],
                "batch_id": job["batch_id"],
                "video_path": job["video_path"],
                "output_file": job["output_file"],
                "flow_run_id": job.get("flow_run_id"),
            }
    tmp = BATCH_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(BATCH_STATE_FILE)


async def _restore_batch_state():
    """Reconstruct in-memory batch state from disk + Prefect flow run states."""
    if not BATCH_STATE_FILE.exists():
        return

    try:
        with open(BATCH_STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Could not read batch state file; starting fresh")
        return

    from prefect.client.orchestration import get_client

    try:
        async with get_client() as client:
            for jid, jmeta in state.get("jobs", {}).items():
                status = JobStatus.failed
                error: str | None = None
                started_at: str | None = None
                completed_at: str | None = None
                flow_run_id = jmeta.get("flow_run_id")

                if flow_run_id:
                    try:
                        run = await client.read_flow_run(uuid.UUID(flow_run_id))
                        sname = (run.state_name or "").upper()
                        if sname == "COMPLETED":
                            status = JobStatus.completed
                        elif sname in ("RUNNING", "PENDING", "SCHEDULED"):
                            status = JobStatus.failed
                            error = "Server restarted while job was in progress"
                        else:
                            status = JobStatus.failed
                            if sname:
                                error = f"Prefect state: {run.state_name}"
                        started_at = (
                            run.start_time.isoformat() if run.start_time else None
                        )
                        end_time = getattr(run, "end_time", None)
                        completed_at = (
                            end_time.isoformat() if end_time else None
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not read flow run %s: %s", flow_run_id, exc,
                        )
                        error = "Could not retrieve state from Prefect"
                else:
                    error = "No flow run ID recorded (job may not have started)"

                _jobs[jid] = {
                    "job_id": jid,
                    "batch_id": jmeta["batch_id"],
                    "video_path": jmeta["video_path"],
                    "output_file": jmeta["output_file"],
                    "status": status,
                    "error": error,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "flow_run_id": flow_run_id,
                }

            for bid, bmeta in state.get("batches", {}).items():
                valid_job_ids = [j for j in bmeta["job_ids"] if j in _jobs]
                _batches[bid] = {
                    "batch_id": bid,
                    "directory": bmeta["directory"],
                    "created_at": bmeta["created_at"],
                    "job_ids": valid_job_ids,
                    "status": JobStatus.pending,
                }
                _refresh_batch_status(bid)

        logger.info(
            "Restored %d batch(es) and %d job(s) from previous state",
            len(_batches), len(_jobs),
        )
        _save_batch_state()
    except Exception as exc:
        logger.warning("Failed to restore batch state: %s", exc)


def _run_service(label: str, cmd: list[str], output_file: str, timeout: int = 600):
    """Run a subprocess, appending all output to *output_file*. Returns True on success."""
    with open(output_file, "a") as f:
        f.write(f"\n--- {label} ---\n")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        with open(output_file, "a") as f:
            f.write(f"Exit code: {result.returncode}\n")
            if result.stdout:
                f.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    f.write("\n")
            if result.stderr:
                f.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    f.write("\n")
        return result.returncode == 0
    except Exception as exc:
        with open(output_file, "a") as f:
            f.write(f"Error: {exc}\n")
        return False


@task(name="batch-scenedetect", retries=1, log_prints=True)
def batch_run_scenedetect(video_path: str, output_file: str) -> bool:
    """Run SceneDetect on a video as part of a batch."""
    return _run_service(
        "SceneDetect",
        [str(SCENEDETECT_SCRIPT),
         video_path, "-j"],
        output_file,
        timeout=600,
    )


@task(name="batch-auditok", retries=1, log_prints=True)
def batch_run_auditok(video_path: str, output_file: str) -> bool:
    """Run Auditok on a video as part of a batch."""
    return _run_service(
        "Auditok",
        [str(AUDITOK_SCRIPT),
         video_path],
        output_file,
        timeout=600,
    )


@task(name="batch-whisperx", retries=1, log_prints=True)
def batch_run_whisperx(video_path: str, output_file: str) -> bool:
    """Run WhisperX on a video as part of a batch."""
    return _run_service(
        "WhisperX",
        [str(WHISPERX_SCRIPT),
         video_path],
        output_file,
        timeout=1800,
    )


@flow(name="process-video", log_prints=True)
def process_video_flow(job_id: str, batch_id: str):
    """Prefect flow: run scenedetect, auditok, and whisperx on a single video."""
    job = _jobs[job_id]
    video_path = job["video_path"]
    output_file = job["output_file"]

    ctx = get_run_context()
    with _lock:
        job["flow_run_id"] = str(ctx.flow_run.id)
        job["status"] = JobStatus.running
        job["started_at"] = datetime.now(timezone.utc).isoformat()
    _refresh_batch_status(batch_id)
    _save_batch_state()

    with open(output_file, "w") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Started: {job['started_at']}\n")
        f.write("=" * 60 + "\n")

    try:
        ok1 = batch_run_scenedetect(video_path, output_file)
        ok2 = batch_run_auditok(video_path, output_file)
        ok3 = batch_run_whisperx(video_path, output_file)
        all_ok = ok1 and ok2 and ok3
    except Exception:
        all_ok = False

    now = datetime.now(timezone.utc).isoformat()
    with open(output_file, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Completed: {now}\n")
        f.write(f"Result: {'OK' if all_ok else 'ERRORS — see above'}\n")

    with _lock:
        job["status"] = JobStatus.completed if all_ok else JobStatus.failed
        job["completed_at"] = now
        if not all_ok:
            job["error"] = "One or more services failed; see output file"
    _refresh_batch_status(batch_id)
    _save_batch_state()

    if not all_ok:
        raise RuntimeError("One or more services failed; see output file")


def _on_flow_done(job_id: str, batch_id: str, future):
    """ThreadPoolExecutor callback — marks the job as failed if the flow raised
    before our in-flow status update had a chance to run (e.g. Prefect server
    unreachable)."""
    exc = future.exception()
    if exc is None:
        return
    with _lock:
        job = _jobs.get(job_id)
        if job and job["status"] not in (JobStatus.completed, JobStatus.failed):
            job["status"] = JobStatus.failed
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = str(exc)
    _refresh_batch_status(batch_id)
    _save_batch_state()


# ---------------------------------------------------------------------------
# Prefect dashboard lifecycle
# ---------------------------------------------------------------------------

PREFECT_SERVER_HOST = os.environ.get("PREFECT_DASHBOARD_HOST", "0.0.0.0")
PREFECT_SERVER_PORT = int(os.environ.get("PREFECT_DASHBOARD_PORT", "4200"))

_prefect_server_process: subprocess.Popen | None = None


def _start_prefect_server() -> subprocess.Popen | None:
    """Launch `prefect server start` as a background subprocess."""
    api_url = f"http://127.0.0.1:{PREFECT_SERVER_PORT}/api"
    os.environ["PREFECT_API_URL"] = api_url

    cmd = [
        sys.executable, "-m", "prefect", "server", "start",
        "--host", PREFECT_SERVER_HOST,
        "--port", str(PREFECT_SERVER_PORT),
    ]
    try:
        log_path = Path(__file__).resolve().parent / ".prefect_server.log"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        logger.info(
            "Prefect dashboard starting on http://%s:%s  (pid %s, log %s)",
            PREFECT_SERVER_HOST, PREFECT_SERVER_PORT, proc.pid, log_path,
        )

        import httpx

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logger.error(
                    "Prefect server exited immediately (code %s)", proc.returncode,
                )
                return None
            try:
                resp = httpx.get(
                    f"http://127.0.0.1:{PREFECT_SERVER_PORT}/api/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    logger.info("Prefect dashboard is ready")
                    return proc
            except Exception:
                pass
            time.sleep(1)

        logger.warning(
            "Prefect dashboard did not become healthy within 30 s — continuing anyway"
        )
        return proc
    except FileNotFoundError:
        logger.error("prefect CLI not found — dashboard will not be available")
        return None
    except Exception as exc:
        logger.error("Failed to start Prefect dashboard: %s", exc)
        return None


def _stop_prefect_server():
    global _prefect_server_process
    proc = _prefect_server_process
    if proc is None or proc.poll() is not None:
        return
    logger.info("Shutting down Prefect dashboard (pid %s) …", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    _prefect_server_process = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    global _prefect_server_process
    _prefect_server_process = _start_prefect_server()
    await _restore_batch_state()
    yield
    _stop_prefect_server()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Microservices Orchestration API",
    description="Prefect-powered REST API for scenedetect, auditok, and whisperx microservices",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(scenedetect_router)
app.include_router(auditok_router)
app.include_router(whisperx_router)
app.include_router(test_router)


# ---------------------------------------------------------------------------
# Batch processing endpoints
# ---------------------------------------------------------------------------


@app.post("/api/batch", response_model=BatchCreateResponse)
async def api_create_batch(request: BatchRequest):
    """Recursively find all video files in *filePath* and queue them for processing."""
    directory = request.filePath
    if not Path(directory).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    videos = _find_video_files(directory)
    if not videos:
        raise HTTPException(
            status_code=400,
            detail=f"No video files found in: {directory}",
        )

    batch_id = str(uuid.uuid4())
    job_ids: list[str] = []

    with _lock:
        for video_path in videos:
            job_id = str(uuid.uuid4())
            _jobs[job_id] = {
                "job_id": job_id,
                "batch_id": batch_id,
                "video_path": video_path,
                "output_file": video_path + ".output.txt",
                "status": JobStatus.pending,
                "error": None,
                "started_at": None,
                "completed_at": None,
            }
            job_ids.append(job_id)

        _batches[batch_id] = {
            "batch_id": batch_id,
            "directory": directory,
            "status": JobStatus.running,
            "job_ids": job_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    _save_batch_state()

    for jid in job_ids:
        future = _executor.submit(process_video_flow, jid, batch_id)
        future.add_done_callback(
            lambda f, j=jid, b=batch_id: _on_flow_done(j, b, f)
        )

    return BatchCreateResponse(
        batch_id=batch_id,
        directory=directory,
        total=len(videos),
        video_files=videos,
    )


@app.get("/api/batch", response_model=list[BatchSummary])
async def api_list_batches():
    """List all batches with progress counts."""
    with _lock:
        results = []
        for batch_id, batch in _batches.items():
            counts = _batch_counts(batch_id)
            results.append(BatchSummary(
                batch_id=batch["batch_id"],
                directory=batch["directory"],
                status=batch["status"],
                created_at=batch["created_at"],
                **counts,
            ))
    return results


@app.get("/api/batch/{batch_id}", response_model=BatchDetail)
async def api_get_batch(batch_id: str):
    """Get detailed status for a batch including all individual jobs."""
    with _lock:
        batch = _batches.get(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

        counts = _batch_counts(batch_id)
        jobs = [
            VideoJobInfo(
                job_id=_jobs[jid]["job_id"],
                video_path=_jobs[jid]["video_path"],
                status=_jobs[jid]["status"],
                output_file=_jobs[jid]["output_file"],
                error=_jobs[jid]["error"],
                started_at=_jobs[jid]["started_at"],
                completed_at=_jobs[jid]["completed_at"],
                flow_run_id=_jobs[jid].get("flow_run_id"),
            )
            for jid in batch["job_ids"]
        ]

    return BatchDetail(
        batch_id=batch["batch_id"],
        directory=batch["directory"],
        status=batch["status"],
        created_at=batch["created_at"],
        jobs=jobs,
        **counts,
    )


@app.get("/api/batch/{batch_id}/jobs/{job_id}", response_model=VideoJobInfo)
async def api_get_job(batch_id: str, job_id: str):
    """Get status for a single video processing job."""
    with _lock:
        batch = _batches.get(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        if job_id not in batch["job_ids"]:
            raise HTTPException(status_code=404, detail="Job not found in this batch")
        job = _jobs[job_id]

    return VideoJobInfo(
        job_id=job["job_id"],
        video_path=job["video_path"],
        status=job["status"],
        output_file=job["output_file"],
        error=job["error"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        flow_run_id=job.get("flow_run_id"),
    )


@app.get("/health")
async def health():
    dashboard_alive = (
        _prefect_server_process is not None
        and _prefect_server_process.poll() is None
    )
    return {
        "status": "ok",
        "prefect_dashboard": {
            "running": dashboard_alive,
            "url": f"http://{PREFECT_SERVER_HOST}:{PREFECT_SERVER_PORT}",
        },
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8800)
