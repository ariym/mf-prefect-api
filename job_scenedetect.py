import json
import sqlite3
import subprocess

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from prefect import flow, task

from job_shared import SCENEDETECT_SCRIPT, init_db, logger, sqlite_db_path, upsert_media_file

router = APIRouter()


class SceneDetectRequest(BaseModel):
    input: str = Field(..., description="Path to input video file")
    threshold: float = Field(27.0, description="ContentDetector threshold")
    output: str | None = Field(None, description="Save scene list JSON to this path")
    quiet: bool = Field(False, description="Suppress progress output")


class SceneDetectScene(BaseModel):
    id: int
    start: float
    end: float
    start_timecode: str
    end_timecode: str
    start_frame: int
    end_frame: int


class SceneDetectResponse(BaseModel):
    video: str
    scenes: list[SceneDetectScene]
    count: int


# ---------------------------------------------------------------------------
# Subprocess/program execution
# ---------------------------------------------------------------------------


@task(name="run-scenedetect", retries=1, log_prints=True)
def run_scenedetect(request: SceneDetectRequest) -> SceneDetectResponse:
    """Run PySceneDetect on a video file and return structured results."""
    script = str(SCENEDETECT_SCRIPT)
    cmd = [script, request.input, "-j", "-t", str(request.threshold)]

    if request.output:
        cmd += ["-o", request.output]
    if request.quiet:
        cmd.append("-q")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"scenedetect exited with code {result.returncode}: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)
    response = SceneDetectResponse(**data)
    _save_scenedetect_to_db(response.video, response.scenes)
    return response


# ---------------------------------------------------------------------------
# Prefect flow + API endpoint
# ---------------------------------------------------------------------------


@flow(name="scenedetect-flow", log_prints=True)
def scenedetect_flow(request: SceneDetectRequest) -> SceneDetectResponse:
    return run_scenedetect(request)


@router.post("/api/scenedetect", response_model=SceneDetectResponse)
async def api_scenedetect(request: SceneDetectRequest):
    """Detect scene cuts in a video file using PySceneDetect."""
    try:
        return scenedetect_flow(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------


def _save_scenedetect_to_db(video_path: str, scenes: list[SceneDetectScene]) -> None:
    """Persist SceneDetect output to SQLite (best-effort)."""
    conn = None
    try:
        conn = sqlite3.connect(sqlite_db_path())
        init_db(conn)
        media_file_id = upsert_media_file(conn, video_path, "VID")
        conn.execute("DELETE FROM scenedetect_scenes WHERE media_file_id = ?", (media_file_id,))
        conn.executemany(
            """
            INSERT INTO scenedetect_scenes
            (media_file_id, scene_id, start, end, start_timecode, end_timecode, start_frame, end_frame)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    media_file_id,
                    scene.id,
                    scene.start,
                    scene.end,
                    scene.start_timecode,
                    scene.end_timecode,
                    scene.start_frame,
                    scene.end_frame,
                )
                for scene in scenes
            ],
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to persist SceneDetect output for %s: %s", video_path, exc)
    finally:
        if conn is not None:
            conn.close()
