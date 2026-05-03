import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prefect import flow

from job_auditok import AuditokRequest, AuditokResponse, run_auditok
from job_scenedetect import SceneDetectRequest, SceneDetectResponse, run_scenedetect
from job_shared import APPS_DIR, SAMPLE_MEDIA_DIR
from job_whisperx import WhisperXRequest, WhisperXResponse, run_whisperx

router = APIRouter()


class TestResponse(BaseModel):
    scenedetect: SceneDetectResponse
    auditok: AuditokResponse
    whisperx: WhisperXResponse


# ---------------------------------------------------------------------------
# Subprocess/program execution
# ---------------------------------------------------------------------------
# None in this module: it orchestrates existing service tasks.


# ---------------------------------------------------------------------------
# Prefect flow + API endpoint
# ---------------------------------------------------------------------------


@flow(name="test-all-services", log_prints=True)
def test_all_services_flow() -> TestResponse:
    """Run all three microservices against sample media for validation."""
    video_path = os.getenv("PATH_TEST_VIDEO", str(SAMPLE_MEDIA_DIR / "road_to_damascus.mp4"))
    audio_path = os.getenv("PATH_TEST_AUDIO", str(SAMPLE_MEDIA_DIR / "ct_beans.mp3"))

    if not Path(video_path).is_absolute():
        video_path = str((APPS_DIR / video_path).resolve())
    if not Path(audio_path).is_absolute():
        audio_path = str((APPS_DIR / audio_path).resolve())

    scene_result = run_scenedetect(SceneDetectRequest(input=video_path))
    auditok_result = run_auditok(AuditokRequest(input=audio_path))
    whisperx_result = run_whisperx(WhisperXRequest(audio=[audio_path]))

    return TestResponse(
        scenedetect=scene_result,
        auditok=auditok_result,
        whisperx=whisperx_result,
    )


@router.post("/api/test", response_model=TestResponse)
async def api_test():
    """
    Test all microservices using media from environment variables.

    - PATH_TEST_VIDEO for scenedetect input
    - PATH_TEST_AUDIO for whisperx and auditok input
    - Falls back to files under /home/cursor/sample_media when unset
    """
    try:
        return test_all_services_flow()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------
# None in this module.
