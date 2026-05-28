import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from prefect import flow, task

from job_shared import init_db, sqlite_db_path, upsert_media_file

router = APIRouter()

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".vob",
}
AUDITOK_EXTENSIONS = VIDEO_EXTENSIONS | {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".aiff",
    ".aif",
    ".wma",
    ".opus",
}


class RegisterMediaDirectoryRequest(BaseModel):
    directory: str = Field(..., description="Directory to scan for media files")


class RegisterMediaDirectoryResponse(BaseModel):
    directory: str
    total: int
    inserted: int
    updated: int
    files: list[str]


def _media_type_from_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "VID"
    if suffix in AUDITOK_EXTENSIONS:
        return "AUD"
    return None


@task(name="register-media-directory", retries=1, log_prints=True)
def register_media_directory(
    request: RegisterMediaDirectoryRequest,
) -> RegisterMediaDirectoryResponse:
    directory = Path(request.directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files_with_type: list[tuple[str, str]] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        media_type = _media_type_from_path(path)
        if media_type:
            files_with_type.append((str(path), media_type))

    if not files_with_type:
        raise ValueError(f"No supported media files found in: {directory}")

    inserted = 0
    updated = 0
    conn = sqlite3.connect(sqlite_db_path())
    try:
        init_db(conn)
        for file_path, media_type in files_with_type:
            existing = conn.execute(
                "SELECT 1 FROM media_files WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            upsert_media_file(conn, file_path, media_type)
            if existing:
                updated += 1
            else:
                inserted += 1
        conn.commit()
    finally:
        conn.close()

    return RegisterMediaDirectoryResponse(
        directory=str(directory),
        total=len(files_with_type),
        inserted=inserted,
        updated=updated,
        files=[file_path for file_path, _ in files_with_type],
    )


@flow(name="register-media-directory-flow", log_prints=True)
def register_media_directory_flow(
    request: RegisterMediaDirectoryRequest,
) -> RegisterMediaDirectoryResponse:
    return register_media_directory(request)


@router.post(
    "/api/media-files/register-directory",
    response_model=RegisterMediaDirectoryResponse,
)
async def api_register_media_directory(request: RegisterMediaDirectoryRequest):
    """Register all supported media files from a directory into media_files."""
    try:
        return register_media_directory_flow(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
