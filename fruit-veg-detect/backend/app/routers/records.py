from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.record_store import delete_record_with_files, get_record, list_records

router = APIRouter(prefix="/api/records", tags=["records"])


@router.get("/images")
def get_image_records(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> dict[str, object]:
    return list_records("image", page=page, page_size=page_size)


@router.get("/videos")
def get_video_records(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> dict[str, object]:
    return list_records("video", page=page, page_size=page_size)


@router.delete("/{record_id}")
def remove_record(record_id: int) -> dict[str, object]:
    if get_record(record_id) is None:
        raise HTTPException(status_code=404, detail="Record not found")

    ok = delete_record_with_files(record_id)
    return {"ok": ok}
