from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.detector import detector_service
from app.services.tracker import camera_tracker_manager
from app.utils.cv import decode_image_bytes

router = APIRouter(prefix="/api/detect/camera", tags=["detect-camera"])


class CameraResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


@router.get("/status")
def camera_status() -> dict[str, object]:
    return {
        "status": "ready",
        "recommended_mode": "frontend_getUserMedia_then_post_camera_frame",
        "active_sessions": camera_tracker_manager.session_count(),
    }


@router.post("/session")
def create_camera_session() -> dict[str, str]:
    session_id = camera_tracker_manager.create_or_get_session(None)
    return {"session_id": session_id}


@router.post("/reset")
def reset_camera_session(payload: CameraResetRequest) -> dict[str, object]:
    ok = camera_tracker_manager.reset_session(payload.session_id)
    return {"ok": ok}


@router.post("/frame")
async def detect_camera_frame(
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    model_key: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    tracker_backend: str | None = Form(default=None),
    tracker_max_time_since_update: int | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detection_result = detector_service.predict_image(image, conf=conf, iou=iou, model_key=model_key)
    if detection_result.get("error"):
        raise HTTPException(status_code=500, detail=f"Inference failed: {detection_result['error']}")

    tracking_result = camera_tracker_manager.update_session(
        session_id=session_id,
        detections=detection_result["boxes"],
        frame_bgr=image,
        tracker_backend=tracker_backend,
        max_time_since_update_override=tracker_max_time_since_update,
    )

    return {
        "session_id": tracking_result["session_id"],
        "result": {
            "boxes": tracking_result["boxes"],
            "image_size": detection_result["image_size"],
            "model_key": detection_result["model_key"],
        },
        "tracking": {
            "trajectories": tracking_result["trajectories"],
            "track_summaries": tracking_result["track_summaries"],
            "tracker": tracking_result["tracker"],
            "deepsort_enabled": tracking_result["deepsort_enabled"],
        },
    }
