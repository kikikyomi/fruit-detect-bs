from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.detector import detector_service
from app.services.tracker import camera_tracker_manager
from app.utils.cv import decode_image_bytes

router = APIRouter(prefix="/api/detect/camera", tags=["detect-camera"])


class CameraResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


def _resolve_camera_half(raw_half: bool | None, device: str) -> bool:
    if raw_half is not None:
        return bool(raw_half)

    configured = settings.CAMERA_HALF
    if configured in {"1", "true", "yes", "on"}:
        return True
    if configured in {"0", "false", "no", "off"}:
        return False

    runtime_device = detector_service._resolve_runtime_device(device)
    return runtime_device != "cpu"


@router.get("/status")
def camera_status() -> dict[str, object]:
    return {
        "status": "ready",
        "recommended_mode": "frontend_getUserMedia_then_post_camera_frame",
        "active_sessions": camera_tracker_manager.session_count(),
        "defaults": {
            "imgsz": settings.CAMERA_IMGSZ,
            "conf": settings.CAMERA_CONF_THRES,
            "iou": settings.CAMERA_IOU_THRES,
            "camera_width": settings.CAMERA_WIDTH,
            "camera_height": settings.CAMERA_HEIGHT,
            "frame_skip": settings.CAMERA_FRAME_SKIP,
            "jpeg_quality": settings.CAMERA_JPEG_QUALITY,
            "max_age": settings.CAMERA_TRACKER_MAX_AGE,
            "n_init": settings.CAMERA_TRACKER_N_INIT,
            "max_cosine_distance": settings.CAMERA_TRACKER_MAX_COSINE_DISTANCE,
            "nn_budget": settings.CAMERA_TRACKER_NN_BUDGET,
        },
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
    imgsz: int | None = Form(default=None),
    device: str | None = Form(default=None),
    half: bool | None = Form(default=None),
    model_key: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    tracker_backend: str | None = Form(default=None),
    tracker_max_time_since_update: int | None = Form(default=None),
    max_age: int | None = Form(default=None),
    n_init: int | None = Form(default=None),
    max_cosine_distance: float | None = Form(default=None),
    nn_budget: int | None = Form(default=None),
    disable_deepsort: bool = Form(default=False),
    save: bool = Form(default=False),
    save_video: bool = Form(default=False),
    save_csv: bool = Form(default=False),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    total_started_at = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    active_imgsz = int(imgsz if imgsz is not None else settings.CAMERA_IMGSZ)
    active_device = device or settings.CAMERA_DEVICE
    active_half = _resolve_camera_half(half, active_device)
    active_conf = float(conf if conf is not None else settings.CAMERA_CONF_THRES)
    active_iou = float(iou if iou is not None else settings.CAMERA_IOU_THRES)

    detection_result = detector_service.predict_image(
        image,
        conf=active_conf,
        iou=active_iou,
        model_key=model_key,
        imgsz=active_imgsz,
        device=active_device,
        half=active_half,
    )
    if detection_result.get("error"):
        raise HTTPException(status_code=500, detail=f"Inference failed: {detection_result['error']}")

    tracker_started_at = time.perf_counter()
    if disable_deepsort:
        tracking_result = {
            "session_id": session_id or "",
            "boxes": detection_result["boxes"],
            "trajectories": {},
            "track_summaries": [],
            "tracker": "disabled-yolo-only",
            "deepsort_enabled": False,
            "tracker_config": {},
        }
    else:
        tracking_result = camera_tracker_manager.update_session(
            session_id=session_id,
            detections=detection_result["boxes"],
            frame_bgr=image,
            tracker_backend=tracker_backend,
            max_time_since_update_override=tracker_max_time_since_update,
            max_age_override=max_age,
            n_init_override=n_init,
            max_cosine_distance_override=max_cosine_distance,
            nn_budget_override=nn_budget,
        )
    deepsort_ms = (time.perf_counter() - tracker_started_at) * 1000.0
    total_ms = (time.perf_counter() - total_started_at) * 1000.0

    return {
        "session_id": tracking_result["session_id"],
        "result": {
            "boxes": tracking_result["boxes"],
            "image_size": detection_result["image_size"],
            "model_key": detection_result["model_key"],
            "imgsz": detection_result.get("imgsz"),
            "device": detection_result.get("device"),
            "half": detection_result.get("half"),
        },
        "tracking": {
            "trajectories": tracking_result["trajectories"],
            "track_summaries": tracking_result["track_summaries"],
            "tracker": tracking_result["tracker"],
            "deepsort_enabled": tracking_result["deepsort_enabled"],
            "tracker_config": tracking_result.get("tracker_config", {}),
        },
        "performance": {
            "total_ms": round(total_ms, 3),
            "yolo_ms": detection_result.get("timings", {}).get("yolo_ms", 0.0),
            "deepsort_ms": round(deepsort_ms, 3),
        },
        "saved": False,
        "save_requested": {
            "save": save,
            "save_video": save_video,
            "save_csv": save_csv,
        },
    }
