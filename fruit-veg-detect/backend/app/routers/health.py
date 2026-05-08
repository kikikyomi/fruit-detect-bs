from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.detector import detector_service
from app.services.tracker import get_deepsort_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    deepsort = get_deepsort_status()
    return {
        "status": "ok",
        "model_path": detector_service.model_path,
        "model_loaded": detector_service.model_loaded,
        "load_error": detector_service.load_error,
        "models": {
            key: {
                "path": detector_service.model_paths[key],
                "loaded": detector_service.model_loaded_by_key.get(key, False),
                "load_error": detector_service.load_errors.get(key),
            }
            for key in detector_service.model_paths
        },
        "requested_device": detector_service.requested_device,
        "runtime_device": detector_service.runtime_device,
        "detection_class_allowlist": settings.DETECTION_CLASS_ALLOWLIST,
        "tracker_backend_default": settings.TRACKER_BACKEND,
        "tracker_max_time_since_update_default": settings.TRACKER_MAX_TIME_SINCE_UPDATE,
        "tracker_render_stale_frames": settings.TRACKER_RENDER_STALE_FRAMES,
        "deepsort_available": deepsort["available"],
        "deepsort_error": deepsort["import_error"],
    }
