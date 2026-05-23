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
        "camera_defaults": {
            "imgsz": settings.CAMERA_IMGSZ,
            "conf": settings.CAMERA_CONF_THRES,
            "iou": settings.CAMERA_IOU_THRES,
            "width": settings.CAMERA_WIDTH,
            "height": settings.CAMERA_HEIGHT,
            "frame_skip": settings.CAMERA_FRAME_SKIP,
            "show_fps": settings.CAMERA_SHOW_FPS,
            "debug": settings.CAMERA_DEBUG,
            "save_every_frame": settings.CAMERA_SAVE_EVERY_FRAME,
            "use_deepsort": settings.CAMERA_USE_DEEPSORT,
            "device": settings.CAMERA_DEVICE,
            "half": settings.CAMERA_HALF,
            "max_age": settings.CAMERA_TRACKER_MAX_AGE,
            "n_init": settings.CAMERA_TRACKER_N_INIT,
            "max_iou_distance": settings.TRACKER_MAX_IOU_DISTANCE,
            "max_cosine_distance": settings.CAMERA_TRACKER_MAX_COSINE_DISTANCE,
            "nn_budget": settings.CAMERA_TRACKER_NN_BUDGET,
        },
        "video_defaults": {
            "imgsz": settings.VIDEO_IMGSZ,
            "conf": settings.VIDEO_CONF_THRES,
            "iou": settings.VIDEO_IOU_THRES,
            "frame_skip": settings.VIDEO_FRAME_SKIP,
            "output_width": settings.VIDEO_OUTPUT_WIDTH,
            "output_height": settings.VIDEO_OUTPUT_HEIGHT,
            "keep_original_resolution": settings.VIDEO_KEEP_ORIGINAL_RESOLUTION,
            "resize_output": settings.VIDEO_RESIZE_OUTPUT,
            "show_stats": settings.VIDEO_SHOW_STATS,
            "max_age": settings.VIDEO_TRACKER_MAX_AGE,
            "max_time_since_update": settings.VIDEO_TRACKER_MAX_TIME_SINCE_UPDATE,
            "n_init": settings.VIDEO_TRACKER_N_INIT,
            "max_iou_distance": settings.TRACKER_MAX_IOU_DISTANCE,
            "max_cosine_distance": settings.VIDEO_TRACKER_MAX_COSINE_DISTANCE,
            "nn_budget": settings.VIDEO_TRACKER_NN_BUDGET,
            "trail_length": settings.VIDEO_TRAIL_LENGTH,
            "smooth_window": settings.VIDEO_SMOOTH_WINDOW,
            "min_box_area": settings.VIDEO_MIN_BOX_AREA,
            "max_center_jump": settings.VIDEO_MAX_CENTER_JUMP,
        },
        "smoothing_defaults": {
            "enabled": settings.SMOOTHING_ENABLED,
            "alpha": settings.SMOOTHING_ALPHA,
            "apply_to_display_only": settings.SMOOTHING_APPLY_TO_DISPLAY_ONLY,
            "max_center_jump": settings.SMOOTHING_MAX_CENTER_JUMP,
        },
        "appearance_match_defaults": {
            "enabled": settings.TRACKER_APPEARANCE_MATCH_ENABLED,
            "min_score": settings.TRACKER_APPEARANCE_MIN_SCORE,
            "min_color_sim": settings.TRACKER_APPEARANCE_MIN_COLOR_SIM,
            "min_texture_sim": settings.TRACKER_APPEARANCE_MIN_TEXTURE_SIM,
            "max_age": settings.TRACKER_APPEARANCE_MAX_AGE,
            "max_center_distance_ratio": settings.TRACKER_APPEARANCE_MAX_CENTER_DISTANCE_RATIO,
            "color_weight": settings.TRACKER_APPEARANCE_COLOR_WEIGHT,
            "texture_weight": settings.TRACKER_APPEARANCE_TEXTURE_WEIGHT,
            "position_weight": settings.TRACKER_APPEARANCE_POSITION_WEIGHT,
        },
        "deepsort_available": deepsort["available"],
        "deepsort_error": deepsort["import_error"],
    }
