from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


_backend_dir = Path(__file__).resolve().parents[2]
_env_path = _backend_dir / ".env"
if load_dotenv is not None and _env_path.exists():
    load_dotenv(_env_path, override=True)


def _parse_csv_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_bool_env(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_requested_device(raw: str | None) -> str:
    value = (raw or "auto").strip()
    if not value:
        return "auto"

    normalized = value.lower()
    if normalized in {"gpu", "cuda"}:
        return "cuda:0"

    return value


def _resolve_backend_path(raw: str, backend_dir: Path) -> str:
    path_obj = Path(raw)
    if not path_obj.is_absolute():
        path_obj = (backend_dir / path_obj).resolve()
    return str(path_obj)


class Settings:
    def __init__(self) -> None:
        self.APP_DIR = Path(__file__).resolve().parents[1]
        self.BACKEND_DIR = self.APP_DIR.parent
        self.DATA_DIR = self.APP_DIR / "data"
        self.MODEL_DIR = self.DATA_DIR / "model"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        self.OUTPUT_DIR = self.DATA_DIR / "outputs"
        self.DB_PATH = self.DATA_DIR / "records.sqlite3"

        self.FRUIT_MODEL_DIR = self.MODEL_DIR / "fruit"
        self.VEGETABLE_MODEL_DIR = self.MODEL_DIR / "vegetable"

        default_model_path = self.FRUIT_MODEL_DIR / "best.pt"
        model_path_raw = os.getenv("MODEL_PATH", str(default_model_path))
        self.MODEL_PATH = _resolve_backend_path(model_path_raw, self.BACKEND_DIR)
        self.FRUIT_MODEL_PATH = _resolve_backend_path(
            os.getenv("FRUIT_MODEL_PATH", self.MODEL_PATH),
            self.BACKEND_DIR,
        )
        self.VEGETABLE_MODEL_PATH = _resolve_backend_path(
            os.getenv("VEGETABLE_MODEL_PATH", str(self.VEGETABLE_MODEL_DIR / "best.pt")),
            self.BACKEND_DIR,
        )
        self.DETECTION_MODEL_PATHS = {
            "fruit": self.FRUIT_MODEL_PATH,
            "vegetable": self.VEGETABLE_MODEL_PATH,
        }
        self.CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
        self.IOU_THRES = float(os.getenv("IOU_THRES", "0.45"))
        self.DETECTION_CLASS_ALLOWLIST = _parse_csv_env(os.getenv("DETECTION_CLASS_ALLOWLIST"))
        self.DEVICE = _normalize_requested_device(os.getenv("DEVICE", "cpu"))
        self.SAVE_EMPTY_RESULTS = _parse_bool_env(os.getenv("SAVE_EMPTY_RESULTS"), default=False)
        self.RECORD_RETENTION_LIMIT = int(os.getenv("RECORD_RETENTION_LIMIT", "50"))
        self.VIDEO_SAMPLE_INTERVAL = int(os.getenv("VIDEO_SAMPLE_INTERVAL", "10"))
        self.TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "deepsort").strip().lower() or "deepsort"
        self.TRACKER_MAX_AGE = int(os.getenv("TRACKER_MAX_AGE", "80"))
        self.TRACKER_N_INIT = int(os.getenv("TRACKER_N_INIT", "2"))
        self.TRACKER_MAX_IOU_DISTANCE = float(os.getenv("TRACKER_MAX_IOU_DISTANCE", "0.8"))
        self.TRACKER_MAX_COSINE_DISTANCE = float(os.getenv("TRACKER_MAX_COSINE_DISTANCE", "0.5"))
        self.TRACKER_NN_BUDGET = int(os.getenv("TRACKER_NN_BUDGET", "150"))
        self.TRACKER_MAX_TIME_SINCE_UPDATE = int(os.getenv("TRACKER_MAX_TIME_SINCE_UPDATE", "80"))
        self.TRACKER_TRAJECTORY_LEN = int(os.getenv("TRACKER_TRAJECTORY_LEN", "40"))
        self.TRACKER_RENDER_STALE_FRAMES = int(os.getenv("TRACKER_RENDER_STALE_FRAMES", "12"))
        self.TRACKER_FALLBACK_IOU = float(os.getenv("TRACKER_FALLBACK_IOU", "0.3"))
        self.TRACKER_FALLBACK_MAX_CENTER_DISTANCE_RATIO = float(
            os.getenv("TRACKER_FALLBACK_MAX_CENTER_DISTANCE_RATIO", "0.9")
        )
        self.TRACKER_FALLBACK_RECONNECT_DISTANCE_RATIO = float(
            os.getenv("TRACKER_FALLBACK_RECONNECT_DISTANCE_RATIO", "0.45")
        )
        self.TRACKER_FALLBACK_CENTER_DISTANCE_WEIGHT = float(
            os.getenv("TRACKER_FALLBACK_CENTER_DISTANCE_WEIGHT", "0.35")
        )
        self.CAMERA_TRACKER_SESSION_TTL_SEC = int(os.getenv("CAMERA_TRACKER_SESSION_TTL_SEC", "120"))
        self.CAMERA_IMGSZ = int(os.getenv("CAMERA_IMGSZ", "480"))
        self.CAMERA_CONF_THRES = float(os.getenv("CAMERA_CONF_THRES", "0.15"))
        self.CAMERA_IOU_THRES = float(os.getenv("CAMERA_IOU_THRES", "0.45"))
        self.CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
        self.CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
        self.CAMERA_FRAME_SKIP = int(os.getenv("CAMERA_FRAME_SKIP", "0"))
        self.CAMERA_SHOW_FPS = _parse_bool_env(os.getenv("CAMERA_SHOW_FPS"), default=True)
        self.CAMERA_DEBUG = _parse_bool_env(os.getenv("CAMERA_DEBUG"), default=False)
        self.CAMERA_SAVE_EVERY_FRAME = _parse_bool_env(os.getenv("CAMERA_SAVE_EVERY_FRAME"), default=False)
        self.CAMERA_USE_DEEPSORT = _parse_bool_env(os.getenv("CAMERA_USE_DEEPSORT"), default=True)
        self.CAMERA_DEVICE = _normalize_requested_device(os.getenv("CAMERA_DEVICE", "auto"))
        self.CAMERA_HALF = os.getenv("CAMERA_HALF", "auto").strip().lower()
        self.CAMERA_JPEG_QUALITY = int(os.getenv("CAMERA_JPEG_QUALITY", "75"))
        self.CAMERA_TRACKER_MAX_AGE = int(os.getenv("CAMERA_TRACKER_MAX_AGE", "80"))
        self.CAMERA_TRACKER_N_INIT = int(os.getenv("CAMERA_TRACKER_N_INIT", "2"))
        self.CAMERA_TRACKER_MAX_COSINE_DISTANCE = float(os.getenv("CAMERA_TRACKER_MAX_COSINE_DISTANCE", "0.5"))
        self.CAMERA_TRACKER_NN_BUDGET = int(os.getenv("CAMERA_TRACKER_NN_BUDGET", "150"))
        self.VIDEO_IMGSZ = int(os.getenv("VIDEO_IMGSZ", "512"))
        self.VIDEO_CONF_THRES = float(os.getenv("VIDEO_CONF_THRES", "0.30"))
        self.VIDEO_IOU_THRES = float(os.getenv("VIDEO_IOU_THRES", "0.45"))
        self.VIDEO_FRAME_SKIP = int(os.getenv("VIDEO_FRAME_SKIP", "1"))
        self.VIDEO_OUTPUT_WIDTH = int(os.getenv("VIDEO_OUTPUT_WIDTH", "1280"))
        self.VIDEO_OUTPUT_HEIGHT = int(os.getenv("VIDEO_OUTPUT_HEIGHT", "720"))
        self.VIDEO_KEEP_ORIGINAL_RESOLUTION = _parse_bool_env(
            os.getenv("VIDEO_KEEP_ORIGINAL_RESOLUTION"),
            default=False,
        )
        self.VIDEO_RESIZE_OUTPUT = _parse_bool_env(os.getenv("VIDEO_RESIZE_OUTPUT"), default=True)
        self.VIDEO_SHOW_STATS = _parse_bool_env(os.getenv("VIDEO_SHOW_STATS"), default=False)
        self.VIDEO_TRACKER_MAX_AGE = int(os.getenv("VIDEO_TRACKER_MAX_AGE", "80"))
        self.VIDEO_TRACKER_MAX_TIME_SINCE_UPDATE = int(os.getenv("VIDEO_TRACKER_MAX_TIME_SINCE_UPDATE", "0"))
        self.VIDEO_TRACKER_N_INIT = int(os.getenv("VIDEO_TRACKER_N_INIT", "2"))
        self.VIDEO_TRACKER_MAX_COSINE_DISTANCE = float(os.getenv("VIDEO_TRACKER_MAX_COSINE_DISTANCE", "0.5"))
        self.VIDEO_TRACKER_NN_BUDGET = int(os.getenv("VIDEO_TRACKER_NN_BUDGET", "150"))
        self.VIDEO_TRAIL_LENGTH = int(os.getenv("VIDEO_TRAIL_LENGTH", "40"))
        self.VIDEO_SMOOTH_WINDOW = int(os.getenv("VIDEO_SMOOTH_WINDOW", "1"))
        self.VIDEO_MIN_BOX_AREA = float(os.getenv("VIDEO_MIN_BOX_AREA", "300"))
        self.VIDEO_MAX_CENTER_JUMP = float(os.getenv("VIDEO_MAX_CENTER_JUMP", "220"))
        self.SMOOTHING_ENABLED = _parse_bool_env(os.getenv("SMOOTHING_ENABLED"), default=True)
        self.SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.65"))
        self.SMOOTHING_APPLY_TO_DISPLAY_ONLY = _parse_bool_env(
            os.getenv("SMOOTHING_APPLY_TO_DISPLAY_ONLY"),
            default=True,
        )
        self.SMOOTHING_MAX_CENTER_JUMP = float(os.getenv("SMOOTHING_MAX_CENTER_JUMP", "220"))

        self.CORS_ORIGINS = _parse_csv_env(os.getenv("CORS_ORIGINS")) or [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
        self.CORS_ALLOW_ORIGIN_REGEX = os.getenv(
            "CORS_ALLOW_ORIGIN_REGEX",
            r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$",
        )

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.FRUIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.VEGETABLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
