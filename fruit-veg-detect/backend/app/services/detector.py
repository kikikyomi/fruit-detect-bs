from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class DetectorService:
    DEFAULT_MODEL_KEY = "fruit"

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.model_paths: dict[str, str] = dict(settings.DETECTION_MODEL_PATHS)
        self.model_loaded_by_key: dict[str, bool] = {
            key: False for key in self.model_paths
        }
        self.load_errors: dict[str, str | None] = {
            key: None for key in self.model_paths
        }

        # Backward-compatible default model status used by /health.
        self.model: Any | None = None
        self.model_loaded = False
        self.model_path: str = self.model_paths[self.DEFAULT_MODEL_KEY]
        self.requested_device = settings.DEVICE
        self.runtime_device = self._resolve_runtime_device(self.requested_device)
        self.load_error: str | None = None

    def _resolve_runtime_device(self, requested_device: str) -> str:
        device = (requested_device or "auto").strip() or "auto"
        normalized = device.lower()

        if normalized == "auto":
            try:
                import torch
            except Exception:
                return "cpu"
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if normalized in {"gpu", "cuda"}:
            device = "cuda:0"
            normalized = device

        if normalized == "cpu":
            return "cpu"

        is_cuda_request = normalized.startswith("cuda")
        is_numeric_cuda_request = normalized.isdigit() or (
            "," in normalized and all(part.isdigit() for part in normalized.split(","))
        )
        if not is_cuda_request and not is_numeric_cuda_request:
            logger.warning("Unsupported device %s requested. Falling back to cpu.", device)
            return "cpu"

        try:
            import torch
        except Exception as exc:
            logger.warning(
                "Requested device %s but PyTorch is unavailable (%s). Falling back to cpu.",
                device,
                exc,
            )
            return "cpu"

        if not torch.cuda.is_available():
            logger.warning("Requested device %s but CUDA is unavailable. Falling back to cpu.", device)
            return "cpu"

        if ":" in normalized:
            raw_index = normalized.split(":", 1)[1]
            if raw_index.isdigit() and int(raw_index) >= torch.cuda.device_count():
                logger.warning(
                    "Requested device %s but only %s CUDA device(s) are available. Falling back to cpu.",
                    device,
                    torch.cuda.device_count(),
                )
                return "cpu"
        elif is_numeric_cuda_request:
            requested_indices = [int(part) for part in normalized.split(",")]
            if any(index >= torch.cuda.device_count() for index in requested_indices):
                logger.warning(
                    "Requested device %s but only %s CUDA device(s) are available. Falling back to cpu.",
                    device,
                    torch.cuda.device_count(),
                )
                return "cpu"

        return device

    def normalize_model_key(self, model_key: str | None) -> str:
        normalized = (model_key or self.DEFAULT_MODEL_KEY).strip().lower()
        if normalized in self.model_paths:
            return normalized
        logger.warning("Unsupported model key %s requested. Falling back to %s.", model_key, self.DEFAULT_MODEL_KEY)
        return self.DEFAULT_MODEL_KEY

    def _normalize_model_key(self, model_key: str | None) -> str:
        return self.normalize_model_key(model_key)

    def _set_model_status(self, model_key: str, loaded: bool, error: str | None) -> None:
        self.model_loaded_by_key[model_key] = loaded
        self.load_errors[model_key] = error

        if model_key == self.DEFAULT_MODEL_KEY:
            self.model_loaded = loaded
            self.load_error = error
            self.model = self.models.get(model_key)
            self.model_path = self.model_paths[model_key]

    def load_models(self) -> None:
        for model_key, model_path in self.model_paths.items():
            self.load_model(model_path, model_key=model_key)

    def load_model(self, model_path: str, model_key: str | None = None) -> bool:
        normalized_model_key = self._normalize_model_key(model_key)
        self.model_paths[normalized_model_key] = model_path
        self.requested_device = settings.DEVICE
        self.runtime_device = self._resolve_runtime_device(self.requested_device)
        path_obj = Path(model_path)

        if not path_obj.exists():
            self.models.pop(normalized_model_key, None)
            load_error = (
                f"Model file not found: {model_path}. "
                "System will run in mock mode with empty detections."
            )
            self._set_model_status(normalized_model_key, False, load_error)
            logger.warning("[%s] %s", normalized_model_key, load_error)
            return False

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            self.models.pop(normalized_model_key, None)
            load_error = (
                "ultralytics is not available. Install with `pip install ultralytics` "
                "and ensure MODEL_PATH points to a valid .pt file."
            )
            self._set_model_status(normalized_model_key, False, load_error)
            logger.warning("[%s] %s (%s)", normalized_model_key, load_error, exc)
            return False

        try:
            self.models[normalized_model_key] = YOLO(model_path)
            self._set_model_status(normalized_model_key, True, None)
            logger.info("Loaded %s model from %s", normalized_model_key, model_path)
            return True
        except Exception as exc:
            self.models.pop(normalized_model_key, None)
            load_error = f"Failed to load model from {model_path}: {exc}"
            self._set_model_status(normalized_model_key, False, load_error)
            logger.exception("[%s] %s", normalized_model_key, load_error)
            return False

    def _get_model(self, model_key: str | None) -> tuple[str, Any | None]:
        normalized_model_key = self._normalize_model_key(model_key)
        model = self.models.get(normalized_model_key)
        if model is None:
            model_path = self.model_paths[normalized_model_key]
            if Path(model_path).exists():
                self.load_model(model_path, model_key=normalized_model_key)
                model = self.models.get(normalized_model_key)
        return normalized_model_key, model

    def _resolve_class_name(self, model: Any | None, cls_id: int) -> str:
        if model is None:
            return str(cls_id)
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def _normalize_class_name(self, cls_name: str) -> str:
        return "".join(ch for ch in str(cls_name).casefold() if ch.isalnum())

    def _is_allowed_class(self, cls_name: str) -> bool:
        allowlist = settings.DETECTION_CLASS_ALLOWLIST
        if not allowlist:
            return True

        normalized_name = self._normalize_class_name(cls_name)
        return any(self._normalize_class_name(allowed) == normalized_name for allowed in allowlist)

    def predict_image(
        self,
        image_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        model_key: str | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        half: bool | None = None,
    ) -> dict[str, Any]:
        h, w = image_bgr.shape[:2]
        normalized_model_key, model = self._get_model(model_key)

        if conf is None:
            conf = settings.CONF_THRES
        if iou is None:
            iou = settings.IOU_THRES
        runtime_device = self._resolve_runtime_device(device if device is not None else self.requested_device)
        use_half = bool(half) and runtime_device != "cpu"

        if model is None:
            return {
                "boxes": [],
                "image_size": {"w": w, "h": h},
                "model_key": normalized_model_key,
                "device": runtime_device,
                "timings": {"yolo_ms": 0.0},
            }

        try:
            predict_kwargs: dict[str, Any] = {
                "source": image_bgr,
                "conf": conf,
                "iou": iou,
                "device": runtime_device,
                "half": use_half,
                "verbose": False,
            }
            if imgsz is not None:
                predict_kwargs["imgsz"] = int(imgsz)

            started_at = time.perf_counter()
            results = model.predict(**predict_kwargs)
            yolo_ms = (time.perf_counter() - started_at) * 1000.0
            first = results[0]
            parsed_boxes: list[dict[str, Any]] = []

            boxes = getattr(first, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf_score = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = self._resolve_class_name(model, cls_id)
                    if not self._is_allowed_class(cls_name):
                        continue
                    parsed_boxes.append(
                        {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                            "conf": conf_score,
                            "cls_id": cls_id,
                            "cls_name": cls_name,
                        }
                    )

            return {
                "boxes": parsed_boxes,
                "image_size": {"w": w, "h": h},
                "model_key": normalized_model_key,
                "device": runtime_device,
                "imgsz": int(imgsz) if imgsz is not None else None,
                "half": use_half,
                "timings": {"yolo_ms": round(yolo_ms, 3)},
            }
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            return {
                "boxes": [],
                "image_size": {"w": w, "h": h},
                "model_key": normalized_model_key,
                "device": runtime_device,
                "imgsz": int(imgsz) if imgsz is not None else None,
                "half": use_half,
                "timings": {"yolo_ms": 0.0},
                "error": str(exc),
            }

    def draw_boxes(self, image_bgr: np.ndarray, boxes: list[dict[str, Any]]) -> np.ndarray:
        return self.draw_boxes_with_trajectories(image_bgr, boxes, trajectories=None)

    def _color_for_track(self, track_id: int | None) -> tuple[int, int, int]:
        if track_id is None:
            return (255, 120, 0)
        digest = hashlib.md5(f"track-{track_id}".encode("utf-8")).digest()
        return (int(digest[0]), int(digest[1]), int(digest[2]))

    def draw_boxes_with_trajectories(
        self,
        image_bgr: np.ndarray,
        boxes: list[dict[str, Any]],
        trajectories: dict[int, list[tuple[int, int]]] | None = None,
    ) -> np.ndarray:
        output = image_bgr.copy()
        image_h, image_w = output.shape[:2]
        box_thickness = max(2, int(round(min(image_h, image_w) / 320)))
        text_scale = max(0.6, min(image_h, image_w) / 900.0)
        text_thickness = max(1, box_thickness - 1)

        if trajectories:
            for track_id, points in trajectories.items():
                if len(points) < 2:
                    continue
                polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    output,
                    [polyline],
                    False,
                    self._color_for_track(track_id),
                    box_thickness,
                    cv2.LINE_AA,
                )

        for box in boxes:
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            conf = float(box.get("conf", 0.0))
            cls_name = str(box.get("cls_name", "obj"))
            track_id_raw = box.get("track_id")
            track_id = int(track_id_raw) if track_id_raw is not None else None

            if track_id is None:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = f"ID {track_id} {cls_name} {conf:.2f}"
            color = self._color_for_track(track_id)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            y_text = max(y1 - 8, 0)
            cv2.rectangle(output, (x1, y_text - th - baseline), (x1 + tw + 4, y_text + baseline), color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return output

detector_service = DetectorService()
