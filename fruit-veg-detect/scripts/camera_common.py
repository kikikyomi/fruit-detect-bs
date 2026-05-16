from __future__ import annotations

import csv
import hashlib
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def resolve_weights_path(raw: str) -> str:
    path = Path(raw)
    candidates = [
        path,
        PROJECT_ROOT / raw,
        BACKEND_DIR / raw,
        BACKEND_DIR / "app" / "data" / "model" / "fruit" / raw,
        BACKEND_DIR / "app" / "data" / "model" / "vegetable" / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def resolve_device(requested: str | None) -> tuple[str, bool, str]:
    requested = (requested or "auto").strip() or "auto"
    normalized = requested.lower()

    if normalized == "cpu":
        return "cpu", False, "CPU"

    try:
        import torch
    except Exception:
        print("PyTorch is unavailable. Falling back to CPU; FPS will be much lower.")
        return "cpu", False, "CPU"

    cuda_available = bool(torch.cuda.is_available())
    if normalized == "auto":
        if cuda_available:
            return "0", True, "GPU"
        print("CUDA is unavailable. Running on CPU; FPS will be much lower.")
        return "cpu", False, "CPU"

    is_cuda_request = normalized in {"cuda", "gpu"} or normalized.startswith("cuda") or normalized.isdigit()
    if is_cuda_request and cuda_available:
        if normalized in {"cuda", "gpu", "cuda:0"}:
            return "0", True, "GPU"
        if normalized.startswith("cuda:"):
            return normalized.split(":", 1)[1], True, "GPU"
        return requested, True, "GPU"

    if is_cuda_request:
        print(f"Requested device {requested}, but CUDA is unavailable. Falling back to CPU; FPS will be much lower.")
        return "cpu", False, "CPU"

    print(f"Unsupported device {requested}. Falling back to CPU; FPS will be much lower.")
    return "cpu", False, "CPU"


def resolve_half(raw_half: bool | None, device_is_cuda: bool) -> bool:
    if raw_half is None:
        return device_is_cuda
    return bool(raw_half and device_is_cuda)


def parse_source(raw: str) -> int | str:
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    return text


def open_capture(source: str, camera_width: int, camera_height: int) -> cv2.VideoCapture:
    capture_source = parse_source(source)
    is_camera = isinstance(capture_source, int)
    if is_camera and platform.system().lower().startswith("win"):
        cap = cv2.VideoCapture(capture_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(capture_source)

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def load_yolo(weights: str) -> Any:
    from ultralytics import YOLO  # type: ignore

    return YOLO(resolve_weights_path(weights))


def _class_name(model: Any, cls_id: int) -> str:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def predict_boxes(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.perf_counter()
    results = model.predict(
        frame_bgr,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device=device,
        half=bool(half),
        verbose=False,
    )
    yolo_ms = (time.perf_counter() - started_at) * 1000.0

    first = results[0]
    parsed: list[dict[str, Any]] = []
    boxes = getattr(first, "boxes", None)
    if boxes is None:
        return parsed, yolo_ms

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf_score = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        parsed.append(
            {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "conf": conf_score,
                "cls_id": cls_id,
                "cls_name": _class_name(model, cls_id),
            }
        )
    return parsed, yolo_ms


def color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 120, 0)
    digest = hashlib.md5(f"track-{track_id}".encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    trajectories: dict[int, list[tuple[int, int]]] | None = None,
) -> float:
    started_at = time.perf_counter()
    image_h, image_w = frame_bgr.shape[:2]
    box_thickness = max(1, int(round(min(image_h, image_w) / 360)))
    text_scale = max(0.45, min(image_h, image_w) / 1000.0)
    text_thickness = max(1, box_thickness)

    if trajectories:
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue
            polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [polyline], False, color_for_track(track_id), box_thickness, cv2.LINE_AA)

    for box in boxes:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        track_id_raw = box.get("track_id")
        track_id = int(track_id_raw) if track_id_raw is not None else None
        color = color_for_track(track_id)
        label_prefix = f"ID {track_id} " if track_id is not None else ""
        label = f"{label_prefix}{box.get('cls_name', 'obj')} {float(box.get('conf', 0.0)):.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, box_thickness)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        y_text = max(y1 - 6, th + baseline + 2)
        cv2.rectangle(frame_bgr, (x1, y_text - th - baseline - 2), (x1 + tw + 4, y_text + baseline), color, -1)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return (time.perf_counter() - started_at) * 1000.0


class SlidingAverage:
    def __init__(self, size: int = 30) -> None:
        self.values: deque[float] = deque(maxlen=max(1, size))

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def average(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))


@dataclass
class FrameMetrics:
    frame_index: int
    total_ms: float
    yolo_ms: float
    deepsort_ms: float
    draw_ms: float
    fps: float
    detections: int
    tracks: int
    detection_frame: bool


def print_metrics(metrics: FrameMetrics) -> None:
    print(
        "frame={frame} total_fps={fps:.2f} total_ms={total:.2f} "
        "yolo_ms={yolo:.2f} deepsort_ms={deepsort:.2f} draw_ms={draw:.2f} "
        "detections={detections} tracks={tracks} detect_frame={detect_frame}".format(
            frame=metrics.frame_index,
            fps=metrics.fps,
            total=metrics.total_ms,
            yolo=metrics.yolo_ms,
            deepsort=metrics.deepsort_ms,
            draw=metrics.draw_ms,
            detections=metrics.detections,
            tracks=metrics.tracks,
            detect_frame=int(metrics.detection_frame),
        ),
        flush=True,
    )


class CsvMetricWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "frame_index",
                "total_ms",
                "yolo_ms",
                "deepsort_ms",
                "draw_ms",
                "fps",
                "detections",
                "tracks",
                "detection_frame",
            ],
        )
        self._writer.writeheader()

    def write(self, metrics: FrameMetrics) -> None:
        self._writer.writerow(
            {
                "frame_index": metrics.frame_index,
                "total_ms": f"{metrics.total_ms:.3f}",
                "yolo_ms": f"{metrics.yolo_ms:.3f}",
                "deepsort_ms": f"{metrics.deepsort_ms:.3f}",
                "draw_ms": f"{metrics.draw_ms:.3f}",
                "fps": f"{metrics.fps:.3f}",
                "detections": metrics.detections,
                "tracks": metrics.tracks,
                "detection_frame": int(metrics.detection_frame),
            }
        )

    def close(self) -> None:
        self._file.close()


def build_video_tracker(args: Any) -> Any:
    from app.services.tracker import VideoTracker

    return VideoTracker(
        backend_override="deepsort",
        n_init_override=int(args.n_init),
        max_age_override=int(args.max_age),
        max_time_since_update_override=int(args.max_age),
        max_cosine_distance_override=float(args.max_cosine_distance),
        nn_budget_override=int(args.nn_budget),
        device_override=str(args.device),
    )


def iter_existing_trajectories(tracker: Any) -> dict[int, list[tuple[int, int]]]:
    raw = tracker.get_trajectories(active_within_frames=getattr(tracker, "max_time_since_update", None))
    return {int(track_id): points for track_id, points in raw.items()}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
