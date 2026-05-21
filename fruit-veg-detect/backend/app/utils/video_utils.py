from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float


@dataclass(frozen=True)
class OutputLayout:
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int


def read_video_info(capture: cv2.VideoCapture) -> VideoInfo:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    return VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames, duration=duration)


def resolve_output_layout(
    input_width: int,
    input_height: int,
    output_width: int | None = 1280,
    output_height: int | None = 720,
    keep_original_resolution: bool = False,
    resize_output: bool = True,
) -> OutputLayout:
    if input_width <= 0 or input_height <= 0:
        raise ValueError("Invalid input video resolution")

    if keep_original_resolution or not resize_output:
        target_w = int(input_width)
        target_h = int(input_height)
    else:
        target_w = int(output_width or 1280)
        target_h = int(output_height or 720)

    target_w = max(2, target_w)
    target_h = max(2, target_h)
    scale = min(target_w / input_width, target_h / input_height)
    resized_w = max(1, int(round(input_width * scale)))
    resized_h = max(1, int(round(input_height * scale)))
    pad_x = int((target_w - resized_w) // 2)
    pad_y = int((target_h - resized_h) // 2)

    return OutputLayout(
        input_width=input_width,
        input_height=input_height,
        output_width=target_w,
        output_height=target_h,
        scale=float(scale),
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_w,
        resized_height=resized_h,
    )


def resize_frame_keep_ratio(frame_bgr: np.ndarray, layout: OutputLayout) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (layout.resized_width, layout.resized_height), interpolation=cv2.INTER_LINEAR)
    if layout.resized_width == layout.output_width and layout.resized_height == layout.output_height:
        return resized

    canvas = np.zeros((layout.output_height, layout.output_width, 3), dtype=frame_bgr.dtype)
    y2 = layout.pad_y + layout.resized_height
    x2 = layout.pad_x + layout.resized_width
    canvas[layout.pad_y:y2, layout.pad_x:x2] = resized
    return canvas


def scale_point_to_output(point: tuple[float, float], layout: OutputLayout) -> tuple[int, int]:
    x = int(round(float(point[0]) * layout.scale + layout.pad_x))
    y = int(round(float(point[1]) * layout.scale + layout.pad_y))
    return (
        max(0, min(layout.output_width - 1, x)),
        max(0, min(layout.output_height - 1, y)),
    )


def scale_box_to_output(box: dict[str, Any], layout: OutputLayout) -> dict[str, Any]:
    x1 = float(box["x1"]) * layout.scale + layout.pad_x
    y1 = float(box["y1"]) * layout.scale + layout.pad_y
    x2 = float(box["x2"]) * layout.scale + layout.pad_x
    y2 = float(box["y2"]) * layout.scale + layout.pad_y
    mapped = dict(box)
    mapped["x1"] = max(0.0, min(float(layout.output_width - 1), x1))
    mapped["y1"] = max(0.0, min(float(layout.output_height - 1), y1))
    mapped["x2"] = max(0.0, min(float(layout.output_width - 1), x2))
    mapped["y2"] = max(0.0, min(float(layout.output_height - 1), y2))
    return mapped


def map_original_boxes_to_output(boxes: list[dict[str, Any]], layout: OutputLayout) -> list[dict[str, Any]]:
    return [scale_box_to_output(box, layout) for box in boxes]


def map_original_trajectories_to_output(
    trajectories: dict[int, list[tuple[int, int]]],
    layout: OutputLayout,
) -> dict[int, list[tuple[int, int]]]:
    return {
        int(track_id): [scale_point_to_output((x, y), layout) for x, y in points]
        for track_id, points in trajectories.items()
    }


def box_xywh(box: dict[str, Any]) -> tuple[float, float, float, float]:
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def box_center(box: dict[str, Any]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2.0, (float(box["y1"]) + float(box["y2"])) / 2.0)


def box_area(box: dict[str, Any]) -> float:
    _, _, w, h = box_xywh(box)
    return float(w * h)


def compute_frame_range(
    info: VideoInfo,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[int, int]:
    fps = max(1e-6, float(info.fps))
    start_frame = 0 if start_time is None else max(0, int(math.floor(float(start_time) * fps)))
    if end_time is None or float(end_time) <= 0:
        end_frame = info.total_frames
    else:
        end_frame = int(math.ceil(float(end_time) * fps))
        if info.total_frames > 0:
            end_frame = min(info.total_frames, end_frame)
    if info.total_frames > 0:
        start_frame = min(start_frame, info.total_frames)
    end_frame = max(start_frame, end_frame)
    return start_frame, end_frame


def parse_classes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    classes: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        classes.append(int(text))
    return classes or None


def bbox_json(box: dict[str, Any]) -> str:
    return json.dumps(
        [
            round(float(box["x1"]), 3),
            round(float(box["y1"]), 3),
            round(float(box["x2"]), 3),
            round(float(box["y2"]), 3),
        ],
        ensure_ascii=False,
    )


def bbox_xywh_json(box: dict[str, Any]) -> str:
    x, y, w, h = box_xywh(box)
    return json.dumps([round(x, 3), round(y, 3), round(w, 3), round(h, 3)], ensure_ascii=False)
