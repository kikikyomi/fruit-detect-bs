from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.utils.video_utils import (
    OutputLayout,
    bbox_json,
    bbox_xywh_json,
    box_area,
    box_center,
    box_xywh,
    compute_frame_range,
    map_original_boxes_to_output,
    read_video_info,
    resize_frame_keep_ratio,
    resolve_output_layout,
)

PredictFn = Callable[[np.ndarray], tuple[list[dict[str, Any]], float]]
ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class VideoProcessOptions:
    source_path: str
    run_dir: Path
    weights_path: str
    model_name: str
    imgsz: int = 512
    conf: float = 0.30
    iou: float = 0.45
    device: str = "auto"
    half: bool = False
    frame_skip: int = 1
    output_width: int = 1280
    output_height: int = 720
    keep_original_resolution: bool = False
    resize_output: bool = True
    show_stats: bool = False
    start_time: float | None = None
    end_time: float | None = None
    save_video: bool = True
    save_csv: bool = True
    save_json: bool = True
    enable_deepsort: bool = False
    tracker_max_time_since_update: int = 0
    trail_length: int = 30
    smooth_window: int = 5
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.65
    min_box_area: float = 300.0
    max_center_jump: float = 160.0
    debug: bool = False


@dataclass
class VideoProcessResult:
    run_dir: Path
    output_video_path: Path | None
    result_csv_path: Path | None
    result_json_path: Path | None
    summary_path: Path
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def _color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 120, 0)
    value = (track_id * 2654435761) & 0xFFFFFF
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)


def _draw_annotations(
    frame_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    trajectories: dict[int, list[tuple[int, int] | None]],
    *,
    show_stats: bool,
    stats: dict[str, Any],
) -> float:
    started_at = time.perf_counter()
    image_h, image_w = frame_bgr.shape[:2]
    box_thickness = max(1, int(round(min(image_h, image_w) / 360)))
    text_scale = max(0.45, min(image_h, image_w) / 1000.0)
    text_thickness = max(1, box_thickness)

    trail_thickness = 2
    for track_id, points in trajectories.items():
        segment: list[tuple[int, int]] = []
        for point in points:
            if point is None:
                if len(segment) >= 2:
                    polyline = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [polyline], False, _color_for_track(track_id), trail_thickness, cv2.LINE_AA)
                segment = []
                continue
            segment.append(point)
        if len(segment) >= 2:
            polyline = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [polyline], False, _color_for_track(track_id), trail_thickness, cv2.LINE_AA)

    for box in boxes:
        x1 = int(round(float(box["x1"])))
        y1 = int(round(float(box["y1"])))
        x2 = int(round(float(box["x2"])))
        y2 = int(round(float(box["y2"])))
        track_id_raw = box.get("track_id")
        track_id = int(track_id_raw) if track_id_raw is not None else None
        color = _color_for_track(track_id)
        prefix = f"ID {track_id} " if track_id is not None else ""
        label = f"{prefix}{box.get('cls_name', 'obj')} {float(box.get('conf', 0.0)):.2f}"

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

    if show_stats:
        lines = [
            f"FPS {float(stats.get('fps', 0.0)):.2f}",
            f"Frame {int(stats.get('frame_id', 0))}/{int(stats.get('end_frame', 0))}",
            f"Progress {float(stats.get('progress', 0.0)):.1f}%",
        ]
        y = 24
        for line in lines:
            cv2.putText(frame_bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 255, 255), 2, cv2.LINE_AA)
            y += 24

    return (time.perf_counter() - started_at) * 1000.0


def _filter_boxes(boxes: list[dict[str, Any]], min_box_area: float) -> list[dict[str, Any]]:
    if min_box_area <= 0:
        return boxes
    return [box for box in boxes if box_area(box) >= min_box_area]


def _smooth_tracked_boxes(
    boxes: list[dict[str, Any]],
    history: dict[int, deque[dict[str, Any]]],
    last_center_by_track: dict[int, tuple[float, float]],
    *,
    smooth_window: int,
    max_center_jump: float,
) -> list[dict[str, Any]]:
    if smooth_window <= 1 and max_center_jump <= 0:
        return boxes

    smoothed: list[dict[str, Any]] = []
    for box in boxes:
        track_id_raw = box.get("track_id")
        if track_id_raw is None:
            smoothed.append(box)
            continue

        track_id = int(track_id_raw)
        center = box_center(box)
        last_center_by_track[track_id] = center
        items = history.setdefault(track_id, deque(maxlen=max(1, smooth_window)))
        items.append(dict(box))

        if smooth_window <= 1:
            smoothed.append(box)
            continue

        averaged = dict(box)
        averaged["x1"] = sum(float(item["x1"]) for item in items) / len(items)
        averaged["y1"] = sum(float(item["y1"]) for item in items) / len(items)
        averaged["x2"] = sum(float(item["x2"]) for item in items) / len(items)
        averaged["y2"] = sum(float(item["y2"]) for item in items) / len(items)
        smoothed.append(averaged)
    return smoothed


def _row_for_box(
    *,
    frame_id: int,
    timestamp: float,
    is_detected_frame: bool,
    box_original: dict[str, Any],
    box_output: dict[str, Any],
    include_tracking: bool,
) -> dict[str, Any]:
    x, y, w, h = box_xywh(box_original)
    out_x, out_y, out_w, out_h = box_xywh(box_output)
    center_x, center_y = box_center(box_original)
    out_center_x, out_center_y = box_center(box_output)
    row: dict[str, Any] = {
        "frame_id": frame_id,
        "timestamp": round(timestamp, 6),
        "is_detected_frame": bool(is_detected_frame),
        "class_id": int(box_original.get("cls_id", -1)),
        "class_name": str(box_original.get("cls_name", "obj")),
        "confidence": round(float(box_original.get("conf", 0.0)), 6),
        "bbox_xyxy": bbox_json(box_original),
        "bbox_xywh": bbox_xywh_json(box_original),
        "x1": round(float(box_original["x1"]), 3),
        "y1": round(float(box_original["y1"]), 3),
        "x2": round(float(box_original["x2"]), 3),
        "y2": round(float(box_original["y2"]), 3),
        "width": round(w, 3),
        "height": round(h, 3),
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "output_bbox_xyxy": bbox_json(box_output),
        "output_x1": round(float(box_output["x1"]), 3),
        "output_y1": round(float(box_output["y1"]), 3),
        "output_x2": round(float(box_output["x2"]), 3),
        "output_y2": round(float(box_output["y2"]), 3),
        "output_width": round(out_w, 3),
        "output_height": round(out_h, 3),
        "output_center_x": round(out_center_x, 3),
        "output_center_y": round(out_center_y, 3),
    }
    if include_tracking:
        track_id = box_original.get("track_id")
        row["track_id"] = int(track_id) if track_id is not None else ""
    return row


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "timestamp",
        "is_detected_frame",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "bbox_xyxy",
        "bbox_xywh",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
        "output_bbox_xyxy",
        "output_x1",
        "output_y1",
        "output_x2",
        "output_y2",
        "output_width",
        "output_height",
        "output_center_x",
        "output_center_y",
    ]
    if any("track_id" in row for row in rows):
        pass
    else:
        fieldnames.remove("track_id")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"{key}: {value}" for key, value in summary.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def process_video_file(
    options: VideoProcessOptions,
    predict_fn: PredictFn,
    tracker: Any | None = None,
    progress_fn: ProgressFn | None = None,
) -> VideoProcessResult:
    options.run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(options.source_path)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    info = read_video_info(cap)
    if info.width <= 0 or info.height <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read video resolution: {source_path}")

    start_frame, end_frame = compute_frame_range(info, options.start_time, options.end_time)
    if info.total_frames > 0 and start_frame >= info.total_frames:
        cap.release()
        raise RuntimeError("start_time is outside the input video duration")
    if end_frame <= start_frame:
        cap.release()
        raise RuntimeError("No frames selected for processing")

    layout = resolve_output_layout(
        info.width,
        info.height,
        output_width=options.output_width,
        output_height=options.output_height,
        keep_original_resolution=options.keep_original_resolution,
        resize_output=options.resize_output,
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_video_path = options.run_dir / "output.mp4" if options.save_video else None
    writer: cv2.VideoWriter | None = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            float(info.fps),
            (layout.output_width, layout.output_height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open video writer: {output_video_path}")

    frame_skip = max(1, int(options.frame_skip))
    total_expected = max(0, end_frame - start_frame)
    total_started_at = time.perf_counter()
    frame_id = start_frame
    processed_frames = 0
    detected_frames = 0
    total_detections = 0
    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = defaultdict(int)
    track_ids_by_class: dict[str, set[int]] = defaultdict(set)
    yolo_times: list[float] = []
    deepsort_times: list[float] = []
    draw_times: list[float] = []
    read_times: list[float] = []
    write_times: list[float] = []
    last_detection_boxes: list[dict[str, Any]] = []
    display_boxes_original: list[dict[str, Any]] = []
    output_trails: dict[int, deque[tuple[int, int] | None]] = defaultdict(
        lambda: deque(maxlen=max(1, int(options.trail_length)))
    )
    track_last_seen_frame: dict[int, int] = {}
    last_output_center_by_track: dict[int, tuple[int, int]] = {}

    try:
        while frame_id < end_frame:
            read_started_at = time.perf_counter()
            ok, frame = cap.read()
            read_times.append((time.perf_counter() - read_started_at) * 1000.0)
            if not ok:
                break

            frame_started_at = time.perf_counter()
            is_detected_frame = (frame_id - start_frame) % frame_skip == 0
            detection_boxes: list[dict[str, Any]] = []
            yolo_ms = 0.0
            if is_detected_frame:
                detection_boxes, yolo_ms = predict_fn(frame)
                detection_boxes = _filter_boxes(detection_boxes, float(options.min_box_area))
                yolo_times.append(yolo_ms)
                last_detection_boxes = [dict(box) for box in detection_boxes]
                detected_frames += 1
                total_detections += len(detection_boxes)
                for box in detection_boxes:
                    class_counts[str(box.get("cls_name", box.get("cls_id", "obj")))] += 1

            if options.enable_deepsort and tracker is not None:
                if hasattr(tracker, "debug"):
                    tracker.debug = bool(options.debug)
                track_started_at = time.perf_counter()
                tracked_boxes = tracker.update(detection_boxes if is_detected_frame else [], frame, frame_id)
                deepsort_times.append((time.perf_counter() - track_started_at) * 1000.0)
                display_boxes_original = tracked_boxes
                if is_detected_frame and not detection_boxes and not tracked_boxes:
                    display_boxes_original = []
            else:
                display_boxes_original = detection_boxes if is_detected_frame else last_detection_boxes

            output_frame = resize_frame_keep_ratio(frame, layout)
            output_boxes = map_original_boxes_to_output(display_boxes_original, layout)

            active_track_ids: set[int] = set()
            if options.enable_deepsort:
                for original_box, output_box in zip(display_boxes_original, output_boxes):
                    track_id_raw = original_box.get("track_id")
                    if track_id_raw is None:
                        continue
                    track_id = int(track_id_raw)
                    active_track_ids.add(track_id)
                    track_ids_by_class[str(original_box.get("cls_name", "obj"))].add(track_id)
                    if float(original_box.get("conf", 0.0)) < float(options.conf):
                        track_last_seen_frame[track_id] = frame_id
                        continue
                    center = box_center(output_box)
                    point = (int(round(center[0])), int(round(center[1])))
                    previous_point = last_output_center_by_track.get(track_id)
                    if (
                        bool(options.smoothing_enabled)
                        and previous_point is not None
                        and float(options.max_center_jump) > 0
                    ):
                        jump = math.hypot(point[0] - previous_point[0], point[1] - previous_point[1])
                        if jump > float(options.max_center_jump):
                            if bool(options.debug):
                                print(
                                    f"[Smoothing] large jump ignored: track_id={track_id}, jump={jump:.2f}",
                                    flush=True,
                                )
                            if not output_trails[track_id] or output_trails[track_id][-1] is not None:
                                output_trails[track_id].append(None)
                            last_output_center_by_track[track_id] = point
                            track_last_seen_frame[track_id] = frame_id
                            continue
                    if not output_trails[track_id] or output_trails[track_id][-1] != point:
                        output_trails[track_id].append(point)
                    last_output_center_by_track[track_id] = point
                    track_last_seen_frame[track_id] = frame_id

            for track_id in list(output_trails.keys()):
                if frame_id - track_last_seen_frame.get(track_id, frame_id) > max(1, int(getattr(tracker, "max_age", options.trail_length))):
                    output_trails.pop(track_id, None)
                    track_last_seen_frame.pop(track_id, None)
                    last_output_center_by_track.pop(track_id, None)

            draw_ms = _draw_annotations(
                output_frame,
                output_boxes,
                {track_id: list(points) for track_id, points in output_trails.items()},
                show_stats=bool(options.show_stats),
                stats={
                    "fps": processed_frames / max(1e-6, time.perf_counter() - total_started_at),
                    "frame_id": frame_id,
                    "end_frame": end_frame,
                    "progress": (processed_frames + 1) * 100.0 / max(1, total_expected),
                },
            )
            draw_times.append(draw_ms)

            timestamp = float(frame_id / max(1e-6, info.fps))
            for original_box, output_box in zip(display_boxes_original, output_boxes):
                rows.append(
                    _row_for_box(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        is_detected_frame=is_detected_frame,
                        box_original=original_box,
                        box_output=output_box,
                        include_tracking=bool(options.enable_deepsort),
                    )
                )

            if writer is not None:
                write_started_at = time.perf_counter()
                writer.write(output_frame)
                write_times.append((time.perf_counter() - write_started_at) * 1000.0)

            processed_frames += 1
            elapsed = time.perf_counter() - total_started_at
            if progress_fn is not None:
                progress_fn(
                    {
                        "status": "running",
                        "progress": round(processed_frames * 100.0 / max(1, total_expected), 2),
                        "current_frame": frame_id,
                        "total_frames": total_expected,
                        "fps": round(processed_frames / max(1e-6, elapsed), 3),
                        "message": "processing video",
                    }
                )

            frame_id += 1
            _ = frame_started_at
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    total_seconds = max(1e-6, time.perf_counter() - total_started_at)
    track_count = sum(len(ids) for ids in track_ids_by_class.values())
    if tracker is not None and hasattr(tracker, "get_track_summaries"):
        try:
            summaries = tracker.get_track_summaries()
            track_count = len(summaries)
        except Exception:
            pass

    result_csv_path = options.run_dir / "results.csv" if options.save_csv else None
    result_json_path = options.run_dir / "results.json" if options.save_json else None
    summary_path = options.run_dir / "summary.txt"

    if result_csv_path is not None:
        _write_rows_csv(result_csv_path, rows)
    if result_json_path is not None:
        result_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "input_video_path": str(source_path),
        "output_video_path": str(output_video_path) if output_video_path else "",
        "weights_path": options.weights_path,
        "model_name": options.model_name,
        "original_total_frames": info.total_frames,
        "processed_start_frame": start_frame,
        "processed_end_frame": frame_id,
        "processed_frames": processed_frames,
        "original_fps": round(float(info.fps), 3),
        "output_fps": round(float(info.fps), 3),
        "original_resolution": f"{info.width}x{info.height}",
        "output_resolution": f"{layout.output_width}x{layout.output_height}",
        "imgsz": int(options.imgsz),
        "conf": float(options.conf),
        "iou": float(options.iou),
        "frame_skip": frame_skip,
        "deepsort_enabled": bool(options.enable_deepsort),
        "tracker_max_time_since_update": int(options.tracker_max_time_since_update),
        "show_stats": bool(options.show_stats),
        "trail_length": int(options.trail_length),
        "smooth_window": int(options.smooth_window),
        "smoothing_enabled": bool(options.smoothing_enabled),
        "smoothing_alpha": float(options.smoothing_alpha),
        "min_box_area": float(options.min_box_area),
        "max_center_jump": float(options.max_center_jump),
        "total_detections": total_detections,
        "unique_tracks": track_count,
        "average_processing_fps": round(processed_frames / total_seconds, 3),
        "average_yolo_ms": round(_mean(yolo_times), 3),
        "average_deepsort_ms": round(_mean(deepsort_times), 3),
        "average_draw_ms": round(_mean(draw_times), 3),
        "average_read_ms": round(_mean(read_times), 3),
        "average_write_ms": round(_mean(write_times), 3),
        "total_seconds": round(total_seconds, 3),
        "detected_frames": detected_frames,
        "class_counts": dict(class_counts),
        "track_class_counts": {name: len(ids) for name, ids in track_ids_by_class.items()},
    }
    _write_summary(summary_path, summary)

    if progress_fn is not None:
        progress_fn(
            {
                "status": "finished",
                "progress": 100.0,
                "current_frame": frame_id,
                "total_frames": total_expected,
                "fps": round(processed_frames / total_seconds, 3),
                "message": "finished",
            }
        )

    return VideoProcessResult(
        run_dir=options.run_dir,
        output_video_path=output_video_path,
        result_csv_path=result_csv_path,
        result_json_path=result_json_path,
        summary_path=summary_path,
        summary=summary,
        rows=rows,
    )
