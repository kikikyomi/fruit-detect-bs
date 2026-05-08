from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.services.detector import detector_service
from app.services.record_store import create_record, get_record
from app.services.tracker import VideoTracker
from app.utils.cv import save_image, unique_name

router = APIRouter(prefix="/api/detect", tags=["detect-video"])


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)


def _stream_annotated_video(video_path: Path, loop: bool) -> Any:
    while True:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            break

        fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
        frame_delay = 1.0 / max(1.0, min(float(fps), 25.0))

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not encoded:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                time.sleep(frame_delay)
        finally:
            capture.release()

        if not loop:
            break


def _interpolate_value(start: float, end: float, alpha: float) -> float:
    return float(start + (end - start) * alpha)


def _interpolate_box(start_box: dict[str, Any], end_box: dict[str, Any], alpha: float) -> dict[str, Any]:
    cls_name = str(end_box.get("cls_name") or start_box.get("cls_name") or "obj")
    cls_id = int(end_box.get("cls_id", start_box.get("cls_id", -1)))
    conf = _interpolate_value(float(start_box.get("conf", 0.0)), float(end_box.get("conf", 0.0)), alpha)

    interpolated = {
        "x1": _interpolate_value(float(start_box["x1"]), float(end_box["x1"]), alpha),
        "y1": _interpolate_value(float(start_box["y1"]), float(end_box["y1"]), alpha),
        "x2": _interpolate_value(float(start_box["x2"]), float(end_box["x2"]), alpha),
        "y2": _interpolate_value(float(start_box["y2"]), float(end_box["y2"]), alpha),
        "conf": conf,
        "cls_id": cls_id,
        "cls_name": cls_name,
    }

    track_id = end_box.get("track_id", start_box.get("track_id"))
    if track_id is not None:
        interpolated["track_id"] = int(track_id)

    return interpolated


def _build_dense_track_boxes(
    sampled_boxes_by_frame: dict[int, list[dict[str, Any]]],
    max_gap: int,
) -> dict[int, list[dict[str, Any]]]:
    dense_boxes: dict[int, list[dict[str, Any]]] = {
        frame_index: [box.copy() for box in boxes]
        for frame_index, boxes in sampled_boxes_by_frame.items()
    }
    observations_by_track: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for frame_index, boxes in sampled_boxes_by_frame.items():
        for box in boxes:
            track_id_raw = box.get("track_id")
            if track_id_raw is None:
                continue
            observations_by_track[int(track_id_raw)].append((frame_index, box))

    for observations in observations_by_track.values():
        observations.sort(key=lambda item: item[0])
        for idx in range(len(observations) - 1):
            start_frame, start_box = observations[idx]
            end_frame, end_box = observations[idx + 1]
            gap = end_frame - start_frame
            if gap <= 1 or gap > max_gap:
                continue

            for frame_index in range(start_frame + 1, end_frame):
                alpha = (frame_index - start_frame) / gap
                dense_boxes.setdefault(frame_index, []).append(_interpolate_box(start_box, end_box, alpha))

    return dense_boxes


def _append_trajectory_points(
    trajectories: dict[int, list[tuple[int, int]]],
    boxes: list[dict[str, Any]],
) -> None:
    for box in boxes:
        track_id_raw = box.get("track_id")
        if track_id_raw is None:
            continue

        track_id = int(track_id_raw)
        center = (
            int(round((float(box["x1"]) + float(box["x2"])) / 2.0)),
            int(round((float(box["y1"]) + float(box["y2"])) / 2.0)),
        )
        points = trajectories.setdefault(track_id, [])
        if not points or points[-1] != center:
            points.append(center)
        if len(points) > settings.TRACKER_TRAJECTORY_LEN:
            del points[:-settings.TRACKER_TRAJECTORY_LEN]


def _prune_stale_trajectories(
    trajectories: dict[int, list[tuple[int, int]]],
    last_seen_by_track: dict[int, int],
    current_frame_index: int,
    max_stale_frames: int,
) -> None:
    stale_track_ids = [
        track_id
        for track_id, last_seen in last_seen_by_track.items()
        if current_frame_index - last_seen > max_stale_frames
    ]
    for track_id in stale_track_ids:
        trajectories.pop(track_id, None)
        last_seen_by_track.pop(track_id, None)


@router.get("/video/stream/{record_id}")
def stream_annotated_video(record_id: int, loop: bool = True) -> StreamingResponse:
    record = get_record(record_id)
    if record is None or record.type != "video" or not record.output_path:
        raise HTTPException(status_code=404, detail="Annotated video not found")

    output_path = Path(record.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video file missing")

    return StreamingResponse(
        _stream_annotated_video(output_path, loop=loop),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    model_key: str | None = Form(default=None),
    frame_interval: int | None = Form(default=None),
    tracker_backend: str | None = Form(default=None),
    tracker_max_time_since_update: int | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    interval = frame_interval or settings.VIDEO_SAMPLE_INTERVAL
    interval = max(1, interval)
    active_model_key = detector_service.normalize_model_key(model_key)

    input_name = unique_name(file.filename)
    input_path = settings.UPLOAD_DIR / input_name
    input_path.write_bytes(content)
    input_url = f"/static/uploads/{input_name}"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    tracker = VideoTracker(
        backend_override=tracker_backend,
        n_init_override=(settings.TRACKER_N_INIT if interval <= 1 else 1),
        max_time_since_update_override=max(
            tracker_max_time_since_update or settings.TRACKER_MAX_TIME_SINCE_UPDATE,
            interval + 2,
        ),
    )
    total_started_at = time.perf_counter()
    analysis_started_at = total_started_at

    frame_index = 0
    sampled_frames = 0
    detections = 0
    class_stats: dict[str, int] = defaultdict(int)
    keyframes: list[str] = []
    keyframe_details: list[dict[str, Any]] = []
    last_frame_tracks: list[dict[str, Any]] = []
    sampled_boxes_by_frame: dict[int, list[dict[str, Any]]] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detection_boxes: list[dict[str, Any]] = []

            if frame_index % interval == 0:
                sampled_frames += 1
                result = detector_service.predict_image(frame, conf=conf, iou=iou, model_key=active_model_key)
                if result.get("error"):
                    raise HTTPException(status_code=500, detail=f"Inference failed: {result['error']}")
                detection_boxes = result["boxes"]

                detections += len(detection_boxes)
                for box in detection_boxes:
                    name = str(box.get("cls_name", box.get("cls_id", "obj")))
                    class_stats[name] += 1

            tracked_boxes = tracker.update(detection_boxes, frame, frame_index)
            if frame_index % interval == 0:
                boxes_for_frame = tracked_boxes or detection_boxes
                sampled_boxes_by_frame[frame_index] = [box.copy() for box in boxes_for_frame]
                if boxes_for_frame:
                    last_frame_tracks = [box.copy() for box in boxes_for_frame]

            frame_index += 1
    finally:
        cap.release()

    analysis_seconds = max(1e-6, time.perf_counter() - analysis_started_at)
    track_summaries = tracker.get_track_summaries()
    track_class_stats: dict[str, int] = defaultdict(int)
    for item in track_summaries:
        name = str(item.get("cls_name", "obj"))
        track_class_stats[name] += 1

    trajectories = tracker.get_serialized_trajectories()
    has_results = detections > 0 or len(track_summaries) > 0

    stem = Path(input_name).stem
    output_name = f"annotated_{stem}.mp4"
    output_path = settings.OUTPUT_DIR / output_name
    output_url: str | None = f"/static/outputs/{output_name}"
    dense_boxes_by_frame = _build_dense_track_boxes(
        sampled_boxes_by_frame,
        max_gap=max(3, interval * 3),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    writer_enabled = writer.isOpened()
    rendered_trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    rendered_last_seen_by_track: dict[int, int] = {}

    render_cap = cv2.VideoCapture(str(input_path))
    try:
        render_frame_index = 0
        while True:
            ok, frame = render_cap.read()
            if not ok:
                break

            boxes_to_draw = [box.copy() for box in dense_boxes_by_frame.get(render_frame_index, [])]
            if boxes_to_draw:
                _append_trajectory_points(rendered_trajectories, boxes_to_draw)
                for box in boxes_to_draw:
                    track_id_raw = box.get("track_id")
                    if track_id_raw is None:
                        continue
                    rendered_last_seen_by_track[int(track_id_raw)] = render_frame_index
                last_frame_tracks = [box.copy() for box in boxes_to_draw]
            _prune_stale_trajectories(
                rendered_trajectories,
                rendered_last_seen_by_track,
                render_frame_index,
                settings.TRACKER_RENDER_STALE_FRAMES,
            )

            frame_to_write = detector_service.draw_boxes_with_trajectories(
                frame,
                boxes_to_draw,
                trajectories=rendered_trajectories,
            )

            if render_frame_index in sampled_boxes_by_frame:
                keyframe_name = f"{stem}_kf_{render_frame_index:06d}.jpg"
                keyframe_path = settings.OUTPUT_DIR / keyframe_name
                save_image(keyframe_path, frame_to_write)
                keyframe_url = f"/static/outputs/{keyframe_name}"
                keyframes.append(keyframe_url)
                keyframe_details.append(
                    {
                        "frame_index": render_frame_index,
                        "image_url": keyframe_url,
                        "tracks": boxes_to_draw,
                    }
                )

            if writer_enabled:
                writer.write(frame_to_write)

            render_frame_index += 1
    finally:
        render_cap.release()
        if writer_enabled:
            writer.release()

    if not writer_enabled and output_path.exists():
        output_path.unlink(missing_ok=True)
        output_url = None

    if rendered_trajectories:
        trajectories = {
            str(track_id): [{"x": int(x), "y": int(y)} for x, y in points]
            for track_id, points in rendered_trajectories.items()
        }

    processing_seconds = max(1e-6, time.perf_counter() - total_started_at)
    processing_fps = round((frame_index / processing_seconds) if frame_index > 0 else 0.0, 2)
    detection_fps = round((sampled_frames / processing_seconds) if sampled_frames > 0 else 0.0, 2)

    summary: dict[str, Any] = {
        "total_frames": frame_index,
        "sampled_frames": sampled_frames,
        "detections": detections,
        "input_fps": round(float(fps), 2),
        "processing_seconds": round(processing_seconds, 3),
        "processing_fps": processing_fps,
        "detection_fps": detection_fps,
        "target_fps_met": processing_fps >= 15.0,
        "analysis_seconds": round(analysis_seconds, 3),
        "render_seconds": round(max(0.0, processing_seconds - analysis_seconds), 3),
        "class_stats": dict(class_stats),
        "frame_interval": interval,
        "unique_tracks": len(track_summaries),
        "track_class_stats": dict(track_class_stats),
        "tracker": tracker.tracker_name,
        "deepsort_enabled": tracker.deepsort_enabled,
        "tracker_n_init": tracker.n_init,
        "tracker_max_time_since_update": tracker.max_time_since_update,
        "model_key": active_model_key,
        "saved": has_results or settings.SAVE_EMPTY_RESULTS,
    }

    record_id: int | None = None
    if has_results or settings.SAVE_EMPTY_RESULTS:
        record_id = create_record(
            record_type="video",
            file_name=file.filename,
            input_path=input_path,
            input_url=input_url,
            output_path=output_path if output_url else None,
            output_url=output_url,
            result={
                "boxes": last_frame_tracks,
            },
            summary={
                **summary,
                "keyframes": keyframes,
                "keyframe_details": keyframe_details,
                "track_summaries": track_summaries,
                "trajectories": trajectories,
            },
        )
    else:
        _remove_file(input_path)
        _remove_file(output_path if output_url else None)
        for item in keyframe_details:
            image_url = item.get("image_url")
            if isinstance(image_url, str) and image_url.startswith("/static/outputs/"):
                _remove_file(settings.OUTPUT_DIR / image_url.replace("/static/outputs/", "", 1))
        output_url = None
        keyframes = []
        keyframe_details = []

    return {
        "record_id": record_id,
        "summary": summary,
        "outputs": {
            "video_url": output_url,
            "stream_url": f"/api/detect/video/stream/{record_id}" if record_id else None,
            "keyframes": keyframes,
            "keyframe_details": keyframe_details,
        },
        "tracking": {
            "track_summaries": track_summaries,
            "trajectories": trajectories,
        },
        "saved": summary["saved"],
    }
