from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.services.detector import detector_service
from app.services.record_store import create_record, get_record
from app.services.tracker import VideoTracker
from app.utils.cv import save_image, unique_name
from app.utils.video_processing import VideoProcessOptions, VideoProcessResult, process_video_file
from app.utils.video_utils import parse_classes

router = APIRouter(prefix="/api/detect", tags=["detect-video"])

_video_tasks: dict[str, dict[str, Any]] = {}
_video_tasks_lock = Lock()


def _static_output_url(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(settings.OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return "/static/outputs/" + relative.as_posix()


def _update_video_task(task_id: str, **updates: Any) -> None:
    with _video_tasks_lock:
        task = _video_tasks.get(task_id)
        if task is None:
            return
        task.update(updates)


def _public_video_task(task_id: str) -> dict[str, Any]:
    with _video_tasks_lock:
        task = _video_tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Video task not found")
        return dict(task)


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)


def _resolve_video_half(raw_half: bool | None, device: str) -> bool:
    if raw_half is not None:
        return bool(raw_half)
    runtime_device = detector_service._resolve_runtime_device(device)
    return runtime_device != "cpu"


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "undefined"}:
        return default

    try:
        return int(text)
    except ValueError:
        return default


def _run_video_task(
    *,
    task_id: str,
    input_path: Path,
    input_url: str,
    file_name: str,
    options: VideoProcessOptions,
    model_key: str,
    tracker_backend: str,
    max_age: int,
    max_time_since_update: int,
    n_init: int,
    max_iou_distance: float,
    max_cosine_distance: float,
    nn_budget: int,
    smoothing_enabled: bool,
    smoothing_alpha: float,
    max_center_jump: float,
    debug: bool,
    classes: list[int] | None,
) -> None:
    def progress(progress_data: dict[str, Any]) -> None:
        _update_video_task(task_id, **progress_data)

    try:
        _update_video_task(task_id, status="running", message="loading model and opening video")

        def predict_fn(frame):
            result = detector_service.predict_image(
                frame,
                conf=options.conf,
                iou=options.iou,
                model_key=model_key,
                imgsz=options.imgsz,
                device=options.device,
                half=options.half,
                classes=classes,
            )
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
            return result["boxes"], float(result.get("timings", {}).get("yolo_ms", 0.0))

        tracker = None
        if options.enable_deepsort:
            tracker = VideoTracker(
                backend_override=tracker_backend,
                n_init_override=n_init,
                max_age_override=max_age,
                max_time_since_update_override=max_time_since_update,
                max_cosine_distance_override=max_cosine_distance,
                max_iou_distance_override=max_iou_distance,
                nn_budget_override=nn_budget,
                device_override=options.device,
                smoothing_enabled_override=smoothing_enabled,
                smoothing_alpha_override=smoothing_alpha,
                smoothing_max_center_jump_override=max_center_jump,
                debug=debug,
            )

        result: VideoProcessResult = process_video_file(options, predict_fn=predict_fn, tracker=tracker, progress_fn=progress)
        output_video_url = _static_output_url(result.output_video_path)
        result_csv_url = _static_output_url(result.result_csv_path)
        result_json_url = _static_output_url(result.result_json_path)

        record_id: int | None = None
        if result.output_video_path is not None:
            record_id = create_record(
                record_type="video",
                file_name=file_name,
                input_path=input_path,
                input_url=input_url,
                output_path=result.output_video_path,
                output_url=output_video_url,
                result={"boxes": []},
                summary=result.summary,
            )

        _update_video_task(
            task_id,
            status="finished",
            progress=100.0,
            message="finished",
            record_id=record_id,
            output_video_url=output_video_url,
            result_csv_url=result_csv_url,
            result_json_url=result_json_url,
            summary_url=_static_output_url(result.summary_path),
            summary=result.summary,
            saved=True,
        )
    except Exception as exc:
        _update_video_task(
            task_id,
            status="failed",
            message=str(exc),
            error=str(exc),
            progress=0.0,
        )


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
    imgsz: int | None = Form(default=None),
    device: str | None = Form(default=None),
    half: bool | None = Form(default=None),
    model_key: str | None = Form(default=None),
    frame_interval: int | None = Form(default=None),
    frame_skip: int | None = Form(default=None),
    enable_deepsort: bool = Form(default=True),
    tracker_backend: str | None = Form(default=None),
    tracker_max_time_since_update: str | None = Form(default=None),
    max_age: int | None = Form(default=None),
    n_init: int | None = Form(default=None),
    max_iou_distance: float | None = Form(default=None),
    max_cosine_distance: float | None = Form(default=None),
    nn_budget: int | None = Form(default=None),
    trail_length: int | None = Form(default=None),
    smooth_window: int | None = Form(default=None),
    smoothing_enabled: bool | None = Form(default=None),
    smooth_alpha: float | None = Form(default=None),
    min_box_area: float = Form(default=0.0),
    max_center_jump: float = Form(default=0.0),
    debug: bool | None = Form(default=None),
    output_width: int | None = Form(default=None),
    output_height: int | None = Form(default=None),
    keep_original_resolution: bool | None = Form(default=None),
    resize_output: bool | None = Form(default=None),
    show_stats: bool | None = Form(default=None),
    start_time: float | None = Form(default=None),
    end_time: float | None = Form(default=None),
    save_video: bool = Form(default=True),
    save_csv: bool = Form(default=True),
    save_json: bool = Form(default=True),
    classes: str | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    active_frame_skip = frame_skip if frame_skip is not None else frame_interval
    active_frame_skip = max(1, int(active_frame_skip if active_frame_skip is not None else settings.VIDEO_FRAME_SKIP))
    active_model_key = detector_service.normalize_model_key(model_key)
    active_device = device or settings.DEVICE
    active_half = _resolve_video_half(half, active_device)
    active_max_age = max(
        1,
        int(
            max_age
            if max_age is not None
            else settings.VIDEO_TRACKER_MAX_AGE
        ),
    )
    active_max_time_since_update = max(
        0,
        _safe_int(
            tracker_max_time_since_update,
            settings.VIDEO_TRACKER_MAX_TIME_SINCE_UPDATE,
        ),
    )
    active_n_init = max(1, int(n_init if n_init is not None else settings.VIDEO_TRACKER_N_INIT))
    active_max_cosine = float(
        max_cosine_distance
        if max_cosine_distance is not None
        else settings.VIDEO_TRACKER_MAX_COSINE_DISTANCE
    )
    active_max_iou = float(
        max_iou_distance
        if max_iou_distance is not None
        else settings.TRACKER_MAX_IOU_DISTANCE
    )
    active_nn_budget = max(0, int(nn_budget if nn_budget is not None else settings.VIDEO_TRACKER_NN_BUDGET))
    active_smoothing_enabled = bool(
        smoothing_enabled
        if smoothing_enabled is not None
        else settings.SMOOTHING_ENABLED
    )
    active_smoothing_alpha = min(1.0, max(0.0, float(smooth_alpha if smooth_alpha is not None else settings.SMOOTHING_ALPHA)))
    active_max_center_jump = float(max_center_jump if max_center_jump > 0 else settings.SMOOTHING_MAX_CENTER_JUMP)
    active_debug = bool(debug) if debug is not None else False
    active_tracker_backend = (tracker_backend or "deepsort").strip().lower()
    if active_tracker_backend == "naive":
        active_tracker_backend = "naive"
    elif active_tracker_backend not in {"auto", "deepsort", "naive"}:
        active_tracker_backend = "deepsort"

    task_id = uuid.uuid4().hex
    input_name = unique_name(file.filename)
    input_path = settings.UPLOAD_DIR / input_name
    input_path.write_bytes(content)
    input_url = f"/static/uploads/{input_name}"

    run_dir = settings.OUTPUT_DIR / "video_tasks" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    options = VideoProcessOptions(
        source_path=str(input_path),
        run_dir=run_dir,
        weights_path=detector_service.model_paths[active_model_key],
        model_name=Path(detector_service.model_paths[active_model_key]).name,
        imgsz=int(imgsz if imgsz is not None else settings.VIDEO_IMGSZ),
        conf=float(conf if conf is not None else settings.VIDEO_CONF_THRES),
        iou=float(iou if iou is not None else settings.VIDEO_IOU_THRES),
        device=active_device,
        half=active_half,
        frame_skip=active_frame_skip,
        output_width=int(output_width if output_width is not None else settings.VIDEO_OUTPUT_WIDTH),
        output_height=int(output_height if output_height is not None else settings.VIDEO_OUTPUT_HEIGHT),
        keep_original_resolution=bool(
            keep_original_resolution
            if keep_original_resolution is not None
            else settings.VIDEO_KEEP_ORIGINAL_RESOLUTION
        ),
        resize_output=bool(resize_output if resize_output is not None else settings.VIDEO_RESIZE_OUTPUT),
        show_stats=bool(show_stats if show_stats is not None else settings.VIDEO_SHOW_STATS),
        start_time=start_time,
        end_time=end_time,
        save_video=save_video,
        save_csv=save_csv,
        save_json=save_json,
        enable_deepsort=enable_deepsort,
        tracker_max_time_since_update=active_max_time_since_update,
        trail_length=int(trail_length if trail_length is not None else settings.VIDEO_TRAIL_LENGTH),
        smooth_window=int(smooth_window if smooth_window is not None else settings.VIDEO_SMOOTH_WINDOW),
        smoothing_enabled=active_smoothing_enabled,
        smoothing_alpha=active_smoothing_alpha,
        min_box_area=float(min_box_area if min_box_area > 0 else settings.VIDEO_MIN_BOX_AREA),
        max_center_jump=active_max_center_jump,
        debug=active_debug,
    )

    task = {
        "task_id": task_id,
        "status": "waiting",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0.0,
        "message": "waiting",
        "record_id": None,
        "output_video_url": None,
        "result_csv_url": None,
        "result_json_url": None,
        "summary_url": None,
        "summary": None,
        "saved": False,
    }
    with _video_tasks_lock:
        _video_tasks[task_id] = task

    worker = Thread(
        target=_run_video_task,
        kwargs={
            "task_id": task_id,
            "input_path": input_path,
            "input_url": input_url,
            "file_name": file.filename,
            "options": options,
            "model_key": active_model_key,
            "tracker_backend": active_tracker_backend,
            "max_age": active_max_age,
            "max_time_since_update": active_max_time_since_update,
            "n_init": active_n_init,
            "max_iou_distance": active_max_iou,
            "max_cosine_distance": active_max_cosine,
            "nn_budget": active_nn_budget,
            "smoothing_enabled": active_smoothing_enabled,
            "smoothing_alpha": active_smoothing_alpha,
            "max_center_jump": active_max_center_jump,
            "debug": active_debug,
            "classes": parse_classes(classes),
        },
        daemon=True,
    )
    worker.start()

    return task


@router.get("/video/tasks/{task_id}")
def get_video_task(task_id: str) -> dict[str, Any]:
    return _public_video_task(task_id)
