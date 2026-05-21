from __future__ import annotations

import argparse
from datetime import datetime
import time
from pathlib import Path

import cv2

from camera_common import (
    CsvMetricWriter,
    FrameMetrics,
    SlidingAverage,
    draw_boxes,
    load_yolo,
    open_capture,
    parse_source,
    predict_boxes,
    print_metrics,
    resolve_device,
    resolve_half,
    resolve_weights_path,
)
from app.utils.video_processing import VideoProcessOptions, process_video_file
from app.utils.video_utils import parse_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO camera or video inference")
    parser.add_argument("--weights", default="best.pt", help="Model weights, e.g. best.pt")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0")
    parser.add_argument("--half", dest="half", action="store_true", default=None)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--classes", default=None, help="Comma-separated class ids, e.g. 0,2,3")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--frame-skip", type=int, default=None, help="Run YOLO every N frames")
    parser.add_argument("--save", action="store_true", help="Save sampled annotated frames")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV results")
    parser.add_argument("--save-json", action="store_true", help="Save JSON results")
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--keep-original-resolution", action="store_true")
    parser.add_argument("--resize-output", dest="resize_output", action="store_true", default=True)
    parser.add_argument("--no-resize-output", dest="resize_output", action="store_false")
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    return parser.parse_args()


def _run_camera(args: argparse.Namespace, model: object, classes: list[int] | None, device_label: str) -> None:
    print(
        f"YOLO camera inference: device={args.device} ({device_label}) half={args.half} "
        f"imgsz={args.imgsz} frame_skip={args.frame_skip}"
    )

    cap = open_capture(args.source, args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    run_dir = Path("runs") / "camera_inference"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_writer = CsvMetricWriter(run_dir / "metrics.csv") if args.save_csv else None
    video_writer: cv2.VideoWriter | None = None

    avg_total_ms = SlidingAverage(30)
    last_boxes: list[dict] = []
    frame_index = 0
    detection_interval = max(1, int(args.frame_skip) + 1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            started_at = time.perf_counter()
            detection_frame = frame_index % detection_interval == 0
            yolo_ms = 0.0
            if detection_frame:
                last_boxes, yolo_ms = predict_boxes(
                    model,
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    half=args.half,
                    classes=classes,
                )

            draw_frame = frame.copy()
            draw_ms = draw_boxes(draw_frame, last_boxes)
            total_ms = (time.perf_counter() - started_at) * 1000.0
            avg_total_ms.add(total_ms)
            fps = 1000.0 / avg_total_ms.average if avg_total_ms.average > 0 else 0.0

            metrics = FrameMetrics(
                frame_index=frame_index,
                total_ms=total_ms,
                yolo_ms=yolo_ms,
                deepsort_ms=0.0,
                draw_ms=draw_ms,
                fps=fps,
                detections=len(last_boxes),
                tracks=0,
                detection_frame=detection_frame,
            )
            if frame_index % max(1, args.log_interval) == 0:
                print_metrics(metrics)
            if csv_writer is not None:
                csv_writer.write(metrics)

            if args.save_video and video_writer is None:
                fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
                h, w = draw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(run_dir / "inference.mp4"), fourcc, float(fps_out), (w, h))
            if video_writer is not None:
                video_writer.write(draw_frame)
            if args.save and frame_index % max(1, args.save_interval) == 0:
                cv2.imwrite(str(run_dir / f"frame_{frame_index:06d}.jpg"), draw_frame)

            if not args.no_display:
                cv2.imshow("YOLO Camera Inference", draw_frame)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break

            frame_index += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_writer is not None:
            csv_writer.close()
        cv2.destroyAllWindows()


def _run_video(args: argparse.Namespace, model: object, classes: list[int] | None, device_label: str) -> None:
    weights_path = resolve_weights_path(args.weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "video_detect" / timestamp

    def predict_fn(frame):
        return predict_boxes(
            model,
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            classes=classes,
        )

    print(
        f"YOLO video inference: source={args.source} device={args.device} ({device_label}) "
        f"half={args.half} imgsz={args.imgsz} frame_skip={args.frame_skip}"
    )
    result = process_video_file(
        VideoProcessOptions(
            source_path=args.source,
            run_dir=run_dir,
            weights_path=weights_path,
            model_name=Path(weights_path).name,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            frame_skip=args.frame_skip,
            output_width=args.output_width,
            output_height=args.output_height,
            keep_original_resolution=args.keep_original_resolution,
            resize_output=args.resize_output,
            show_stats=args.show_stats,
            start_time=args.start_time,
            end_time=args.end_time,
            save_video=args.save_video,
            save_csv=args.save_csv,
            save_json=args.save_json,
            enable_deepsort=False,
        ),
        predict_fn=predict_fn,
    )
    print(f"Saved run: {result.run_dir}")
    print(f"Summary: {result.summary_path}")
    if result.output_video_path:
        print(f"Output video: {result.output_video_path}")
    if result.result_csv_path:
        print(f"CSV: {result.result_csv_path}")
    if result.result_json_path:
        print(f"JSON: {result.result_json_path}")
    print(
        "average_fps={fps} yolo_ms={yolo} draw_ms={draw}".format(
            fps=result.summary["average_processing_fps"],
            yolo=result.summary["average_yolo_ms"],
            draw=result.summary["average_draw_ms"],
        )
    )


def main() -> None:
    args = parse_args()
    source_value = parse_source(args.source)
    is_camera = isinstance(source_value, int)
    if args.imgsz is None:
        args.imgsz = 480 if is_camera else 512
    if args.conf is None:
        args.conf = 0.15 if is_camera else 0.35
    if args.frame_skip is None:
        args.frame_skip = 0 if is_camera else 1
    args.frame_skip = max(0 if is_camera else 1, int(args.frame_skip))
    args.device, device_is_cuda, device_label = resolve_device(args.device)
    args.half = resolve_half(args.half, device_is_cuda)
    classes = parse_classes(args.classes)
    model = load_yolo(args.weights)

    if is_camera:
        _run_camera(args, model, classes, device_label)
    else:
        _run_video(args, model, classes, device_label)


if __name__ == "__main__":
    main()
