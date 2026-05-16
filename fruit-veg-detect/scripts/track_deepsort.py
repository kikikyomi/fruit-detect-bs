from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from camera_common import (
    CsvMetricWriter,
    FrameMetrics,
    SlidingAverage,
    build_video_tracker,
    draw_boxes,
    iter_existing_trajectories,
    load_yolo,
    open_capture,
    predict_boxes,
    print_metrics,
    resolve_device,
    resolve_half,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLO + DeepSORT camera tracking")
    parser.add_argument("--weights", default="best.pt", help="Model weights, e.g. best.pt")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0")
    parser.add_argument("--half", dest="half", action="store_true", default=None)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--frame-skip", type=int, default=2, help="Run YOLO every N frames")
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--max-cosine-distance", type=float, default=0.3)
    parser.add_argument("--nn-budget", type=int, default=50)
    parser.add_argument("--disable-deepsort", action="store_true", help="Display YOLO detections only")
    parser.add_argument("--save", action="store_true", help="Save sampled annotated frames")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--save-csv", action="store_true", help="Save per-frame metrics CSV")
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    args.device, device_is_cuda, device_label = resolve_device(args.device)
    args.half = resolve_half(args.half, device_is_cuda)

    print(
        f"YOLO + DeepSORT camera tracking: device={args.device} ({device_label}) half={args.half} "
        f"imgsz={args.imgsz} frame_skip={args.frame_skip} disable_deepsort={args.disable_deepsort}"
    )

    model = load_yolo(args.weights)
    tracker = None if args.disable_deepsort else build_video_tracker(args)
    cap = open_capture(args.source, args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    run_dir = Path("runs") / "camera_deepsort"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_writer = CsvMetricWriter(run_dir / "metrics.csv") if args.save_csv else None
    video_writer: cv2.VideoWriter | None = None

    avg_total_ms = SlidingAverage(30)
    last_detections: list[dict] = []
    last_display_boxes: list[dict] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            started_at = time.perf_counter()
            detection_frame = frame_index % args.frame_skip == 0
            yolo_ms = 0.0
            detections: list[dict] = []
            if detection_frame:
                detections, yolo_ms = predict_boxes(
                    model,
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    half=args.half,
                )
                last_detections = detections

            deepsort_ms = 0.0
            trajectories = None
            if args.disable_deepsort or tracker is None:
                if detection_frame:
                    last_display_boxes = detections
                display_boxes = last_display_boxes
            else:
                track_started_at = time.perf_counter()
                tracked_boxes = tracker.update(detections, frame, frame_index)
                deepsort_ms = (time.perf_counter() - track_started_at) * 1000.0
                if tracked_boxes:
                    last_display_boxes = tracked_boxes
                elif detection_frame and not detections:
                    last_display_boxes = []
                elif detection_frame:
                    last_display_boxes = detections
                display_boxes = last_display_boxes
                trajectories = iter_existing_trajectories(tracker)

            draw_frame = frame.copy()
            draw_ms = draw_boxes(draw_frame, display_boxes, trajectories=trajectories)
            total_ms = (time.perf_counter() - started_at) * 1000.0
            avg_total_ms.add(total_ms)
            fps = 1000.0 / avg_total_ms.average if avg_total_ms.average > 0 else 0.0

            metrics = FrameMetrics(
                frame_index=frame_index,
                total_ms=total_ms,
                yolo_ms=yolo_ms,
                deepsort_ms=deepsort_ms,
                draw_ms=draw_ms,
                fps=fps,
                detections=len(last_detections),
                tracks=len(display_boxes),
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
                video_writer = cv2.VideoWriter(str(run_dir / "track_deepsort.mp4"), fourcc, float(fps_out), (w, h))
            if video_writer is not None:
                video_writer.write(draw_frame)
            if args.save and frame_index % max(1, args.save_interval) == 0:
                cv2.imwrite(str(run_dir / f"frame_{frame_index:06d}.jpg"), draw_frame)

            if not args.no_display:
                cv2.imshow("YOLO + DeepSORT Camera Tracking", draw_frame)
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


if __name__ == "__main__":
    main()
