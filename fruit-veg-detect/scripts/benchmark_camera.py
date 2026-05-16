from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from camera_common import (
    build_video_tracker,
    draw_boxes,
    iter_existing_trajectories,
    load_yolo,
    mean,
    open_capture,
    predict_boxes,
    resolve_device,
    resolve_half,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark realtime camera FPS")
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
    parser.add_argument("--disable-deepsort", action="store_true")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--duration", type=float, default=0.0, help="Optional benchmark duration in seconds")
    parser.add_argument("--display", action="store_true", help="Show benchmark frames")
    return parser.parse_args()


def write_results(run_dir: Path, summary: dict[str, str | int | float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result_txt = run_dir / "result.txt"
    result_csv = run_dir / "result.csv"

    lines = [f"{key}: {value}" for key, value in summary.items()]
    result_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with result_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    args.device, device_is_cuda, device_label = resolve_device(args.device)
    args.half = resolve_half(args.half, device_is_cuda)

    print(
        f"Benchmark camera: device={args.device} ({device_label}) half={args.half} "
        f"imgsz={args.imgsz} frame_skip={args.frame_skip} disable_deepsort={args.disable_deepsort}"
    )

    model = load_yolo(args.weights)
    tracker = None if args.disable_deepsort else build_video_tracker(args)
    cap = open_capture(args.source, args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    total_frames = 0
    detection_frames = 0
    yolo_times: list[float] = []
    deepsort_times: list[float] = []
    draw_times: list[float] = []
    last_display_boxes: list[dict] = []

    benchmark_started_at = time.perf_counter()
    try:
        while True:
            if args.max_frames and total_frames >= args.max_frames:
                break
            if args.duration > 0 and time.perf_counter() - benchmark_started_at >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                break

            detection_frame = total_frames % args.frame_skip == 0
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
                yolo_times.append(yolo_ms)
                detection_frames += 1

            trajectories = None
            if args.disable_deepsort or tracker is None:
                if detection_frame:
                    last_display_boxes = detections
                display_boxes = last_display_boxes
            else:
                track_started_at = time.perf_counter()
                tracked_boxes = tracker.update(detections, frame, total_frames)
                deepsort_ms = (time.perf_counter() - track_started_at) * 1000.0
                deepsort_times.append(deepsort_ms)
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
            draw_times.append(draw_ms)

            if args.display:
                cv2.imshow("Camera Benchmark", draw_frame)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break

            total_frames += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.perf_counter() - benchmark_started_at)
    summary: dict[str, str | int | float] = {
        "average_fps": round(total_frames / elapsed, 3),
        "average_yolo_ms": round(mean(yolo_times), 3),
        "average_deepsort_ms": round(mean(deepsort_times), 3),
        "average_draw_ms": round(mean(draw_times), 3),
        "total_frames": total_frames,
        "effective_detection_frames": detection_frames,
        "device": device_label,
        "device_arg": args.device,
        "half": int(bool(args.half)),
        "imgsz": int(args.imgsz),
        "frame_skip": int(args.frame_skip),
        "deepsort_enabled": int(not args.disable_deepsort),
    }

    run_dir = Path("runs") / "benchmark_camera"
    write_results(run_dir, summary)

    print("Benchmark result")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved: {run_dir / 'result.txt'}")
    print(f"Saved: {run_dir / 'result.csv'}")


if __name__ == "__main__":
    main()
