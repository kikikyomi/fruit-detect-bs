from __future__ import annotations

import argparse
import csv
from pathlib import Path

from camera_common import build_video_tracker, load_yolo, predict_boxes, resolve_device, resolve_half, resolve_weights_path
from app.utils.video_processing import VideoProcessOptions, process_video_file
from app.utils.video_utils import parse_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO video detection/tracking")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", dest="half", action="store_true", default=None)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--keep-original-resolution", action="store_true")
    parser.add_argument("--resize-output", dest="resize_output", action="store_true", default=True)
    parser.add_argument("--no-resize-output", dest="resize_output", action="store_false")
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--disable-deepsort", action="store_true")
    parser.add_argument("--max-age", type=int, default=40)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-cosine-distance", type=float, default=0.25)
    parser.add_argument("--nn-budget", type=int, default=100)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-box-area", type=float, default=300.0)
    parser.add_argument("--max-center-jump", type=float, default=160.0)
    parser.add_argument("--save-video", action="store_true", help="Include video writing in the benchmark")
    return parser.parse_args()


def write_benchmark_outputs(run_dir: Path, summary: dict[str, object]) -> None:
    result_txt = run_dir / "result.txt"
    result_csv = run_dir / "result.csv"
    result_txt.write_text("\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n", encoding="utf-8")
    with result_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    args.device, device_is_cuda, device_label = resolve_device(args.device)
    args.half = resolve_half(args.half, device_is_cuda)
    classes = parse_classes(args.classes)
    weights_path = resolve_weights_path(args.weights)

    model = load_yolo(args.weights)
    tracker = None if args.disable_deepsort else build_video_tracker(args)

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

    run_dir = Path("runs") / "benchmark_video"
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
            save_csv=False,
            save_json=False,
            enable_deepsort=not args.disable_deepsort,
            trail_length=args.trail_length,
            smooth_window=args.smooth_window,
            min_box_area=args.min_box_area,
            max_center_jump=args.max_center_jump,
        ),
        predict_fn=predict_fn,
        tracker=tracker,
    )

    benchmark_summary = {
        **result.summary,
        "device": device_label,
        "device_arg": args.device,
        "half": int(bool(args.half)),
        "keep_original_resolution": int(bool(args.keep_original_resolution)),
        "resize_output": int(bool(args.resize_output)),
    }
    write_benchmark_outputs(run_dir, benchmark_summary)

    print("Benchmark result")
    for key, value in benchmark_summary.items():
        print(f"{key}: {value}")
    print(f"Saved: {run_dir / 'result.txt'}")
    print(f"Saved: {run_dir / 'result.csv'}")


if __name__ == "__main__":
    main()
