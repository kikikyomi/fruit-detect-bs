from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_utils import evaluate_model
from tools.ultra_helpers import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one model and export confusion matrix.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--class-aliases", default="")
    args = parser.parse_args()

    result = evaluate_model(
        weights=resolve_path(ROOT, args.weights),
        data=resolve_path(ROOT, args.data),
        outdir=resolve_path(ROOT, args.outdir),
        conf=float(args.conf),
        iou_threshold=float(args.iou),
        imgsz=int(args.imgsz),
        device=str(args.device),
        alias_map_path=resolve_path(ROOT, args.class_aliases) if args.class_aliases else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

