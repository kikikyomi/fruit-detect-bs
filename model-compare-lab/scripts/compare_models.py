from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_utils import evaluate_model
from tools.ultra_helpers import resolve_path


def parse_alias_map(items: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid alias-map entry: {item}")
        name, path = item.split("=", 1)
        result[name.strip()] = resolve_path(ROOT, path.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple models on one dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alias-map", nargs="*", default=[])
    args = parser.parse_args()

    if len(args.weights) != len(args.names):
        raise ValueError("`--weights` and `--names` must have the same length.")

    outdir = resolve_path(ROOT, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    alias_maps = parse_alias_map(args.alias_map)
    rows: list[dict[str, object]] = []

    for model_name, weight_path in zip(args.names, args.weights, strict=True):
        model_outdir = outdir / model_name
        summary = evaluate_model(
            weights=resolve_path(ROOT, weight_path),
            data=resolve_path(ROOT, args.data),
            outdir=model_outdir,
            conf=float(args.conf),
            iou_threshold=float(args.iou),
            imgsz=int(args.imgsz),
            device=str(args.device),
            alias_map_path=alias_maps.get(model_name),
        )
        rows.append({"model_name": model_name, **summary})

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    json_path = outdir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(f"[done] comparison summary: {csv_path}")


if __name__ == "__main__":
    main()

