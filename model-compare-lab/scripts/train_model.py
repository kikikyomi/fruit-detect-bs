from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_utils import evaluate_model
from tools.ultra_helpers import build_model, load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detection model with Ultralytics.")
    parser.add_argument("--config", default="", help="Experiment YAML config.")
    parser.add_argument("--model", default="", help="Model name or weights path.")
    parser.add_argument("--data", default="", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--optimizer", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--alias-map", default="", help="Optional class alias map for post-train eval.")
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        config = load_yaml(Path(args.config).resolve())

    def pick(key: str, current: Any, empty_value: Any) -> Any:
        return current if current != empty_value else config.get(key)

    runtime = {
        "model": pick("model", args.model, ""),
        "data": pick("data", args.data, ""),
        "epochs": pick("epochs", args.epochs, 0),
        "imgsz": pick("imgsz", args.imgsz, 0),
        "batch": pick("batch", args.batch, 0),
        "device": pick("device", args.device, ""),
        "workers": pick("workers", args.workers, -1),
        "patience": pick("patience", args.patience, -1),
        "project": pick("project", args.project, ""),
        "name": pick("name", args.name, ""),
        "optimizer": pick("optimizer", args.optimizer, ""),
        "seed": pick("seed", args.seed, -1),
        "conf": pick("conf", args.conf, 0.25),
        "iou": pick("iou", args.iou, 0.5),
        "alias_map": pick("alias_map", args.alias_map, ""),
    }
    for required_key in ("model", "data", "epochs", "imgsz", "batch", "device", "project", "name"):
        if not runtime.get(required_key):
            raise ValueError(f"Missing required config value: {required_key}")
    return runtime


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args)

    data_path = resolve_path(ROOT, cfg["data"])
    project_path = resolve_path(ROOT, cfg["project"])
    run_name = str(cfg["name"])
    run_dir = project_path / run_name

    model = build_model(str(cfg["model"]))
    train_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "device": str(cfg["device"]),
        "project": str(project_path),
        "name": run_name,
        "optimizer": str(cfg["optimizer"]) if cfg["optimizer"] else "auto",
        "exist_ok": True,
        "verbose": True,
    }
    if int(cfg["workers"]) >= 0:
        train_kwargs["workers"] = int(cfg["workers"])
    if int(cfg["patience"]) >= 0:
        train_kwargs["patience"] = int(cfg["patience"])
    if int(cfg["seed"]) >= 0:
        train_kwargs["seed"] = int(cfg["seed"])

    model.train(
        **train_kwargs,
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_pt}")

    eval_outdir = run_dir / "analysis"
    summary = evaluate_model(
        weights=best_pt,
        data=data_path,
        outdir=eval_outdir,
        conf=float(cfg["conf"]),
        iou_threshold=float(cfg["iou"]),
        imgsz=int(cfg["imgsz"]),
        device=str(cfg["device"]),
        alias_map_path=cfg["alias_map"] or None,
    )

    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[done] best.pt: {best_pt}")
    print(f"[done] analysis: {eval_outdir}")


if __name__ == "__main__":
    main()
