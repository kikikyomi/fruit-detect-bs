from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from tools.ultra_helpers import build_model, load_json, load_yaml, resolve_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetSpec:
    yaml_path: Path
    root: Path
    names: list[str]
    val_images: list[Path]


def load_dataset_spec(dataset_yaml: str | Path) -> DatasetSpec:
    yaml_path = Path(dataset_yaml).resolve()
    data = load_yaml(yaml_path)
    root = resolve_path(yaml_path.parent, data["path"])
    names_raw = data["names"]
    if isinstance(names_raw, dict):
        names = [str(names_raw[idx]) for idx in sorted(int(k) for k in names_raw.keys())]
    elif isinstance(names_raw, list):
        names = [str(item) for item in names_raw]
    else:
        raise ValueError("`names` in dataset YAML must be dict or list.")
    val_images = _collect_images_from_split(root, data["val"])
    return DatasetSpec(yaml_path=yaml_path, root=root, names=names, val_images=val_images)


def _collect_images_from_split(root: Path, split_value: Any) -> list[Path]:
    candidates: list[Path] = []
    if isinstance(split_value, (str, Path)):
        candidates.extend(_collect_images_from_path(resolve_path(root, split_value)))
    elif isinstance(split_value, list):
        for item in split_value:
            candidates.extend(_collect_images_from_path(resolve_path(root, item)))
    else:
        raise ValueError("Unsupported split format in dataset YAML.")
    if not candidates:
        raise FileNotFoundError("No validation images were found from dataset YAML.")
    return sorted(set(path.resolve() for path in candidates))


def _collect_images_from_path(path: Path) -> list[Path]:
    if path.is_dir():
        return [item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES]
    if path.is_file() and path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return [Path(line.strip()).resolve() for line in handle if line.strip()]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path.resolve()]
    raise FileNotFoundError(f"Unsupported image split path: {path}")


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain `images/`: {image_path}") from exc
    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def load_yolo_ground_truth(label_path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    gt: list[dict[str, Any]] = []
    if not label_path.exists():
        return gt

    img_w, img_h = image_size
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            items = stripped.split()
            if len(items) < 5:
                continue
            cls_id = int(float(items[0]))
            xc = float(items[1]) * img_w
            yc = float(items[2]) * img_h
            bw = float(items[3]) * img_w
            bh = float(items[4]) * img_h
            x1 = xc - bw / 2.0
            y1 = yc - bh / 2.0
            x2 = xc + bw / 2.0
            y2 = yc + bh / 2.0
            gt.append({"cls_id": cls_id, "bbox": (x1, y1, x2, y2)})
    return gt


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(area_a + area_b - inter, 1e-6))


def greedy_match(
    gt: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt_item in enumerate(gt):
        for pred_idx, pred_item in enumerate(preds):
            iou = box_iou(gt_item["bbox"], pred_item["bbox"])
            if iou >= iou_threshold:
                pairs.append((iou, gt_idx, pred_idx))

    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx))

    unmatched_gt = [idx for idx in range(len(gt)) if idx not in used_gt]
    unmatched_pred = [idx for idx in range(len(preds)) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def _normalize_model_names(names_raw: Any) -> dict[int, str]:
    if isinstance(names_raw, dict):
        return {int(key): str(value) for key, value in names_raw.items()}
    if isinstance(names_raw, list):
        return {idx: str(value) for idx, value in enumerate(names_raw)}
    raise ValueError("Unsupported model names structure.")


def _prepare_alias_map(alias_path: str | Path | None) -> dict[str, str]:
    if not alias_path:
        return {}
    raw = load_json(alias_path)
    return {str(key): str(value) for key, value in raw.items()}


def _remap_prediction_class(
    cls_id: int,
    model_names: dict[int, str],
    dataset_name_to_id: dict[str, int],
    alias_map: dict[str, str],
) -> int | None:
    source_name = model_names.get(cls_id, str(cls_id))
    target_name = alias_map.get(source_name, source_name)
    return dataset_name_to_id.get(target_name)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool,
) -> None:
    data = matrix.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    labels = class_names + ["background"]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 0.9)))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text = f"{value:.2f}" if normalize else str(int(value))
            ax.text(col, row, text, ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_per_class_csv(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bg_idx = len(class_names)
    for cls_id, cls_name in enumerate(class_names):
        tp = float(matrix[cls_id, cls_id])
        fp = float(matrix[:, cls_id].sum() - tp)
        fn = float(matrix[cls_id, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "missed_as_background": int(matrix[cls_id, bg_idx]),
                "background_false_positive": int(matrix[bg_idx, cls_id]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def evaluate_model(
    weights: str | Path,
    data: str | Path,
    outdir: str | Path,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    imgsz: int = 960,
    device: str = "0",
    alias_map_path: str | Path | None = None,
) -> dict[str, Any]:
    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_spec(data)
    model = build_model(str(weights))
    model_core = getattr(model, "model", None)
    model_names = _normalize_model_names(getattr(model_core, "names", getattr(model, "names", {})))
    dataset_name_to_id = {name: idx for idx, name in enumerate(dataset.names)}
    alias_map = _prepare_alias_map(alias_map_path)

    matrix = np.zeros((len(dataset.names) + 1, len(dataset.names) + 1), dtype=np.int64)
    bg_idx = len(dataset.names)

    for image_path in tqdm(dataset.val_images, desc="Evaluating", unit="img"):
        with Image.open(image_path) as image:
            image_size = image.size

        gt_items = load_yolo_ground_truth(image_to_label_path(image_path), image_size)
        pred_items: list[dict[str, Any]] = []

        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                remapped_cls_id = _remap_prediction_class(cls_id, model_names, dataset_name_to_id, alias_map)
                if remapped_cls_id is None:
                    continue
                xyxy = box.xyxy[0].tolist()
                pred_items.append(
                    {
                        "cls_id": remapped_cls_id,
                        "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    }
                )

        matches, unmatched_gt, unmatched_pred = greedy_match(gt_items, pred_items, iou_threshold)
        for gt_idx, pred_idx in matches:
            gt_cls = int(gt_items[gt_idx]["cls_id"])
            pred_cls = int(pred_items[pred_idx]["cls_id"])
            matrix[gt_cls, pred_cls] += 1
        for gt_idx in unmatched_gt:
            gt_cls = int(gt_items[gt_idx]["cls_id"])
            matrix[gt_cls, bg_idx] += 1
        for pred_idx in unmatched_pred:
            pred_cls = int(pred_items[pred_idx]["cls_id"])
            matrix[bg_idx, pred_cls] += 1

    val_results = model.val(
        data=str(Path(data).resolve()),
        split="val",
        conf=conf,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=False,
        save_json=False,
    )
    box_metrics = getattr(val_results, "box", None)
    summary = {
        "weights": str(Path(weights).resolve()),
        "data": str(Path(data).resolve()),
        "images_evaluated": len(dataset.val_images),
        "conf": conf,
        "iou_threshold": iou_threshold,
        "imgsz": imgsz,
        "device": device,
        "map50": float(getattr(box_metrics, "map50", 0.0)) if box_metrics is not None else 0.0,
        "map75": float(getattr(box_metrics, "map75", 0.0)) if box_metrics is not None else 0.0,
        "map50_95": float(getattr(box_metrics, "map", 0.0)) if box_metrics is not None else 0.0,
        "precision": float(getattr(box_metrics, "mp", 0.0)) if box_metrics is not None else 0.0,
        "recall": float(getattr(box_metrics, "mr", 0.0)) if box_metrics is not None else 0.0,
    }

    per_class_rows = _save_per_class_csv(matrix, dataset.names, outdir_path / "per_class_metrics.csv")
    save_confusion_matrix(matrix, dataset.names, outdir_path / "confusion_matrix.png", normalize=False)
    save_confusion_matrix(
        matrix,
        dataset.names,
        outdir_path / "confusion_matrix_normalized.png",
        normalize=True,
    )

    with (outdir_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **summary,
                "per_class": per_class_rows,
                "confusion_matrix": matrix.tolist(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        **summary,
        "outdir": str(outdir_path),
    }
