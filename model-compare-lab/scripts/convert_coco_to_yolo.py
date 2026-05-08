from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / image_width
    yc = (y + h / 2.0) / image_height
    nw = w / image_width
    nh = h / image_height
    return xc, yc, nw, nh


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO detection annotations to YOLO labels.")
    parser.add_argument("--images-root", required=True, help="Directory containing source images.")
    parser.add_argument("--annotations-json", required=True, help="COCO json file path.")
    parser.add_argument("--output-root", required=True, help="Output dataset root in YOLO format.")
    parser.add_argument("--split-name", required=True, help="Split name, such as train / val / test.")
    parser.add_argument(
        "--class-map",
        default="",
        help="Optional JSON mapping from source class name to target class name.",
    )
    parser.add_argument(
        "--image-subdir",
        default="",
        help="Optional extra subdir inside images root if COCO file names are relative.",
    )
    args = parser.parse_args()

    images_root = Path(args.images_root).resolve()
    annotations_path = Path(args.annotations_json).resolve()
    output_root = Path(args.output_root).resolve()
    split_name = args.split_name.strip()
    class_map_path = Path(args.class_map).resolve() if args.class_map else None
    image_subdir = Path(args.image_subdir) if args.image_subdir else Path(".")

    coco = load_json(annotations_path)
    class_map: dict[str, str] = {}
    if class_map_path:
        class_map = {str(k): str(v) for k, v in load_json(class_map_path).items()}

    category_id_to_name = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    target_names: list[str] = []
    target_name_to_id: dict[str, int] = {}

    def get_target_id(source_category_id: int) -> int | None:
        source_name = category_id_to_name[source_category_id]
        target_name = class_map.get(source_name, source_name)
        if not target_name:
            return None
        if target_name not in target_name_to_id:
            target_name_to_id[target_name] = len(target_names)
            target_names.append(target_name)
        return target_name_to_id[target_name]

    image_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        image_annotations[int(ann["image_id"])].append(ann)

    out_images = output_root / "images" / split_name
    out_labels = output_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_info in coco["images"]:
        image_id = int(image_info["id"])
        file_name = Path(str(image_info["file_name"]))
        src_image = (images_root / image_subdir / file_name).resolve()
        if not src_image.exists():
            raise FileNotFoundError(f"Image not found: {src_image}")

        dst_image = out_images / file_name.name
        shutil.copy2(src_image, dst_image)

        label_lines: list[str] = []
        for ann in image_annotations.get(image_id, []):
            target_cls = get_target_id(int(ann["category_id"]))
            if target_cls is None:
                continue
            xc, yc, bw, bh = convert_bbox_to_yolo(
                bbox=[float(v) for v in ann["bbox"]],
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
            )
            label_lines.append(
                f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        label_path = out_labels / f"{file_name.stem}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(label_lines))

    names_path = output_root / "class_names.json"
    with names_path.open("w", encoding="utf-8") as handle:
        json.dump(target_names, handle, ensure_ascii=False, indent=2)

    print(f"[done] YOLO dataset written to: {output_root}")
    print(f"[done] classes: {target_names}")


if __name__ == "__main__":
    main()

