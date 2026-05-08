from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return cleaned or "file"


def unique_name(filename: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = sanitize_filename(filename)
    return f"{stamp}_{safe}"


def decode_image_bytes(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise RuntimeError(f"Failed to save image to {path}: {exc}") from exc
