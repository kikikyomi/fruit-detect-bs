from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw
import win32com.client


PPT_PATH = Path(r"D:\毕设\fruit-veg-dete\presentation\fruit_veg_defense.pptx")
BACKUP_PATH = PPT_PATH.with_name("fruit_veg_defense.before_layout_fix.pptx")
TEMP_DIR = PPT_PATH.parent / "_normalized_assets"

RAW_COVER_IMAGE = Path(
    r"D:\微信\xwechat_files\wxid_102j5bzfkxyd22_1c5f\temp\RWTemp\2026-04\ad3cf603c3b1c80efd6be82dcf7a37ef.jpg"
)
ANNOTATED_IMAGE = Path(
    r"D:\毕设\fruit-veg-dete\fruit-veg-detect\backend\app\data\outputs\annotated_20260422_133451_396542_ad3cf603c3b1c80efd6be82dcf7a37ef.jpg"
)
DB_PATH = Path(r"D:\毕设\fruit-veg-dete\fruit-veg-detect\backend\app\data\records.sqlite3")
OUTPUT_DIR = Path(r"D:\毕设\fruit-veg-dete\fruit-veg-detect\backend\app\data\outputs")

MSO_BRING_TO_FRONT = 0


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def rgb(r: int, g: int, b: int) -> int:
    return r + (g << 8) + (b << 16)


def delete_shape_if_present(slide, name: str) -> None:
    for idx in range(slide.Shapes.Count, 0, -1):
        shape = slide.Shapes(idx)
        if shape.Name == name:
            shape.Delete()


def shape_exists(slide, name: str) -> bool:
    for idx in range(1, slide.Shapes.Count + 1):
        if slide.Shapes(idx).Name == name:
            return True
    return False


def get_shape(slide, name: str):
    for idx in range(1, slide.Shapes.Count + 1):
        shape = slide.Shapes(idx)
        if shape.Name == name:
            return shape
    raise KeyError(f"Shape not found on slide {slide.SlideIndex}: {name}")


def latest_video_keyframe() -> Path:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT summary_json
        FROM records
        WHERE type = 'video'
        ORDER BY id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        raise RuntimeError("Could not find latest video record.")

    summary = json.loads(row[0])
    details = summary.get("keyframe_details") or []
    if not details:
        raise RuntimeError("Latest video record does not contain keyframe details.")

    image_url = details[0].get("image_url")
    if not isinstance(image_url, str) or not image_url.startswith("/static/outputs/"):
        raise RuntimeError(f"Unexpected keyframe image url: {image_url}")

    keyframe_path = OUTPUT_DIR / image_url.replace("/static/outputs/", "", 1)
    assert_exists(keyframe_path)
    return keyframe_path


def crop_to_ratio(image: Image.Image, target_ratio: float, focus_x: float = 0.5, focus_y: float = 0.5) -> Image.Image:
    width, height = image.size
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 1e-6:
        return image

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left_max = width - new_width
        left = int(left_max * focus_x)
        left = max(0, min(left, left_max))
        return image.crop((left, 0, left + new_width, height))

    new_height = int(width / target_ratio)
    top_max = height - new_height
    top = int(top_max * focus_y)
    top = max(0, min(top, top_max))
    return image.crop((0, top, width, top + new_height))


def rounded_asset(
    source: Path,
    out_path: Path,
    width_pt: float,
    height_pt: float,
    *,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
    scale: int = 4,
    radius_px: int = 48,
) -> Path:
    target_width = max(1, int(round(width_pt * scale)))
    target_height = max(1, int(round(height_pt * scale)))
    target_ratio = target_width / target_height

    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        cropped = crop_to_ratio(rgba, target_ratio, focus_x=focus_x, focus_y=focus_y)
        fitted = cropped.resize((target_width, target_height), Image.LANCZOS)

    mask = Image.new("L", (target_width, target_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [(0, 0), (target_width - 1, target_height - 1)],
        radius=max(8, radius_px),
        fill=255,
    )
    fitted.putalpha(mask)
    fitted.save(out_path)
    return out_path


def style_image_frame(shape) -> None:
    shape.Fill.Visible = True
    shape.Fill.ForeColor.RGB = rgb(255, 255, 255)
    shape.Line.Visible = True
    shape.Line.ForeColor.RGB = rgb(218, 224, 234)
    shape.Line.Weight = 1


def insert_picture(slide, image_path: Path, left: float, top: float, width: float, height: float, name: str):
    picture = slide.Shapes.AddPicture(str(image_path), False, True, left, top, width, height)
    picture.Name = name
    return picture


def format_caption(shape, left: float, top: float, width: float) -> None:
    shape.Left = left
    shape.Top = top
    shape.Width = width
    text_range = shape.TextFrame.TextRange
    text_range.Font.Size = 10.5
    text_range.Font.Color.RGB = rgb(108, 118, 136)


def fix_slide1(presentation, slide_index: int) -> None:
    slide = presentation.Slides(slide_index)
    frame = get_shape(slide, "Rounded Rectangle 2")
    frame.Left = 510
    frame.Top = 88
    frame.Width = 405
    frame.Height = 340
    style_image_frame(frame)
    left, top, width, height = frame.Left, frame.Top, frame.Width, frame.Height

    if shape_exists(slide, "CoverImage"):
        return

    delete_shape_if_present(slide, "CoverImage")
    slide = presentation.Slides(slide_index)
    cover_asset = rounded_asset(
        RAW_COVER_IMAGE,
        TEMP_DIR / "cover.png",
        width,
        height,
        focus_x=0.52,
        focus_y=0.0,
        radius_px=54,
    )
    insert_picture(slide, cover_asset, left, top, width, height, "CoverImage")


def fix_slide5(presentation, slide_index: int) -> None:
    slide = presentation.Slides(slide_index)
    frame = get_shape(slide, "Rounded Rectangle 7")
    frame.Left = 400
    frame.Top = 96
    frame.Width = 500
    frame.Height = 330
    style_image_frame(frame)
    left, top, width, height = frame.Left, frame.Top, frame.Width, frame.Height

    if shape_exists(slide, "Slide5Image"):
        caption = get_shape(slide, "TextBox 9")
        format_caption(caption, left + 10, top + height + 2, width - 20)
        caption.ZOrder(MSO_BRING_TO_FRONT)
        return

    delete_shape_if_present(slide, "Picture 8")
    delete_shape_if_present(slide, "Slide5Image")
    slide = presentation.Slides(slide_index)
    annotated_asset = rounded_asset(
        ANNOTATED_IMAGE,
        TEMP_DIR / "slide5_detection.png",
        width,
        height,
        focus_x=0.50,
        focus_y=0.45,
        radius_px=46,
    )
    insert_picture(slide, annotated_asset, left, top, width, height, "Slide5Image")

    slide = presentation.Slides(slide_index)
    caption = get_shape(slide, "TextBox 9")
    format_caption(caption, left + 10, top + height + 2, width - 20)
    caption.ZOrder(MSO_BRING_TO_FRONT)


def fix_slide6(presentation, slide_index: int, keyframe_path: Path) -> None:
    slide = presentation.Slides(slide_index)
    frame = get_shape(slide, "Rounded Rectangle 4")
    style_image_frame(frame)
    left, top, width, height = frame.Left, frame.Top, frame.Width, frame.Height

    already_has_image = shape_exists(slide, "Slide6Image")

    if not already_has_image:
        delete_shape_if_present(slide, "Picture 5")
        delete_shape_if_present(slide, "Slide6Image")
        slide = presentation.Slides(slide_index)
        keyframe_asset = rounded_asset(
            keyframe_path,
            TEMP_DIR / "slide6_keyframe.png",
            width,
            height,
            focus_x=0.48,
            focus_y=0.42,
            radius_px=42,
        )
        insert_picture(slide, keyframe_asset, left, top, width, height, "Slide6Image")

    slide = presentation.Slides(slide_index)
    caption = get_shape(slide, "TextBox 6")
    format_caption(caption, left + 10, top + height - 18, width - 20)
    caption.ZOrder(MSO_BRING_TO_FRONT)

    slide = presentation.Slides(slide_index)
    text_box = get_shape(slide, "Rounded Rectangle 11")
    text_box.Top = 282
    text_box.Height = 144
    text_box.Width = 398
    text_box.TextFrame.MarginLeft = 14
    text_box.TextFrame.MarginRight = 14
    text_box.TextFrame.MarginTop = 8
    text_box.TextFrame.MarginBottom = 8
    text_range = text_box.TextFrame.TextRange
    text_range.Font.Size = 21.5
    text_range.Font.Color.RGB = rgb(46, 57, 75)


def main() -> None:
    assert_exists(PPT_PATH)
    assert_exists(RAW_COVER_IMAGE)
    assert_exists(ANNOTATED_IMAGE)
    assert_exists(DB_PATH)

    TEMP_DIR.mkdir(exist_ok=True)
    if not BACKUP_PATH.exists():
        shutil.copy2(PPT_PATH, BACKUP_PATH)

    keyframe_path = latest_video_keyframe()

    app = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = app.Presentations.Open(str(PPT_PATH), False, False, False)

    try:
        fix_slide1(presentation, 1)
        fix_slide5(presentation, 5)
        fix_slide6(presentation, 6, keyframe_path)
        presentation.Save()
    finally:
        presentation.Close()
        app.Quit()


if __name__ == "__main__":
    main()
