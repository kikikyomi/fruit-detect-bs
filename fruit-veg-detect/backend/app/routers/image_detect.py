from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.services.detector import detector_service
from app.services.record_store import create_record
from app.utils.cv import decode_image_bytes, save_image, unique_name

router = APIRouter(prefix="/api/detect", tags=["detect-image"])


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    model_key: str | None = Form(default=None),
) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = detector_service.predict_image(image, conf=conf, iou=iou, model_key=model_key)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"Inference failed: {result['error']}")

    if not result["boxes"] and not settings.SAVE_EMPTY_RESULTS:
        return {
            "record_id": None,
            "result": result,
            "annotated_image_url": "",
            "input_image_url": "",
            "saved": False,
        }

    input_name = unique_name(file.filename)
    input_path = settings.UPLOAD_DIR / input_name
    input_path.write_bytes(content)
    input_url = f"/static/uploads/{input_name}"

    annotated = detector_service.draw_boxes(image, result["boxes"])

    stem = Path(input_name).stem
    output_name = f"annotated_{stem}.jpg"
    output_path = settings.OUTPUT_DIR / output_name
    save_image(output_path, annotated)
    output_url = f"/static/outputs/{output_name}"

    record_id = create_record(
        record_type="image",
        file_name=file.filename,
        input_path=input_path,
        input_url=input_url,
        output_path=output_path,
        output_url=output_url,
        result=result,
    )

    return {
        "record_id": record_id,
        "result": result,
        "annotated_image_url": output_url,
        "input_image_url": input_url,
        "saved": True,
    }
