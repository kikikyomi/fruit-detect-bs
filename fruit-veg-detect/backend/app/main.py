from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers.camera_detect import router as camera_router
from app.routers.health import router as health_router
from app.routers.image_detect import router as image_router
from app.routers.records import router as records_router
from app.routers.users import router as users_router
from app.routers.video_detect import router as video_router
from app.services.detector import detector_service
from app.services.record_store import apply_storage_retention, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fruit Veg Detect API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")
app.mount("/static/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")

app.include_router(health_router)
app.include_router(image_router)
app.include_router(video_router)
app.include_router(camera_router)
app.include_router(records_router)
app.include_router(users_router)


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_dirs()
    init_db()
    cleanup_result = apply_storage_retention(clean_orphans=True)
    if cleanup_result["deleted_records"] or cleanup_result["deleted_files"]:
        logger.info(
            "Storage retention applied: deleted_records=%s deleted_files=%s",
            cleanup_result["deleted_records"],
            cleanup_result["deleted_files"],
        )
    detector_service.load_models()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Fruit Veg Detect Backend Running"}
