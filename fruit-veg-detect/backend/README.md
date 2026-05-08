 # Backend (FastAPI)

## 1. Install

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 3. Config

Environment variables:

- `MODEL_PATH` default `backend/app/data/model/fruit/best.pt` and kept for backward compatibility
- `FRUIT_MODEL_PATH` default `backend/app/data/model/fruit/best.pt`
- `VEGETABLE_MODEL_PATH` default `backend/app/data/model/vegetable/best.pt`
- `CONF_THRES` default `0.25`
- `IOU_THRES` default `0.45`
- `DEVICE` default `cpu`
- `SAVE_EMPTY_RESULTS` default `false`
- `RECORD_RETENTION_LIMIT` default `50`
- `VIDEO_SAMPLE_INTERVAL` default `10`
- `TRACKER_MAX_AGE` default `30`
- `TRACKER_N_INIT` default `2`
- `TRACKER_MAX_IOU_DISTANCE` default `0.7`
- `TRACKER_MAX_TIME_SINCE_UPDATE` default `5`
- `TRACKER_TRAJECTORY_LEN` default `64`
- `TRACKER_FALLBACK_IOU` default `0.3`
- `CAMERA_TRACKER_SESSION_TTL_SEC` default `120`

You can copy `backend/.env.example` to `backend/.env` and edit it.

Example:

```bash
# Windows PowerShell
$env:FRUIT_MODEL_PATH="D:\\models\\fruit\\best.pt"
$env:VEGETABLE_MODEL_PATH="D:\\models\\vegetable\\best.pt"
uvicorn app.main:app --reload
```

## 4. APIs

- `GET /health`
- `POST /api/detect/image`
- `POST /api/detect/video`
- `POST /api/detect/camera/frame`
- `POST /api/detect/camera/session`
- `POST /api/detect/camera/reset`
- `GET /api/records/images`
- `GET /api/records/videos`
- `DELETE /api/records/{record_id}`
- `GET /api/users`
- `POST /api/users`
- `GET /api/profile`

## 5. Static Files

- `/static/uploads/...`
- `/static/outputs/...`

Video and camera detect endpoints support multi-object tracking with DeepSORT and return per-object `track_id` plus trajectory summary.
If DeepSORT dependency is unavailable, backend falls back to a naive IoU tracker while keeping the same API shape.
If a selected model is missing/unavailable, that model runs in mock mode and returns empty boxes.
When `SAVE_EMPTY_RESULTS=false`, image/video requests with no detections will not be persisted to `uploads/` or `outputs/`.
When `RECORD_RETENTION_LIMIT=50`, backend keeps only the latest 50 records and deletes files linked to older records. On startup it also removes orphaned files left in `uploads/` and `outputs/`.
