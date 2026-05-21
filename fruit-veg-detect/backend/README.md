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
- `TRACKER_BACKEND` default `deepsort`
- `TRACKER_MAX_AGE` default `80`
- `TRACKER_N_INIT` default `2`
- `TRACKER_MAX_IOU_DISTANCE` default `0.8`
- `TRACKER_MAX_COSINE_DISTANCE` default `0.5`
- `TRACKER_NN_BUDGET` default `150`
- `TRACKER_MAX_TIME_SINCE_UPDATE` default `80`
- `TRACKER_TRAJECTORY_LEN` default `40`
- `TRACKER_FALLBACK_IOU` default `0.3`
- `CAMERA_TRACKER_SESSION_TTL_SEC` default `120`
- Camera realtime defaults: `CAMERA_IMGSZ=480`, `CAMERA_CONF_THRES=0.15`, `CAMERA_WIDTH=640`, `CAMERA_HEIGHT=480`, `CAMERA_FRAME_SKIP=0`, `CAMERA_SHOW_FPS=true`, `CAMERA_DEBUG=false`, `CAMERA_SAVE_EVERY_FRAME=false`, `CAMERA_USE_DEEPSORT=true`, `CAMERA_DEVICE=auto`, `CAMERA_HALF=auto`, `CAMERA_TRACKER_MAX_AGE=80`, `CAMERA_TRACKER_N_INIT=2`, `CAMERA_TRACKER_MAX_COSINE_DISTANCE=0.5`, `CAMERA_TRACKER_NN_BUDGET=150`
- Video defaults are ID-stability-first: `VIDEO_IMGSZ=512`, `VIDEO_CONF_THRES=0.30`, `VIDEO_FRAME_SKIP=1`, `VIDEO_OUTPUT_WIDTH=1280`, `VIDEO_OUTPUT_HEIGHT=720`, `VIDEO_KEEP_ORIGINAL_RESOLUTION=false`, `VIDEO_SHOW_STATS=false`, `VIDEO_TRACKER_MAX_AGE=80`, `VIDEO_TRACKER_N_INIT=2`, `VIDEO_TRACKER_MAX_COSINE_DISTANCE=0.5`, `VIDEO_TRACKER_NN_BUDGET=150`, `VIDEO_TRAIL_LENGTH=40`, `VIDEO_SMOOTH_WINDOW=1`, `VIDEO_MIN_BOX_AREA=300`, `VIDEO_MAX_CENTER_JUMP=220`
- Smoothing defaults: `SMOOTHING_ENABLED=true`, `SMOOTHING_ALPHA=0.65`, `SMOOTHING_APPLY_TO_DISPLAY_ONLY=true`, `SMOOTHING_MAX_CENTER_JUMP=220`

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
- `POST /api/detect/video` creates a background video task and returns `task_id`
- `GET /api/detect/video/tasks/{task_id}` returns `status`, `progress`, `current_frame`, `total_frames`, processing FPS, output video URL, CSV URL, JSON URL, and summary URL
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

Video and camera detect endpoints support multi-object tracking with DeepSORT. Video output defaults to 720p for 1080p inputs, uses the ID-stability-first preset, and hides FPS/frame/progress overlays unless `show_stats=true`.
Tracked boxes are smoothed per `track_id` for display and trajectory drawing only; raw YOLO/DeepSORT boxes still drive tracker matching.
If DeepSORT dependency is unavailable, backend falls back to a naive IoU tracker while keeping the same API shape.
If a selected model is missing/unavailable, that model runs in mock mode and returns empty boxes.
When `SAVE_EMPTY_RESULTS=false`, image requests with no detections will not be persisted. Video tasks still keep their `output.mp4`, `results.csv`, `results.json`, and `summary.txt` so processing results can be reviewed.
When `RECORD_RETENTION_LIMIT=50`, backend keeps only the latest 50 records and deletes files linked to older records. On startup it also removes orphaned files left in `uploads/` and `outputs/`.
