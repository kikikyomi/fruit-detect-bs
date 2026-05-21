# 果蔬识别系统（前后端外壳）

本项目提供可直接运行的“果蔬识别系统”骨架：

- 前端：Vue3 + Vite + TypeScript + Element Plus
- 后端：FastAPI + OpenCV + 可替换推理服务（默认 Ultralytics YOLO 风格）
- 跟踪：DeepSORT 多目标跟踪（视频接口返回每个目标独立 `track_id` 与轨迹）
- 模型：预留 `.pt` 权重入口，未放置模型时自动降级为 Mock（返回空框但流程可运行）

## 环境与迁移

- 跨电脑部署、环境版本、常见问题：`ENVIRONMENT.md`
- 后端环境变量模板：`backend/.env.example`
- 前端环境变量模板：`frontend/.env.example`

## 目录结构

```text
fruit-veg-detect/
  backend/
    app/
      main.py
      core/config.py
      services/detector.py
      services/record_store.py
      routers/
        health.py
        image_detect.py
        video_detect.py
        camera_detect.py
        records.py
        users.py
      utils/cv.py
      data/
        uploads/
        outputs/
        records.sqlite3
    requirements.txt
    README.md
  scripts/
    inference.py
    track_deepsort.py
    benchmark_camera.py
  frontend/
    src/
      main.ts
      router/index.ts
      layout/AppLayout.vue
      components/
        UploadPanel.vue
        DetectCanvas.vue
        ResultTable.vue
      pages/
        ImageDetect.vue
        VideoDetect.vue
        CameraDetect.vue
        ImageRecords.vue
        VideoRecords.vue
        Users.vue
        Profile.vue
      api/
        client.ts
        detect.ts
        records.ts
        users.ts
    vite.config.ts
    package.json
    README.md
```

## 一、启动后端

### 一键启动（Windows）

双击项目根目录下的 `start.bat`，脚本会自动启动后端、前端并打开浏览器：

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

使用系统时保持弹出的后端和前端命令窗口不要关闭。

### 手动启动

```bash
cd fruit-veg-detect/backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：`http://localhost:8000/health`

## 二、启动前端

```bash
cd fruit-veg-detect/frontend
npm install
npm run dev
```

访问：`http://localhost:5173`

## 三、模型配置（重点）

### 1. 默认模型路径

后端默认读取：

```text
backend/app/data/model/fruit/best.pt       # 水果模型
backend/app/data/model/vegetable/best.pt   # 蔬菜模型
```

你可以直接把水果模型放到 `model/fruit/best.pt`，把新训练好的蔬菜模型放到 `model/vegetable/best.pt`。

### 2. 用环境变量替换模型路径

PowerShell 示例：

```bash
cd fruit-veg-detect/backend
$env:FRUIT_MODEL_PATH="D:\models\fruit\best.pt"
$env:VEGETABLE_MODEL_PATH="D:\models\vegetable\best.pt"
$env:CONF_THRES="0.25"
$env:IOU_THRES="0.45"
$env:DEVICE="cpu"
uvicorn app.main:app --reload
```

### 3. 未放模型时行为

- `/health` 显示 `model_loaded: false`
- 检测接口返回空 `boxes`
- 前后端流程仍可运行；若 `SAVE_EMPTY_RESULTS=false`，图片空结果不会写入记录或保存文件；视频任务仍会保留 `output.mp4`、CSV、JSON 和 summary 方便复查

## 四、存储清理策略

- 默认只保留最近 `50` 条识别记录
- 超过上限时，后端会自动删除更早的记录以及对应的上传文件、标注图、关键帧、输出视频
- 后端启动时也会清理 `backend/app/data/uploads` 和 `backend/app/data/outputs` 下不再被记录引用的孤儿文件
- 可通过 `backend/.env` 中的 `RECORD_RETENTION_LIMIT` 调整保留数量，设置为 `0` 或负数可关闭该限制

## 五、核心接口

- `GET /health`
- `POST /api/detect/image`（图片检测）
- `POST /api/detect/video`（创建视频检测/跟踪后台任务，返回 `task_id`）
- `GET /api/detect/video/tasks/{task_id}`（查询视频处理进度和输出文件）
- `POST /api/detect/camera/session`（创建摄像头跟踪会话）
- `POST /api/detect/camera/frame`（摄像头帧检测+跟踪，返回 `track_id` 和轨迹）
- `POST /api/detect/camera/reset`（结束会话并释放跟踪状态）
- `GET /api/records/images`
- `GET /api/records/videos`
- `DELETE /api/records/{record_id}`
- `GET /api/users`
- `POST /api/users`
- `GET /api/profile`

静态资源访问：

- `/static/uploads/...`
- `/static/outputs/...`

## 六、摄像头方案说明

已实现方案 A：前端 `getUserMedia` 拉流，默认请求 640×480 画面并以 JPEG quality 0.75 截帧调用 `/api/detect/camera/frame`。摄像头接口默认使用 `imgsz=416`、`conf=0.35`，并通过会话化跟踪接口返回检测框、`track_id` 和轨迹，前端 canvas 叠加显示。

- 支持开始/停止
- 内置单请求在途保护（避免请求堆积）
- 摄像头实时模式默认不保存图片、视频、CSV 或数据库记录

## 七、摄像头性能测试脚本

从项目根目录运行：

```bash
python scripts/inference.py --weights best.pt --source 0 --imgsz 416 --conf 0.35 --device 0 --half --camera-width 640 --camera-height 480
python scripts/track_deepsort.py --weights best.pt --source 0 --imgsz 416 --conf 0.35 --device 0 --half --camera-width 640 --camera-height 480 --frame-skip 2 --max-age 15 --n-init 2
python scripts/track_deepsort.py --weights best.pt --source 0 --imgsz 416 --conf 0.35 --device 0 --half --disable-deepsort
python scripts/benchmark_camera.py --weights best.pt --source 0 --imgsz 416 --frame-skip 2 --device 0 --half
```

`benchmark_camera.py` 会输出平均 FPS、YOLO/DeepSORT/绘制平均耗时、总帧数、有效检测帧数和 CPU/GPU 设备信息，并保存到：

- `runs/benchmark_camera/result.txt`
- `runs/benchmark_camera/result.csv`

## 八、视频检测与视频跟踪优化

视频链路已支持 1080p 上传、YOLO 单独检测、YOLO + DeepSORT 跟踪、输出标注视频、CSV/JSON 结果文件和前端任务进度轮询。默认采用“ID稳定优先”参数，优先保证同一目标 TrackID 不频繁变化。

### 1. 分辨率和画面内容

- 输入视频会读取原始 `width / height / fps / total_frames / duration`。
- 默认输出分辨率为 `1280×720`，1080p 输入默认降到 720p 输出，用于提升处理速度并减小文件体积。
- `--keep-original-resolution` 会保持原始分辨率，例如 1920×1080。
- `--imgsz` 是 YOLO 模型推理尺寸，不是输出视频分辨率。
- `--output-width / --output-height` 是最终 `output.mp4` 分辨率。
- 非 16:9 输入不会被强行拉伸，系统会等比例缩放并 letterbox 填充，检测框和轨迹点会同步映射到输出坐标。
- 默认输出画面只显示检测框、类别、置信度、TrackID 和轨迹线，不显示 FPS、Frame、Progress、帧数或进度。
- 显式传入 `--show-stats` 后，才会把 FPS、帧号和进度画到输出视频上。

### 2. 前端使用

视频检测页上传视频后会创建后台任务，页面轮询 `task_id` 显示：

- `status`: `waiting / running / finished / failed`
- `progress`
- `current_frame / total_frames`
- 当前处理 FPS
- 输出视频、`results.csv`、`results.json`、`summary.txt` 链接

前端默认只展示核心跟踪参数；输出宽度、输出高度、保持原始分辨率、`imgsz`、`max_cosine_distance`、`nn_budget`、`min_box_area`、`max_center_jump` 收在“高级设置”里，默认收起。

前端提供三个预设：

- ID稳定优先（默认）：`imgsz=512`，`conf=0.30`，`frame_skip=1`，`max_age=40`，`max_cosine_distance=0.25`，`nn_budget=100`，`trail_length=30`，`smooth_window=5`，输出 1280×720。
- 平衡：`imgsz=512`，`conf=0.35`，`frame_skip=1`，`max_age=25`，`max_cosine_distance=0.30`，`nn_budget=80`，`trail_length=25`，`smooth_window=3`，输出 1280×720。
- 速度优先：`imgsz=416`，`conf=0.40`，`frame_skip=2`，`max_age=20`，`max_cosine_distance=0.35`，`nn_budget=50`，`trail_length=20`，`smooth_window=3`，输出 1280×720。

### 3. 命令行示例

```bash
python scripts/inference.py --weights best.pt --source test.mp4 --imgsz 512 --conf 0.35 --iou 0.45 --device 0 --half --save-video --save-csv --save-json --output-width 1280 --output-height 720

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --save-video --save-csv --save-json

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --save-video --show-stats

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --imgsz 512 --conf 0.35 --iou 0.45 --device 0 --half --frame-skip 1 --max-age 25 --n-init 3 --max-cosine-distance 0.30 --nn-budget 80 --trail-length 25 --smooth-window 3 --save-video --save-csv --save-json --output-width 1280 --output-height 720

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --imgsz 416 --conf 0.40 --iou 0.45 --device 0 --half --frame-skip 2 --max-age 20 --n-init 2 --trail-length 20 --save-video --output-width 1280 --output-height 720

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --imgsz 512 --conf 0.35 --iou 0.45 --device 0 --half --start-time 10 --end-time 30 --save-video --save-csv --save-json --output-width 1280 --output-height 720

python scripts/track_deepsort.py --weights best.pt --source test.mp4 --imgsz 512 --conf 0.35 --device 0 --half --save-video --output-width 1280 --output-height 720 --show-stats

python scripts/benchmark_video.py --weights best.pt --source test.mp4 --imgsz 512 --conf 0.35 --iou 0.45 --device 0 --half --frame-skip 1 --output-width 1280 --output-height 720
```

### 4. 输出文件

每次命令行任务按时间戳输出到：

- `runs/video_detect/YYYYMMDD_HHMMSS/output.mp4`
- `runs/video_detect/YYYYMMDD_HHMMSS/results.csv`
- `runs/video_detect/YYYYMMDD_HHMMSS/results.json`
- `runs/video_detect/YYYYMMDD_HHMMSS/summary.txt`
- `runs/video_track/YYYYMMDD_HHMMSS/...`

后端前端任务输出到：

- `backend/app/data/outputs/video_tasks/{task_id}/output.mp4`
- `backend/app/data/outputs/video_tasks/{task_id}/results.csv`
- `backend/app/data/outputs/video_tasks/{task_id}/results.json`
- `backend/app/data/outputs/video_tasks/{task_id}/summary.txt`

CSV/JSON 主要字段：

- `frame_id`, `timestamp`, `is_detected_frame`
- `track_id`（跟踪模式）
- `class_id`, `class_name`, `confidence`
- `bbox_xyxy`, `bbox_xywh`
- `center_x`, `center_y`
- `output_bbox_xyxy`, `output_center_x`, `output_center_y`

`summary.txt` 包含输入/输出路径、权重、原始帧数、处理帧范围、FPS、原始/输出分辨率、`imgsz/conf/iou/frame_skip`、DeepSORT 是否启用、`show_stats`、检测数、TrackID 数、平均处理 FPS、YOLO/DeepSORT/绘制/读取/写入耗时和总耗时。

### 5. 常见问题

- 输出视频打不开：优先确认 OpenCV 的 `mp4v` 写入成功，输出路径是否存在，播放器是否支持 mp4v。
- 检测速度慢：先确认 CUDA/half 生效；再使用速度优先预设，把 `imgsz` 降到 416 或 `frame_skip` 调到 2。
- ID 频繁变化：先保持 `frame_skip=1`，把 `conf` 降到 0.30，`max_age` 增大到 40 或 50，`max_cosine_distance` 保持 0.25 到 0.30，`nn_budget` 增大到 100。
- 轨迹线太乱：减小 `trail_length` 到 20 到 30，增大 `smooth_window` 到 3 到 5，并设置 `max_center_jump` 过滤异常跳点。
- 输出视频分辨率不对：检查是否传了 `--keep-original-resolution`；否则默认按 `--output-width / --output-height` 输出。
- 检测框位置偏移：不要把 `imgsz` 当输出分辨率；当前坐标按原始帧检测，再映射到 letterbox 后的输出帧。
- 视频处理后变短：确认 `frame_skip` 只影响检测频率，不会跳过写入帧；若使用 `--start-time/--end-time`，输出只包含指定片段。
- CSV 和视频帧对不上：使用 `frame_id` 和 `timestamp` 对齐，`is_detected_frame=false` 表示该帧复用了上一检测帧或跟踪预测。
- 1080p 视频检测很慢：默认输出 720p 只能降低绘制/编码成本，YOLO 速度主要由 `imgsz`、模型大小、设备和 `frame_skip` 决定。

## 九、后续替换真实推理

当前 `backend/app/services/detector.py` 已与 API 层解耦。你后续可：

1. 保留 `predict_image()` / `draw_boxes()` 返回结构不变
2. 在内部替换为你自己的 YOLOv5 `detect` 流程
3. 前端和现有路由无需改动
