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
- 前后端流程仍可运行；若 `SAVE_EMPTY_RESULTS=false`，空结果不会写入记录或保存文件

## 四、存储清理策略

- 默认只保留最近 `50` 条识别记录
- 超过上限时，后端会自动删除更早的记录以及对应的上传文件、标注图、关键帧、输出视频
- 后端启动时也会清理 `backend/app/data/uploads` 和 `backend/app/data/outputs` 下不再被记录引用的孤儿文件
- 可通过 `backend/.env` 中的 `RECORD_RETENTION_LIMIT` 调整保留数量，设置为 `0` 或负数可关闭该限制

## 五、核心接口

- `GET /health`
- `POST /api/detect/image`（图片检测）
- `POST /api/detect/video`（视频采样检测 + 输出视频/关键帧）
  - 返回 `track_summaries` 与 `trajectories`，可用于查看跨帧移动轨迹
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

## 八、后续替换真实推理

当前 `backend/app/services/detector.py` 已与 API 层解耦。你后续可：

1. 保留 `predict_image()` / `draw_boxes()` 返回结构不变
2. 在内部替换为你自己的 YOLOv5 `detect` 流程
3. 前端和现有路由无需改动
