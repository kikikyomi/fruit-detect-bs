# 项目环境与迁移说明

这份文档用于把本项目迁移到任意新电脑时快速恢复运行环境。

## 1. 运行环境要求

- 操作系统: Windows 10/11 或 Linux/macOS
- Python: 3.10-3.13 (推荐 3.11/3.12)
- Node.js: >= 18 (推荐 20 LTS)
- npm: >= 9
- 浏览器: Chrome/Edge 最新版 (摄像头识别页面依赖 getUserMedia)

可选:

- CUDA GPU 环境 (仅在你要用 GPU 推理时需要)

## 2. 依赖安装范围

- 后端依赖安装在 `backend/.venv` (推荐独立虚拟环境)
- 前端依赖安装在 `frontend/node_modules`
- DeepSORT 依赖 `deep-sort-realtime` 已在 `backend/requirements.txt` 中
- 上传文件目录: `backend/app/data/uploads`
- 输出文件目录: `backend/app/data/outputs`
- 识别记录库: `backend/app/data/records.sqlite3`

## 3. 新电脑一键恢复步骤

### 3.1 获取项目

把整个 `fruit-veg-detect` 文件夹拷贝到新电脑，或者通过 git clone 获取。

### 3.2 启动后端

```bash
cd fruit-veg-detect/backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

复制环境变量模板:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

运行后端:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

检查:

- 打开 `http://localhost:8000/health`

### 3.3 启动前端

```bash
cd fruit-veg-detect/frontend
npm install
```

复制环境变量模板:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

运行前端:

```bash
npm run dev
```

检查:

- 打开 `http://localhost:5173`

## 4. 模型文件放置与替换

两种方式:

- 方式 A: 放默认路径 `backend/app/data/model/best.pt`
- 方式 B: 在 `backend/.env` 设置 `MODEL_PATH=D:\\models\\your.pt`

后端健康检查会显示:

- `model_loaded: true` 代表模型加载成功
- `model_loaded: false` 代表模型未加载 (仍可运行壳流程)

## 5. 迁移时是否需要拷贝历史数据

按需求选择:

- 仅迁移代码: 不拷贝 `backend/app/data/*`，新机器会从空数据开始
- 连历史记录一起迁移: 额外拷贝 `backend/app/data/records.sqlite3`、`uploads/`、`outputs/`

## 6. 常见问题

1. `npm` 命令不存在: 未安装 Node.js，先安装 Node.js LTS。
2. `ModuleNotFoundError: cv2`: 后端依赖未安装到当前 Python 环境，激活 `.venv` 后重新 `pip install -r requirements.txt`。
3. `/health` 显示模型未加载: 检查 `.pt` 路径是否正确、文件是否存在、是否有读取权限。
4. 摄像头页面无画面: 确认浏览器已授权摄像头权限，并使用 `http://localhost` 访问前端。

## 7. 版本升级建议

Python 或 Node 版本变更时，建议重建环境:

1. 删除 `backend/.venv` 后重新创建并安装后端依赖。
2. 删除 `frontend/node_modules` 后重新 `npm install`。
