# Model Compare Lab

这个目录专门用于“新模型训练与对比实验”，与现有 Web 系统隔离。

- 不会修改 `D:\毕设\fruit-veg-dete\fruit-veg-detect`
- 训练产物、混淆矩阵、对比指标都落在本目录
- 适合直接拷到服务器上跑，拿到新的 `best.pt`

## 推荐数据集

### 1. tomatOD

用途：
- 温室番茄检测
- 适合做单类番茄检测对比
- 也适合做 3 类成熟度检测

官方来源：
- GitHub: https://github.com/up2metric/tomatOD
- 说明：277 张图，2418 个标注，COCO 格式，类别为 `unripe / semi-ripe / fully-ripe`

官方下载：
- 图片: https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_images.zip
- 标注: https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_annotations.zip

推荐用途：
- 如果你想和当前 Web 项目里的旧模型直接做“番茄检测能力”对比，优先用 `single_class`
- 如果你想做更贴近采摘的“成熟度识别”实验，用 `ripeness`

### 2. LaboroTomato

用途：
- 更贴近采摘任务
- 支持不同大小番茄和成熟度
- 适合做更强的新模型

官方来源：
- GitHub: https://github.com/laboroai/LaboroTomato
- 说明：804 张图，6 类，适合目标检测和实例分割

官方下载：
- 主数据集: https://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip
- 大果子子集: https://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_big.zip
- 小果子子集: https://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato_little.zip

推荐用途：
- 如果你想训练一个“更像采摘模型”的新模型，优先考虑它
- 如果你只想做公平对比，也可以把 6 类折叠成单类 `tomato`

### 3. CitDet

用途：
- 柑橘果园采摘/检测
- 更接近果园机器人场景

官方来源：
- 项目页: https://robotic-vision-lab.github.io/citdet/
- GitHub: https://github.com/robotic-vision-lab/CitDet-A-Benchmark-Dataset-For-Citrus-Fruit-Detection
- 说明：579 张高分辨率图，32000+ 标注

说明：
- 该数据集更适合你后面往果园采摘方向扩展
- 当前代码已支持 COCO 转 YOLO；如果你下载到的是 COCO 风格标注，可直接转换

## 推荐对比策略

### 方案 A：和当前旧模型做直接公平对比

推荐数据：
- `tomatOD single_class`
- 或 `LaboroTomato single_class`

推荐模型：
- 新模型：`yolo11m.pt`
- 对照模型：当前 Web 系统的 `backend/app/data/model/best.pt`

原因：
- 旧模型里有番茄类
- 单类任务最容易做公平对照
- 4090 训练 `YOLO11m` 没有压力

### 方案 B：做更强、更贴近采摘场景的新模型

推荐数据：
- `LaboroTomato ripeness`
- 或你后续自己清洗的采摘数据

推荐模型：
- 主模型：`yolo11m.pt`
- 对照模型：`rtdetr-l.pt`

原因：
- `YOLO11m` 速度/精度平衡好
- `RT-DETR-L` 适合作为不同范式的对照组
- 两者都能输出 `best.pt`

## 目录结构

```text
model-compare-lab/
  README.md
  requirements.txt
  configs/
    datasets/
    experiments/
    class_maps/
  scripts/
    download_tomatod.py
    convert_coco_to_yolo.py
    train_model.py
    evaluate_model.py
    compare_models.py
  tools/
    ultra_helpers.py
    eval_utils.py
  artifacts/
```

## 服务器使用流程

### 1. 安装依赖

```bash
cd model-compare-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载 tomatOD

```bash
python scripts/download_tomatod.py --output-root data/raw/tomatod
```

### 3. 转成单类检测数据

```bash
python scripts/convert_coco_to_yolo.py \
  --images-root data/raw/tomatod/images \
  --annotations-json data/raw/tomatod/annotations/trainval.json \
  --output-root data/processed/tomatod_single_class \
  --split-name train \
  --class-map configs/class_maps/tomatod_single_class.json
```

如果官方标注已经拆分了 `train/val/test`，就对每个 split 各跑一次。

### 4. 训练新模型

```bash
python scripts/train_model.py \
  --config configs/experiments/yolo11m_tomatod_single.yaml
```

### 5. 评估旧模型

```bash
python scripts/evaluate_model.py \
  --weights ../fruit-veg-detect/backend/app/data/model/best.pt \
  --data configs/datasets/tomatod_single_class.yaml \
  --outdir artifacts/eval_old_best \
  --class-aliases configs/class_maps/current_best_to_tomato.json
```

### 6. 对比新旧模型

```bash
python scripts/compare_models.py \
  --data configs/datasets/tomatod_single_class.yaml \
  --weights ../fruit-veg-detect/backend/app/data/model/best.pt artifacts/train/yolo11m_tomatod_single/weights/best.pt \
  --names old_best yolo11m_new \
  --outdir artifacts/compare_tomatod_single \
  --alias-map old_best=configs/class_maps/current_best_to_tomato.json
```

## 输出产物

训练和评估结束后，会得到这些关键文件：

- `weights/best.pt`
- `metrics.json`
- `summary.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

## 说明

- 当前脚本默认面向“目标检测”对比实验
- 不会碰现有 Web 工程
- 如果你后面要做实例分割，再单独加 `seg` 实验即可
