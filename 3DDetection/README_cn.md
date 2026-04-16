# TabletopSeg3D

English documentation: [README.md](./README.md)

基于可插拔相机后端 + `YOLO Segmentation + Open3D` 的实时桌面目标三维检测工程。

![TabletopSeg3D 演示图](photo/1.png)

功能：

- 实时 Open3D 场景显示
- `headless` 无界面 JSON 输出

系统特性：

- 单相机实时运行，底层通过统一相机后端抽象接入
- 已支持 `realsense`，目前在 `D435i`、`D405` 上经过测试
- 已接入 `Orbbec Gemini2`，默认彩色 `640x480`，深度 `640x400`
- YOLO 实例分割
- 深度回投到点云
- 桌面对齐 OBB
  - 框的 `Z` 方向跟随桌面法向
  - 只输出 1 个旋转自由度 `yaw`
- 可选 Open3D 3D 标注
- `--no-display` 时输出每帧 JSON，包含：
  - 物体类别
  - 相机坐标系中心点
  - 物体尺寸
  - 框旋转角 `yaw`

## 目录

```text
TabletopSeg3D/
├── README.md
├── README_cn.md
├── requirements.txt
├── yolo11n-seg.pt
├── scripts/
│   └── realtime_open3d_scene.py
└── src/
    ├── camera/
    │   ├── factory.py
    │   ├── orbbec_backend.py
    │   ├── realsense_backend.py
    │   ├── realsense_capture.py
    │   └── types.py
    └── geometry/
        └── pointcloud.py
```

## 环境

推荐 Python `3.11`。

安装依赖：

```bash
cd ./3DDetection
python -m pip install -r requirements.txt
```

可选硬件依赖：

- RealSense：如果系统还没有 `librealsense` 对应运行环境，需要先安装 Intel RealSense SDK
- Orbbec：通过 `python -m pip install -r requirements-orbbec.txt` 安装

说明：

- Orbbec 的安装包名是 `pyorbbecsdk2`
- 代码里的导入名仍然是 `pyorbbecsdk`
- `requirements-orbbec.txt` 用来保持 Orbbec 为可选依赖，不强制所有环境安装 Orbbec SDK
- 当前分支在 RealSense 真机回归前不建议合并回 `main`

如果你希望使用 GPU 加速，还需要额外准备：

- NVIDIA 显卡驱动
- 与驱动匹配的 CUDA 运行环境
- GPU 版 `torch`
- GPU 版 `torchvision`

注意：

- 当前 `requirements.txt` 默认是 CPU 版本依赖
- 如果要启用 YOLO 的 GPU 推理，最关键的是把 `torch` 和 `torchvision` 换成与你本机 CUDA 版本匹配的 GPU 版本

## 查看已连接相机

先查看当前连接的相机型号和序列号：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py --list-devices --camera-backend auto
```

输出示例：

```text
Connected camera devices:
- backend=realsense | Intel RealSense D405 | serial=409122273421
- backend=realsense | Intel RealSense D435I | serial=419522072950
- backend=orbbec | Orbbec Gemini2 | serial=AYXXXXXXXXXX
```

## 模型

默认模型是仓库根目录下的：

```bash
yolo11n-seg.pt
```

这个工程支持用户自己更换 YOLO 模型，脚本通过 `--model` 参数加载权重。

可以使用的模型形式：

- 官方模型名，例如 `yolo11n-seg.pt`
- 仓库根目录下的本地权重，例如 `./my_model.pt`
- 训练输出目录里的权重，例如 `runs/segment/train/weights/best.pt`
- 任意绝对路径权重，例如 `/home/yourname/models/best.pt`

最推荐的更换方式有 2 种。

1. 直接替换仓库里的默认模型文件名  
如果你希望继续沿用默认命令，可以把你自己的模型放到仓库根目录，并在运行时显式指定：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --model ./my_model.pt \
  --device cpu
```

2. 直接传训练好的 `best.pt` 路径  
如果你的模型是自己训练出来的，最常见的用法就是：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --model runs/segment/train/weights/best.pt \
  --device cpu
```

如果你想在 `headless` 模式下使用自己的模型：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --model runs/segment/train/weights/best.pt \
  --device cpu \
  --frames 10 \
  --no-display
```

固定顺序目标表格：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial AY3Z331006L \
  --device cpu \
  --target-class mouse,banana \
  --target-table
```

表格会固定目标顺序。上面的例子里第 1 行永远是 `mouse`，第 2 行永远是 `banana`；如果某个目标没检测到，对应行显示 `null`。

如果只想在终端输出表格，不打开 Open3D 窗口，可以加 `--no-display`：

```bash
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial AY3Z331006L \
  --device cpu \
  --target-class mouse,banana \
  --target-table \
  --no-display
```

如果你使用 `D405`，也可以和自定义模型一起使用：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 409122273421 \
  --model runs/segment/train/weights/best.pt \
  --device cpu \
  --min-depth 0.02 \
  --max-depth 0.50
```

### 模型要求

这里必须使用 **实例分割模型**，也就是带 `mask` 输出的模型。

可以：

- `yolo11n-seg.pt`
- `yolo11s-seg.pt`
- 你自己训练得到的 `segment/best.pt`

不可以使用：

- 普通 `detect` 模型
- 只有分类输出的模型
- 没有 `mask` 的权重

原因是这个工程后面要做：

- `mask -> depth -> point cloud`
- 点云生成桌面对齐 OBB
- 输出中心点和 `yaw`

如果模型没有 `mask`，这条链路就无法工作。

### 建议

- 如果你只是想先跑通系统，优先使用 `yolo11n-seg.pt`
- 如果你有自己的桌面物体数据，建议训练 `YOLO segmentation` 模型再接入


## 实时显示

默认示例使用 `D435I`：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu
```

如果要开启 3D 标注：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu \
  --show-labels
```

`Orbbec Gemini2` 默认示例：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial <gemini2-serial> \
  --device cpu
```

`Gemini2` 默认流说明：

- 默认共同目标请求是 `640x480@30`
- 默认彩色流实际为 `640x480`
- 默认深度流实际为 `640x400`
- 实际启用分辨率会记录在后端返回的 `resolved_stream` 中

使用自定义模型：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --model runs/segment/train/weights/best.pt \
  --device cpu
```

## Headless 输出

无界面运行并输出每帧 JSON：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu \
  --frames 10 \
  --no-display
```

输出示例：

```json
{
  "frame_index": 0,
  "fps": 16.36,
  "infer_ms": 46.28,
  "geom_ms": 7.73,
  "scene_point_count": 67930,
  "table_normal_xyz": [0.039279, -0.227524, -0.97298],
  "detections": [
    {
      "class_name": "banana",
      "confidence": 0.792091,
      "center_camera_xyz_m": [0.031458, 0.085936, 0.402289],
      "extent_xyz_m": [0.17001, 0.064569, 0.040624],
      "yaw_rad": -0.826237,
      "yaw_deg": -47.3399,
      "point_count": 14090
    }
  ]
}
```

## D405 示例

`D405` 更偏近距离，建议收紧深度范围以减少计算负担：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 409122273421 \
  --device cpu \
  --min-depth 0.02 \
  --max-depth 0.50
```

如果要无界面输出：

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 409122273421 \
  --device cpu \
  --min-depth 0.02 \
  --max-depth 0.50 \
  --frames 10 \
  --no-display
```

## 主要参数

- `--camera-backend`：选择 `auto`、`realsense` 或 `orbbec`
- `--serial`：指定相机序列号
- `--model`：指定 YOLO 分割模型
- `--device`：推理设备，当前建议 `cpu`
- `--width` / `--height` / `--fps`：传给当前后端的共同目标流配置
- `--imgsz`：YOLO 推理尺寸
- `--min-depth` / `--max-depth`：有效深度范围
- `--target-class`：仅保留一个或多个指定类别；逗号顺序会保留，例如 `mouse,banana`
- `--target-table`：为 `--target-class` 中的目标实时打印固定顺序终端表格
- `--show-labels`：开启 3D 标注
- `--show-object-points`：高亮目标点云
- `--no-display`：关闭可视化，输出每帧 JSON
- `--frames`：运行固定帧数后退出

## 当前实现说明

- 当前相机层已经改成统一硬件抽象，可继续接入新的相机后端
- 桌面法向在启动时估计一次
- 3D 框为“桌面对齐 OBB”
- 输出旋转只有一个自由度 `yaw`
- `yaw` 是相对桌面平面基底定义的角度，适合抓取和姿态筛选
