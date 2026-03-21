# TabletopSeg3D

Realtime desktop-object 3D detection based on `Intel RealSense + YOLO Segmentation + Open3D`.

![TabletopSeg3D demo](3DDetection/photo/1.png)

## Project Location

The current project files are stored in:

[`3DDetection/`](./3DDetection)

Main documents:

- English: [`3DDetection/README.md`](./3DDetection/README.md)
- 中文: [`3DDetection/README_cn.md`](./3DDetection/README_cn.md)

Main entry:

- [`3DDetection/scripts/realtime_open3d_scene.py`](./3DDetection/scripts/realtime_open3d_scene.py)

## Features

- realtime Open3D scene visualization
- headless JSON output
- YOLO instance segmentation
- tabletop-aligned 3D OBB with 1-DoF yaw
- RealSense `D435I` / `D405` support

## Quick Start

```bash
cd 3DDetection
python -m pip install -r requirements.txt
python scripts/realtime_open3d_scene.py --list-devices
python scripts/realtime_open3d_scene.py --serial 419522072950 --device cpu
```
