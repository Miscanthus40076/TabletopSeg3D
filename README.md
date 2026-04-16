# TabletopSeg3D

Realtime desktop-object 3D detection based on camera backends + `YOLO Segmentation + Open3D`.

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
- fixed-order terminal table for selected target classes
- pluggable camera backends
- RealSense `D435I` / `D405` support
- Orbbec `Gemini2` support with default color `640x480` and depth `640x400`

## Quick Start

```bash
cd 3DDetection
python -m pip install -r requirements.txt
python scripts/realtime_open3d_scene.py --list-devices --camera-backend auto
python scripts/realtime_open3d_scene.py --camera-backend realsense --serial 419522072950 --device cpu
```

Orbbec backend install:

```bash
python -m pip install -r requirements-orbbec.txt
```

Current branch status:

- RealSense path is preserved through the new backend abstraction
- Orbbec Gemini2 is wired into the same runtime path
- merge to `main` should wait for a RealSense hardware regression pass

## Fixed-Order Target Table

Use `--target-class` to restrict detection to one or more classes. Use commas to keep a stable table order, then add `--target-table` to print a live terminal table.

```bash
cd 3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial AY3Z331006L \
  --device cpu \
  --target-class mouse,banana \
  --target-table \
  --no-display
```

In this example, row 1 is always `mouse` and row 2 is always `banana`. If a target is not detected, its row is printed as `null`.
