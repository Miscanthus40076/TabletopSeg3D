# TabletopSeg3D

Chinese documentation: [README_cn.md](./README_cn.md)

Realtime desktop-object 3D detection based on pluggable camera backends + `YOLO Segmentation + Open3D`.

![TabletopSeg3D demo](photo/1.png)

This repository is intentionally trimmed down to the final runtime features:

- realtime Open3D scene visualization
- headless JSON output

## Features

- single-camera realtime pipeline with pluggable camera backends
- tested on RealSense `D435I` / `D405`
- Orbbec `Gemini2` backend with default color `640x480` and depth `640x400`
- YOLO instance segmentation
- depth-to-point-cloud projection
- tabletop-aligned OBB
  - box `Z` follows the tabletop normal
  - only one rotational degree of freedom is reported: `yaw`
- optional 3D labels in Open3D
- `--no-display` prints per-frame JSON including:
  - object class
  - center in camera coordinates
  - box size
  - box yaw

## Layout

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

## Environment

Python `3.11` is recommended.

Install dependencies:

```bash
cd /home/misca/3DDetection
python -m pip install -r requirements.txt
```

Optional hardware backends:

- RealSense: install Intel RealSense SDK so `pyrealsense2` can access the camera runtime
- Orbbec: install with `python -m pip install -r requirements-orbbec.txt`

Note:

- the Orbbec package is installed as `pyorbbecsdk2`
- the runtime import used by the code is `import pyorbbecsdk`
- `requirements-orbbec.txt` keeps Orbbec optional instead of forcing every install to include the Orbbec SDK
- this branch should stay out of `main` until RealSense hardware regression is completed

If GPU acceleration is desired, you also need:

- an NVIDIA driver
- a compatible CUDA runtime
- GPU builds of `torch`
- GPU builds of `torchvision`

Note:

- the current `requirements.txt` is CPU-oriented
- the key step for YOLO GPU inference is replacing CPU `torch` / `torchvision` with versions that match your local CUDA stack

## List Connected Cameras

To print the model name and serial number of connected devices:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py --list-devices --camera-backend auto
```

Example output:

```text
Connected camera devices:
- backend=realsense | Intel RealSense D405 | serial=409122273421
- backend=realsense | Intel RealSense D435I | serial=419522072950
- backend=orbbec | Orbbec Gemini2 | serial=AYXXXXXXXXXX
```

## Model

The default model in the repository root is:

```bash
yolo11n-seg.pt
```

You can replace it with your own YOLO segmentation model through `--model`.

Accepted model forms:

- an official model name such as `yolo11n-seg.pt`
- a local weight file such as `./my_model.pt`
- a training output such as `runs/segment/train/weights/best.pt`
- any absolute path such as `/home/yourname/models/best.pt`

Recommended usage with your own model:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --model runs/segment/train/weights/best.pt \
  --device cpu
```

Headless output with your own model:

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

Fixed-order target table:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial AY3Z331006L \
  --device cpu \
  --target-class mouse,banana \
  --target-table
```

The table keeps the target order stable. In the example above, row 1 is always `mouse` and row 2 is always `banana`; missing targets are printed as `null`.

For terminal-only output, add `--no-display`:

```bash
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial AY3Z331006L \
  --device cpu \
  --target-class mouse,banana \
  --target-table \
  --no-display
```

Important:

- you must use a `segmentation` model
- a plain detection model has no masks and cannot produce the 3D box pipeline used here

## Realtime Visualization

Default example with `D435I`:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu
```

With 3D labels:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu \
  --show-labels
```

Default example with `Orbbec Gemini2`:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend orbbec \
  --serial <gemini2-serial> \
  --device cpu
```

Gemini2 default stream behavior:

- default target request is `640x480@30`
- actual default color stream is `640x480`
- actual default depth stream is `640x400`
- the resolved sizes are tracked inside the backend frame bundle

## Headless Output

Run without GUI and print one JSON record per frame:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 419522072950 \
  --device cpu \
  --frames 10 \
  --no-display
```

Example JSON:

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

## D405 Example

`D405` is more suitable for close range. A tighter depth range is recommended:

```bash
cd /home/misca/3DDetection
python scripts/realtime_open3d_scene.py \
  --camera-backend realsense \
  --serial 409122273421 \
  --device cpu \
  --min-depth 0.02 \
  --max-depth 0.50
```

Headless:

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

## Main Arguments

- `--camera-backend`: select `auto`, `realsense`, or `orbbec`
- `--list-devices`: print connected devices and exit
- `--serial`: select the camera serial number
- `--model`: select the YOLO segmentation model
- `--device`: inference device, currently `cpu` is recommended
- `--width` / `--height` / `--fps`: shared stream target request passed to the selected backend
- `--imgsz`: YOLO input size
- `--min-depth` / `--max-depth`: valid depth range
- `--target-class`: keep one or more target classes; comma order is preserved, e.g. `mouse,banana`
- `--target-table`: print a live fixed-order terminal table for `--target-class` objects
- `--show-labels`: enable 3D labels
- `--show-object-points`: highlight object points in the scene point cloud
- `--no-display`: disable visualization and print per-frame JSON
- `--frames`: stop after a fixed number of frames

## Notes

- the tabletop normal is estimated once at startup
- the camera layer now runs through a backend abstraction shared by RealSense and Orbbec
- the 3D box is a tabletop-aligned OBB
- only one rotational degree of freedom is reported: `yaw`
- `yaw` is defined in the tabletop plane basis and is suitable for grasp filtering and pose screening
