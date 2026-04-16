#!/usr/bin/env python3
"""Realtime Open3D scene viewer with full-scene point cloud and 3D boxes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from camera import (  # noqa: E402
    StreamRequest,
    enumerate_devices,
    open_runtime,
    read_frame_bundle,
    select_device,
    stop_runtimes,
)
from geometry.pointcloud import (  # noqa: E402
    filter_points_by_depth_band,
    project_mask_to_points,
    tabletop_aligned_obb,
)


@dataclass
class Detection3D:
    class_name: str
    confidence: float
    bbox_xyxy: list[int]
    mask: np.ndarray
    center_xyz: list[float] | None
    extent_xyz: list[float] | None
    yaw_rad: float | None
    yaw_deg: float | None
    rotation_matrix: np.ndarray | None
    box_corners_xyz: np.ndarray | None
    bbox_min_xyz: list[float] | None
    bbox_max_xyz: list[float] | None
    point_count: int


BOX_LINES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

BACKGROUND_COLOR_RGB = np.array([0.12, 0.14, 0.16], dtype=np.float64)
BACKGROUND_COLOR_RGBA = np.array([0.12, 0.14, 0.16, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Open3D point-cloud scene viewer.")
    parser.add_argument("--camera-backend", type=str, default="auto", choices=["auto", "realsense", "orbbec"], help="Camera backend to use.")
    parser.add_argument("--list-devices", action="store_true", help="List connected camera devices and exit.")
    parser.add_argument("--serial", type=str, default="", help="Camera serial number to use.")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Ultralytics segmentation model.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    parser.add_argument("--width", type=int, default=640, help="Camera stream width.")
    parser.add_argument("--height", type=int, default=480, help="Camera stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera stream FPS.")
    parser.add_argument("--imgsz", type=int, default=448, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections per frame.")
    parser.add_argument("--target-class", type=str, default="", help="Optional class filter. Use commas for a fixed ordered list, e.g. mouse,banana.")
    parser.add_argument("--target-table", action="store_true", help="Print a live fixed-order table for --target-class objects.")
    parser.add_argument("--min-depth", type=float, default=0.10, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.50, help="Maximum valid depth in meters.")
    parser.add_argument("--min-points", type=int, default=500, help="Minimum object point count for a 3D box.")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Camera warm-up frames.")
    parser.add_argument("--frames", type=int, default=0, help="Run a fixed number of frames then exit.")
    parser.add_argument("--point-stride", type=int, default=2, help="Subsample stride for full-scene point cloud.")
    parser.add_argument("--scene-max-points", type=int, default=80000, help="Maximum full-scene points to keep after stride.")
    parser.add_argument("--show-object-points", action="store_true", help="Also color object mask points in the full-scene point cloud.")
    parser.add_argument("--show-labels", action="store_true", help="Show 3D labels for detected objects in the Open3D scene.")
    parser.add_argument("--no-display", action="store_true", help="Disable Open3D window and print timing only.")
    args = parser.parse_args()
    if args.target_table and not args.target_class.strip():
        parser.error("--target-table requires --target-class.")
    args.target_classes = parse_target_classes(args.target_class)
    return args


def load_model(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def parse_target_classes(target_class: str) -> list[str]:
    return [name.strip() for name in target_class.split(",") if name.strip()]


def color_for_index(index: int) -> np.ndarray:
    palette = np.array(
        [
            [1.0, 0.35, 0.35],
            [0.35, 1.0, 0.55],
            [0.35, 0.7, 1.0],
            [1.0, 0.82, 0.35],
            [0.82, 0.35, 1.0],
            [0.35, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return palette[index % len(palette)]


def run_inference(model, color_image: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    results = model.predict(
        source=color_image,
        task="segment",
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )
    result = results[0]
    if result.masks is None or result.boxes is None:
        return []

    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy().astype(float)
    bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
    image_h, image_w = color_image.shape[:2]

    detections: list[dict[str, Any]] = []
    for idx, mask_arr in enumerate(masks):
        class_id = int(class_ids[idx])
        class_name = result.names.get(class_id, str(class_id))
        if args.target_classes and class_name not in args.target_classes:
            continue
        mask_resized = cv2.resize(mask_arr, (image_w, image_h), interpolation=cv2.INTER_NEAREST) > 0.5
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(confidences[idx]),
                "bbox_xyxy": [int(v) for v in bboxes[idx].tolist()],
                "mask": mask_resized,
            }
        )
    return detections


def build_detection_3d(
    detection: dict[str, Any],
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    table_normal: np.ndarray,
    args: argparse.Namespace,
) -> Detection3D:
    raw_points, _ = project_mask_to_points(
        mask=detection["mask"],
        depth_m=depth_m,
        intrinsics=intrinsics,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
    )
    filtered_points = filter_points_by_depth_band(raw_points)
    point_count = int(len(filtered_points))
    if point_count < args.min_points:
        return Detection3D(
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            bbox_xyxy=detection["bbox_xyxy"],
            mask=detection["mask"],
            center_xyz=None,
            extent_xyz=None,
            yaw_rad=None,
            yaw_deg=None,
            rotation_matrix=None,
            box_corners_xyz=None,
            bbox_min_xyz=None,
            bbox_max_xyz=None,
            point_count=point_count,
        )

    obb = tabletop_aligned_obb(filtered_points, plane_normal=table_normal)
    return Detection3D(
        class_name=detection["class_name"],
        confidence=detection["confidence"],
        bbox_xyxy=detection["bbox_xyxy"],
        mask=detection["mask"],
        center_xyz=obb["center_xyz"].tolist(),
        extent_xyz=obb["extent_xyz"].tolist(),
        yaw_rad=obb["yaw_rad"],
        yaw_deg=obb["yaw_deg"],
        rotation_matrix=obb["rotation_matrix"],
        box_corners_xyz=obb["corners_xyz"],
        bbox_min_xyz=obb["bbox_min_xyz"].tolist(),
        bbox_max_xyz=obb["bbox_max_xyz"].tolist(),
        point_count=point_count,
    )


def build_scene_point_cloud(
    color_image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    stride = max(1, int(args.point_stride))
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth_m[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > args.min_depth) & (sampled_depth < args.max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float64)

    z = sampled_depth[valid].astype(np.float32)
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = color_image[grid_y[valid], grid_x[valid]][:, ::-1].astype(np.float64) / 255.0

    max_points = int(args.scene_max_points)
    if max_points > 0 and len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[keep]
        colors = colors[keep]

    return points, colors


def highlight_object_points(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    detections_3d: list[Detection3D],
) -> np.ndarray:
    if len(scene_points) == 0:
        return scene_colors

    colors = scene_colors.copy()
    for idx, detection in enumerate(detections_3d):
        if detection.center_xyz is None or detection.extent_xyz is None or detection.rotation_matrix is None:
            continue
        center = np.asarray(detection.center_xyz, dtype=np.float32)
        half_extent = 0.5 * np.asarray(detection.extent_xyz, dtype=np.float32)
        local = (scene_points.astype(np.float32) - center[None, :]) @ detection.rotation_matrix
        inside = np.all(np.abs(local) <= (half_extent[None, :] + 1e-4), axis=1)
        if np.any(inside):
            colors[inside] = 0.55 * colors[inside] + 0.45 * color_for_index(idx)
    return colors


def update_line_set(line_set: Any, corners_xyz: np.ndarray | None, color: np.ndarray, o3d: Any) -> None:
    if corners_xyz is None or len(corners_xyz) != 8:
        line_set.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return

    line_set.points = o3d.utility.Vector3dVector(np.asarray(corners_xyz, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(BOX_LINES)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color[None, :], (len(BOX_LINES), 1)))


def scene_center(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.array([0.0, 0.0, 0.5], dtype=np.float64)
    return points_xyz.mean(axis=0).astype(np.float64)


def configure_view(vis: Any, center_xyz: np.ndarray) -> None:
    view = vis.get_view_control()
    view.set_lookat(center_xyz.tolist())
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.7)


def scene_extent(points_xyz: np.ndarray) -> float:
    if len(points_xyz) == 0:
        return 1.0
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return float(max(np.linalg.norm(maxs - mins), 0.5))


def scene_eye(points_xyz: np.ndarray, center_xyz: np.ndarray) -> np.ndarray:
    distance = scene_extent(points_xyz) * 1.2
    return center_xyz + np.array([0.0, 0.0, -distance], dtype=np.float32)


def label_anchor(detection: Detection3D) -> np.ndarray:
    if detection.box_corners_xyz is not None:
        corners = np.asarray(detection.box_corners_xyz, dtype=np.float32)
        return corners[corners[:, 1].argmin()]
    if detection.center_xyz is not None:
        return np.asarray(detection.center_xyz, dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def format_detection_label(detection: Detection3D) -> str:
    if detection.center_xyz is None:
        return (
            f"{detection.class_name} {detection.confidence:.2f}\n"
            f"pts={detection.point_count}"
        )

    cx, cy, cz = detection.center_xyz
    yaw_text = "n/a" if detection.yaw_deg is None else f"{detection.yaw_deg:.1f} deg"
    return (
        f"{detection.class_name} {detection.confidence:.2f}\n"
        f"xyz=({cx:.3f}, {cy:.3f}, {cz:.3f}) m\n"
        f"yaw={yaw_text}\n"
        f"pts={detection.point_count}"
    )


def build_legacy_point_cloud(o3d: Any, scene_points: np.ndarray, scene_colors: np.ndarray) -> Any:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors.astype(np.float64))
    return pcd


def update_labels(vis: Any, detections_3d: list[Detection3D]) -> None:
    vis.clear_3d_labels()
    for detection in detections_3d:
        if detection.center_xyz is None:
            continue
        vis.add_3d_label(label_anchor(detection), format_detection_label(detection))


def estimate_table_normal(scene_points: np.ndarray, o3d: Any) -> np.ndarray:
    default_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if len(scene_points) < 128:
        return default_normal

    sampled_points = scene_points
    max_plane_points = 12000
    if len(sampled_points) > max_plane_points:
        keep = np.linspace(0, len(sampled_points) - 1, max_plane_points, dtype=np.int32)
        sampled_points = sampled_points[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=120)
    normal = np.asarray(plane_model[:3], dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return default_normal
    normal = normal / norm
    if float(np.dot(normal, default_normal)) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def frame_output_record(
    frame_index: int,
    fps_value: float,
    infer_ms: float,
    geom_ms: float,
    scene_points: np.ndarray,
    table_normal: np.ndarray,
    detections_3d: list[Detection3D],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "fps": round(float(fps_value), 4),
        "infer_ms": round(float(infer_ms), 4),
        "geom_ms": round(float(geom_ms), 4),
        "scene_point_count": int(len(scene_points)),
        "table_normal_xyz": [round(float(v), 6) for v in table_normal.tolist()],
        "detections": [
            {
                "class_name": det.class_name,
                "confidence": round(float(det.confidence), 6),
                "center_camera_xyz_m": None if det.center_xyz is None else [round(float(v), 6) for v in det.center_xyz],
                "extent_xyz_m": None if det.extent_xyz is None else [round(float(v), 6) for v in det.extent_xyz],
                "yaw_rad": None if det.yaw_rad is None else round(float(det.yaw_rad), 6),
                "yaw_deg": None if det.yaw_deg is None else round(float(det.yaw_deg), 4),
                "point_count": int(det.point_count),
            }
            for det in detections_3d
            if det.center_xyz is not None
        ],
    }


def _format_nullable_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def _best_detection_for_class(
    detections_3d: list[Detection3D],
    class_name: str,
) -> Detection3D | None:
    matches = [det for det in detections_3d if det.class_name == class_name]
    if not matches:
        return None
    return max(matches, key=lambda det: (det.center_xyz is not None, det.confidence))


def target_table_record(
    target_classes: list[str],
    detections_3d: list[Detection3D],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for slot, class_name in enumerate(target_classes, start=1):
        det = _best_detection_for_class(detections_3d, class_name)
        if det is None or det.center_xyz is None or det.extent_xyz is None:
            rows.append(
                {
                    "slot": str(slot),
                    "class": class_name,
                    "status": "null",
                    "conf": "null",
                    "x_m": "null",
                    "y_m": "null",
                    "z_m": "null",
                    "size_m": "null",
                    "yaw_deg": "null",
                    "points": "null",
                }
            )
            continue

        rows.append(
            {
                "slot": str(slot),
                "class": class_name,
                "status": "detected",
                "conf": _format_nullable_float(det.confidence, 3),
                "x_m": _format_nullable_float(det.center_xyz[0], 3),
                "y_m": _format_nullable_float(det.center_xyz[1], 3),
                "z_m": _format_nullable_float(det.center_xyz[2], 3),
                "size_m": "x".join(_format_nullable_float(value, 3) for value in det.extent_xyz),
                "yaw_deg": _format_nullable_float(det.yaw_deg, 1),
                "points": str(int(det.point_count)),
            }
        )
    return rows


def render_target_table(
    frame_index: int,
    fps_value: float,
    infer_ms: float,
    geom_ms: float,
    target_classes: list[str],
    detections_3d: list[Detection3D],
) -> str:
    rows = target_table_record(target_classes, detections_3d)
    headers = ["slot", "class", "status", "conf", "x_m", "y_m", "z_m", "size_m", "yaw_deg", "points"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    line = "+-" + "-+-".join("-" * widths[header] for header in headers) + "-+"
    header_line = "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |"
    body_lines = [
        "| " + " | ".join(row[header].ljust(widths[header]) for header in headers) + " |"
        for row in rows
    ]
    title = (
        f"Target table | frame={frame_index} | fps={fps_value:.2f} | "
        f"infer={infer_ms:.1f}ms | geom={geom_ms:.1f}ms"
    )
    return "\n".join([title, line, header_line, line, *body_lines, line])


def print_live_target_table(table_text: str) -> None:
    print("\033[2J\033[H" + table_text, flush=True)


def print_connected_devices(devices: list[Any]) -> None:
    if not devices:
        print("No camera devices found.")
        return

    print("Connected camera devices:")
    for device in devices:
        print(
            f"- backend={device.backend} | {device.name} | serial={device.serial_number}"
        )


def main() -> int:
    args = parse_args()

    devices = enumerate_devices(args.camera_backend)
    if args.list_devices:
        print_connected_devices(devices)
        return 0
    selected_device = select_device(devices, serial=args.serial, backend_name=args.camera_backend)

    import open3d as o3d

    model = load_model(args.model)
    runtime = open_runtime(
        selected_device,
        StreamRequest(width=args.width, height=args.height, fps=args.fps, align_to_color=True),
    )
    frame_times: list[float] = []
    frame_counter = 0

    vis = None
    gui_app = None
    scene_pcd = o3d.geometry.PointCloud()
    scene_material = None
    box_sets: list[Any] = [o3d.geometry.LineSet() for _ in range(args.max_det)]
    box_added = [False for _ in range(args.max_det)]
    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    label_mode = bool(args.show_labels and not args.no_display)
    window_closed = False
    table_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    try:
        for _ in range(args.warmup_frames):
            read_frame_bundle(runtime)

        warm_bundle = read_frame_bundle(runtime)
        model.predict(
            source=warm_bundle.color,
            task="segment",
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
        )

        initial_scene_points, initial_scene_colors = build_scene_point_cloud(
            color_image=warm_bundle.color,
            depth_m=warm_bundle.depth.astype(np.float32) * float(warm_bundle.depth_scale),
            intrinsics=warm_bundle.color_intrinsics,
            args=args,
        )
        table_normal = estimate_table_normal(initial_scene_points, o3d)
        initial_depth_m = warm_bundle.depth.astype(np.float32) * float(warm_bundle.depth_scale)
        initial_detections = run_inference(model, warm_bundle.color, args)
        initial_detections_3d = [
            build_detection_3d(det, initial_depth_m, warm_bundle.color_intrinsics, table_normal, args)
            for det in initial_detections
        ]
        if args.show_object_points:
            initial_scene_colors = highlight_object_points(initial_scene_points, initial_scene_colors, initial_detections_3d)

        if not args.no_display:
            if label_mode:
                import open3d.visualization.gui as gui
                import open3d.visualization.rendering as rendering

                gui_app = gui.Application.instance
                gui_app.initialize()
                vis = o3d.visualization.O3DVisualizer("Realtime Open3D Scene", 1280, 800)
                vis.show_axes = True
                vis.show_settings = False
                vis.show_ground = False
                vis.show_skybox(False)
                vis.set_background(BACKGROUND_COLOR_RGBA, None)

                def on_window_close() -> bool:
                    nonlocal window_closed
                    window_closed = True
                    return True

                vis.set_on_close(on_window_close)

                scene_material = rendering.MaterialRecord()
                scene_material.shader = "defaultUnlit"
                scene_material.point_size = 2.0

                vis.add_geometry(
                    "scene",
                    build_legacy_point_cloud(o3d, initial_scene_points, initial_scene_colors),
                    scene_material,
                )

                for idx, detection in enumerate(initial_detections_3d):
                    if idx >= args.max_det or detection.box_corners_xyz is None:
                        continue
                    update_line_set(box_sets[idx], detection.box_corners_xyz, color_for_index(idx), o3d)
                    vis.add_geometry(f"box_{idx}", box_sets[idx])
                    box_added[idx] = True

                update_labels(vis, initial_detections_3d)
                gui_app.add_window(vis)
                vis.setup_camera(
                    45.0,
                    scene_center(initial_scene_points).astype(np.float32),
                    scene_eye(initial_scene_points, scene_center(initial_scene_points)).astype(np.float32),
                    np.array([0.0, -1.0, 0.0], dtype=np.float32),
                )
                gui_app.run_one_tick()
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Realtime Open3D Scene", width=1280, height=800)
                scene_pcd.points = o3d.utility.Vector3dVector(initial_scene_points.astype(np.float64))
                scene_pcd.colors = o3d.utility.Vector3dVector(initial_scene_colors.astype(np.float64))
                vis.add_geometry(scene_pcd)

                first_center = None
                for idx, detection in enumerate(initial_detections_3d):
                    if idx >= len(box_sets):
                        break
                    if detection.box_corners_xyz is None:
                        continue
                    update_line_set(box_sets[idx], detection.box_corners_xyz, color_for_index(idx), o3d)
                    vis.add_geometry(box_sets[idx], reset_bounding_box=False)
                    box_added[idx] = True
                    if first_center is None and detection.center_xyz is not None:
                        first_center = np.array(detection.center_xyz, dtype=np.float64)

                if first_center is not None:
                    center_frame.translate(first_center, relative=True)
                vis.add_geometry(center_frame)
                render_option = vis.get_render_option()
                render_option.background_color = BACKGROUND_COLOR_RGB
                render_option.point_size = 2.0
                configure_view(vis, scene_center(initial_scene_points))

        while True:
            loop_start = time.perf_counter()
            bundle = read_frame_bundle(runtime)
            depth_m = bundle.depth.astype(np.float32) * float(bundle.depth_scale)

            infer_start = time.perf_counter()
            detections = run_inference(model, bundle.color, args)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            geom_start = time.perf_counter()
            detections_3d = [
                build_detection_3d(det, depth_m, bundle.color_intrinsics, table_normal, args)
                for det in detections
            ]
            scene_points, scene_colors = build_scene_point_cloud(
                color_image=bundle.color,
                depth_m=depth_m,
                intrinsics=bundle.color_intrinsics,
                args=args,
            )
            if args.show_object_points:
                scene_colors = highlight_object_points(scene_points, scene_colors, detections_3d)
            geom_ms = (time.perf_counter() - geom_start) * 1000.0

            loop_time = time.perf_counter() - loop_start
            frame_times.append(loop_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_value = len(frame_times) / sum(frame_times)

            if args.target_table:
                print_live_target_table(
                    render_target_table(
                        frame_index=frame_counter,
                        fps_value=fps_value,
                        infer_ms=infer_ms,
                        geom_ms=geom_ms,
                        target_classes=args.target_classes,
                        detections_3d=detections_3d,
                    )
                )

            if args.no_display:
                if not args.target_table:
                    print(
                        json.dumps(
                            frame_output_record(
                                frame_index=frame_counter,
                                fps_value=fps_value,
                                infer_ms=infer_ms,
                                geom_ms=geom_ms,
                                scene_points=scene_points,
                                table_normal=table_normal,
                                detections_3d=detections_3d,
                            ),
                            ensure_ascii=False,
                        )
                    )
            else:
                if label_mode:
                    vis.remove_geometry("scene")
                    vis.add_geometry(
                        "scene",
                        build_legacy_point_cloud(o3d, scene_points, scene_colors),
                        scene_material,
                    )

                    for idx in range(args.max_det):
                        box_name = f"box_{idx}"
                        if idx < len(detections_3d) and detections_3d[idx].box_corners_xyz is not None:
                            update_line_set(
                                box_sets[idx],
                                detections_3d[idx].box_corners_xyz,
                                color_for_index(idx),
                                o3d,
                            )
                            if box_added[idx]:
                                vis.remove_geometry(box_name)
                            vis.add_geometry(box_name, box_sets[idx])
                            box_added[idx] = True
                        elif box_added[idx]:
                            vis.remove_geometry(box_name)
                            box_added[idx] = False

                    update_labels(vis, detections_3d)
                    vis.post_redraw()
                    if window_closed or not gui_app.run_one_tick():
                        break
                else:
                    scene_pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
                    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors.astype(np.float64))
                    vis.update_geometry(scene_pcd)

                    first_center = None
                    for idx, box_set in enumerate(box_sets):
                        if idx < len(detections_3d):
                            detection = detections_3d[idx]
                            if detection.box_corners_xyz is not None:
                                update_line_set(
                                    box_set,
                                    detection.box_corners_xyz,
                                    color_for_index(idx),
                                    o3d,
                                )
                                if not box_added[idx]:
                                    vis.add_geometry(box_set, reset_bounding_box=False)
                                    box_added[idx] = True
                            elif box_added[idx]:
                                vis.remove_geometry(box_set, reset_bounding_box=False)
                                box_added[idx] = False
                            if first_center is None and detection.center_xyz is not None:
                                first_center = np.array(detection.center_xyz, dtype=np.float64)
                        else:
                            if box_added[idx]:
                                vis.remove_geometry(box_set, reset_bounding_box=False)
                                box_added[idx] = False

                        if box_added[idx]:
                            vis.update_geometry(box_set)

                    if first_center is None:
                        center_frame.translate(-np.asarray(center_frame.get_center()), relative=True)
                    else:
                        center_frame.translate(first_center - np.asarray(center_frame.get_center()), relative=True)
                    vis.update_geometry(center_frame)

                    if not vis.poll_events():
                        break
                    vis.update_renderer()

            frame_counter += 1
            if args.frames > 0 and frame_counter >= args.frames:
                break

    finally:
        if vis is not None:
            if hasattr(vis, "destroy_window"):
                vis.destroy_window()
            elif hasattr(vis, "close"):
                vis.close()
        if gui_app is not None and hasattr(gui_app, "quit"):
            gui_app.quit()
        stop_runtimes([runtime])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
