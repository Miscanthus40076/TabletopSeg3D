from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    serial_number: str
    firmware_version: str
    usb_type_descriptor: str
    product_line: str


@dataclass
class CameraRuntime:
    info: DeviceInfo
    pipeline: rs.pipeline
    align: rs.align
    depth_scale: float


def safe_get_info(device: rs.device, info_key: rs.camera_info) -> str:
    try:
        return device.get_info(info_key)
    except RuntimeError:
        return ""


def enumerate_devices() -> list[DeviceInfo]:
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append(
            DeviceInfo(
                name=safe_get_info(dev, rs.camera_info.name),
                serial_number=safe_get_info(dev, rs.camera_info.serial_number),
                firmware_version=safe_get_info(dev, rs.camera_info.firmware_version),
                usb_type_descriptor=safe_get_info(dev, rs.camera_info.usb_type_descriptor),
                product_line=safe_get_info(dev, rs.camera_info.product_line),
            )
        )
    return devices


def select_serials(devices: list[DeviceInfo], serials: list[str] | None, expected_count: int = 2) -> list[str]:
    available = {dev.serial_number for dev in devices}
    if serials:
        missing = [serial for serial in serials if serial not in available]
        if missing:
            raise RuntimeError(f"Requested serials not found: {missing}")
        return serials

    if len(devices) < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} RealSense devices, found {len(devices)}. "
            "Use --list to inspect detection state."
        )

    return [device.serial_number for device in devices[:expected_count]]


def intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def build_runtime(
    device_info: DeviceInfo,
    width: int,
    height: int,
    fps: int,
    enable_depth: bool = True,
    enable_color: bool = True,
) -> CameraRuntime:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_info.serial_number)
    if enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    if enable_color:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

    return CameraRuntime(
        info=device_info,
        pipeline=pipeline,
        align=rs.align(rs.stream.color),
        depth_scale=depth_sensor.get_depth_scale(),
    )


def get_aligned_frame_bundle(
    runtime: CameraRuntime,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any]:
    frames = runtime.pipeline.wait_for_frames()
    aligned_frames = runtime.align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError(f"Missing color or depth frame from RealSense pipeline {runtime.info.serial_number}.")

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    return {
        "serial_number": runtime.info.serial_number,
        "device_name": runtime.info.name,
        "firmware_version": runtime.info.firmware_version,
        "product_line": runtime.info.product_line,
        "usb_type_descriptor": runtime.info.usb_type_descriptor,
        "color": color,
        "depth": depth,
        "timestamp_ms": frames.get_timestamp(),
        "frame_number": frames.get_frame_number(),
        "depth_scale": runtime.depth_scale,
        "color_intrinsics": intrinsics_to_dict(color_intrinsics),
        "depth_intrinsics": intrinsics_to_dict(depth_intrinsics),
    }


def stop_runtimes(runtimes: list[CameraRuntime]) -> None:
    for runtime in runtimes:
        try:
            runtime.pipeline.stop()
        except RuntimeError:
            pass
