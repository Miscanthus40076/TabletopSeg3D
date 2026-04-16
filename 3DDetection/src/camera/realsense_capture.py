"""Backward-compatible RealSense helpers re-exported from the new backend module."""

from .factory import stop_runtimes
from .realsense_backend import BACKEND, intrinsics_to_dict, safe_get_info
from .types import StreamRequest

enumerate_devices = BACKEND.enumerate_devices


def select_serials(devices, serials, expected_count: int = 2):
    available = {device.serial_number for device in devices}
    if serials:
        missing = [serial for serial in serials if serial not in available]
        if missing:
            raise RuntimeError(f"Requested serials not found: {missing}")
        return serials

    if len(devices) < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} RealSense devices, found {len(devices)}. "
            "Use --list-devices to inspect detection state."
        )
    return [device.serial_number for device in devices[:expected_count]]


def build_runtime(
    device_info,
    width: int,
    height: int,
    fps: int,
    enable_depth: bool = True,
    enable_color: bool = True,
):
    if not enable_depth or not enable_color:
        raise RuntimeError("The compatibility wrapper only supports color+depth streams.")
    return BACKEND.open_runtime(
        device_info,
        StreamRequest(width=width, height=height, fps=fps, align_to_color=True),
    )


def get_aligned_frame_bundle(runtime, depth_min_m: float, depth_max_m: float):
    del depth_min_m, depth_max_m
    bundle = BACKEND.read_frame_bundle(runtime)
    return {
        "serial_number": bundle.serial_number,
        "device_name": bundle.device_name,
        "firmware_version": bundle.firmware_version,
        "product_line": bundle.product_line,
        "usb_type_descriptor": bundle.usb_type_descriptor,
        "color": bundle.color,
        "depth": bundle.depth,
        "timestamp_ms": bundle.timestamp_ms,
        "frame_number": bundle.frame_number,
        "depth_scale": bundle.depth_scale,
        "color_intrinsics": bundle.color_intrinsics,
        "depth_intrinsics": bundle.depth_intrinsics,
        "resolved_stream": bundle.resolved_stream,
    }
