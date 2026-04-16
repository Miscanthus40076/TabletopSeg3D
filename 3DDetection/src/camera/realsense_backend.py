from __future__ import annotations

from typing import Any

import numpy as np

from .types import CameraRuntime, DeviceInfo, FrameBundle, ResolvedStreamConfig, StreamRequest

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local SDK install
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_sdk():
    if rs is None:
        raise RuntimeError(f"pyrealsense2 is not installed: {_IMPORT_ERROR}")


def safe_get_info(device: Any, info_key: Any) -> str:
    try:
        return device.get_info(info_key)
    except RuntimeError:
        return ""


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def _resolved_stream_for_request(request: StreamRequest) -> ResolvedStreamConfig:
    size = (int(request.width), int(request.height))
    return ResolvedStreamConfig(
        requested_width=int(request.width),
        requested_height=int(request.height),
        requested_fps=int(request.fps),
        actual_color_size=size,
        actual_depth_size=size,
        actual_fps=int(request.fps),
    )


class RealSenseBackend:
    backend_name = "realsense"

    def is_available(self) -> bool:
        return rs is not None

    def unavailable_reason(self) -> str | None:
        if rs is None:
            return str(_IMPORT_ERROR)
        return None

    def enumerate_devices(self) -> list[DeviceInfo]:
        _require_sdk()
        ctx = rs.context()
        devices: list[DeviceInfo] = []
        for dev in ctx.query_devices():
            devices.append(
                DeviceInfo(
                    backend=self.backend_name,
                    name=safe_get_info(dev, rs.camera_info.name),
                    serial_number=safe_get_info(dev, rs.camera_info.serial_number),
                    firmware_version=safe_get_info(dev, rs.camera_info.firmware_version),
                    vendor="Intel",
                    usb_type_descriptor=safe_get_info(dev, rs.camera_info.usb_type_descriptor),
                    product_line=safe_get_info(dev, rs.camera_info.product_line),
                )
            )
        return devices

    def open_runtime(self, device_info: DeviceInfo, stream_request: StreamRequest) -> CameraRuntime:
        _require_sdk()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_info.serial_number)
        config.enable_stream(
            rs.stream.depth,
            int(stream_request.width),
            int(stream_request.height),
            rs.format.z16,
            int(stream_request.fps),
        )
        config.enable_stream(
            rs.stream.color,
            int(stream_request.width),
            int(stream_request.height),
            rs.format.bgr8,
            int(stream_request.fps),
        )
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        return CameraRuntime(
            backend_name=self.backend_name,
            info=device_info,
            pipeline=pipeline,
            aligner=rs.align(rs.stream.color) if stream_request.align_to_color else None,
            depth_scale=float(depth_sensor.get_depth_scale()),
            resolved_stream=_resolved_stream_for_request(stream_request),
            state={"profile": profile},
        )

    def read_frame_bundle(self, runtime: CameraRuntime) -> FrameBundle:
        _require_sdk()
        frames = runtime.pipeline.wait_for_frames()
        if runtime.aligner is not None:
            frames = runtime.aligner.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"Missing color or depth frame from RealSense pipeline {runtime.info.serial_number}.")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        actual_color_size = (int(color.shape[1]), int(color.shape[0]))
        actual_depth_size = (int(depth.shape[1]), int(depth.shape[0]))
        resolved_stream = runtime.resolved_stream.updated(
            actual_color_size=actual_color_size,
            actual_depth_size=actual_depth_size,
        )

        return FrameBundle(
            backend=self.backend_name,
            serial_number=runtime.info.serial_number,
            device_name=runtime.info.name,
            firmware_version=runtime.info.firmware_version,
            product_line=runtime.info.product_line,
            usb_type_descriptor=runtime.info.usb_type_descriptor,
            color=color,
            depth=depth,
            timestamp_ms=float(frames.get_timestamp()),
            frame_number=int(frames.get_frame_number()),
            depth_scale=float(runtime.depth_scale),
            color_intrinsics=intrinsics_to_dict(color_intrinsics),
            depth_intrinsics=intrinsics_to_dict(depth_intrinsics),
            resolved_stream=resolved_stream,
        )

    def close_runtime(self, runtime: CameraRuntime) -> None:
        try:
            runtime.pipeline.stop()
        except RuntimeError:
            pass


BACKEND = RealSenseBackend()
