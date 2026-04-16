from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import CameraRuntime, DeviceInfo, FrameBundle, ResolvedStreamConfig, StreamRequest

try:
    import pyorbbecsdk as ob
except ImportError as exc:  # pragma: no cover - depends on local SDK install
    ob = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_sdk():
    if ob is None:
        raise RuntimeError(f"pyorbbecsdk is not installed: {_IMPORT_ERROR}")


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _distortion_to_coeffs(distortion: Any) -> list[float]:
    if distortion is None:
        return []
    keys = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    return [float(getattr(distortion, key, 0.0)) for key in keys]


def intrinsics_to_dict(profile: Any) -> dict[str, Any]:
    video_profile = profile.as_video_stream_profile()
    intrinsic = video_profile.get_intrinsic()
    distortion = video_profile.get_distortion()
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "ppx": float(intrinsic.cx),
        "ppy": float(intrinsic.cy),
        "model": _enum_name(getattr(distortion, "model", None)),
        "coeffs": _distortion_to_coeffs(distortion),
    }


def _iter_video_profiles(profile_list: Any) -> list[Any]:
    profiles: list[Any] = []
    for index in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(index)
        if hasattr(profile, "as_video_stream_profile"):
            profiles.append(profile.as_video_stream_profile())
    return profiles


def _profile_score(profile: Any, width: int, height: int, fps: int, format_priority: dict[str, int]) -> tuple[int, int, int, int]:
    format_name = _enum_name(profile.get_format())
    return (
        format_priority.get(format_name, len(format_priority) + 1),
        abs(int(profile.get_width()) - int(width)),
        abs(int(profile.get_height()) - int(height)),
        abs(int(profile.get_fps()) - int(fps)),
    )


def _select_best_profile(
    profile_list: Any,
    width: int,
    height: int,
    fps: int,
    preferred_formats: list[str] | None = None,
) -> Any:
    profiles = _iter_video_profiles(profile_list)
    if not profiles:
        raise RuntimeError("No video stream profiles are available.")
    format_priority = {name: index for index, name in enumerate(preferred_formats or [])}
    return min(profiles, key=lambda profile: _profile_score(profile, width, height, fps, format_priority))


def _select_depth_profile_for_color(
    pipeline: Any,
    color_profile: Any,
    width: int,
    height: int,
    fps: int,
) -> tuple[Any | None, str]:
    try:
        profile_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
    except Exception:
        return None, "sw"

    profiles = []
    try:
        for index in range(profile_list.get_count()):
            profiles.append(profile_list.get_stream_profile_by_index(index).as_video_stream_profile())
    except Exception:
        try:
            for index in range(len(profile_list)):
                profiles.append(profile_list[index].as_video_stream_profile())
        except Exception:
            profiles = []

    if not profiles:
        return None, "sw"

    selected = min(
        profiles,
        key=lambda profile: (
            abs(int(profile.get_width()) - int(width)),
            abs(int(profile.get_height()) - int(height)),
            abs(int(profile.get_fps()) - int(fps)),
        ),
    )
    return selected, "hw"


def _reshape_color_frame(frame: Any) -> np.ndarray | None:
    width = int(frame.get_width())
    height = int(frame.get_height())
    frame_format = _enum_name(frame.get_format())
    data = np.asanyarray(frame.get_data())

    if frame_format == "RGB":
        return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
    if frame_format == "BGR":
        return np.resize(data, (height, width, 3)).copy()
    if frame_format in {"YUYV", "YUY2"}:
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_YUY2)
    if frame_format == "UYVY":
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    if frame_format == "MJPG":
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_format == "NV12":
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)
    if frame_format == "NV21":
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV21)
    if frame_format == "I420":
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_I420)
    return None


def _resolved_stream(request: StreamRequest, color_profile: Any, depth_profile: Any) -> ResolvedStreamConfig:
    return ResolvedStreamConfig(
        requested_width=int(request.width),
        requested_height=int(request.height),
        requested_fps=int(request.fps),
        actual_color_size=(int(color_profile.get_width()), int(color_profile.get_height())),
        actual_depth_size=(int(depth_profile.get_width()), int(depth_profile.get_height())),
        actual_fps=int(color_profile.get_fps()),
    )


class OrbbecBackend:
    backend_name = "orbbec"
    _gemini2_default_color = (640, 480)
    _gemini2_default_depth = (640, 400)
    _preferred_color_formats = ["BGR", "RGB", "MJPG", "YUYV", "YUY2", "UYVY", "NV12", "NV21", "I420"]

    def is_available(self) -> bool:
        return ob is not None

    def unavailable_reason(self) -> str | None:
        if ob is None:
            return str(_IMPORT_ERROR)
        return None

    def enumerate_devices(self) -> list[DeviceInfo]:
        _require_sdk()
        ctx = ob.Context()
        device_list = ctx.query_devices()
        devices: list[DeviceInfo] = []
        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            devices.append(
                DeviceInfo(
                    backend=self.backend_name,
                    name=info.get_name(),
                    serial_number=info.get_serial_number(),
                    firmware_version=info.get_firmware_version(),
                    vendor="Orbbec",
                    product_line=_enum_name(info.get_device_type()),
                    usb_type_descriptor=info.get_connection_type(),
                )
            )
        return devices

    def _resolve_target_sizes(self, request: StreamRequest) -> tuple[tuple[int, int], tuple[int, int]]:
        requested = (int(request.width), int(request.height))
        if requested == self._gemini2_default_color:
            return self._gemini2_default_color, self._gemini2_default_depth
        return requested, requested

    def _get_device(self, serial_number: str) -> Any:
        ctx = ob.Context()
        device_list = ctx.query_devices()
        device = device_list.get_device_by_serial_number(serial_number)
        if device is None:
            raise RuntimeError(f"Requested Orbbec serial '{serial_number}' was not found.")
        return device

    def open_runtime(self, device_info: DeviceInfo, stream_request: StreamRequest) -> CameraRuntime:
        _require_sdk()
        device = self._get_device(device_info.serial_number)
        pipeline = ob.Pipeline(device)
        config = ob.Config()
        color_target, depth_target = self._resolve_target_sizes(stream_request)

        color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = _select_best_profile(
            color_profiles,
            width=color_target[0],
            height=color_target[1],
            fps=stream_request.fps,
            preferred_formats=self._preferred_color_formats,
        )

        depth_profile, align_mode = _select_depth_profile_for_color(
            pipeline,
            color_profile=color_profile,
            width=depth_target[0],
            height=depth_target[1],
            fps=stream_request.fps,
        )
        if depth_profile is None:
            depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = _select_best_profile(
                depth_profiles,
                width=depth_target[0],
                height=depth_target[1],
                fps=stream_request.fps,
            )

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

        aligner = None
        if stream_request.align_to_color:
            if align_mode == "hw":
                config.set_align_mode(ob.OBAlignMode.HW_MODE)
            else:
                aligner = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        try:
            pipeline.enable_frame_sync()
        except Exception:
            pass

        pipeline.start(config)
        resolved_stream = _resolved_stream(stream_request, color_profile, depth_profile)

        return CameraRuntime(
            backend_name=self.backend_name,
            info=device_info,
            pipeline=pipeline,
            aligner=aligner,
            depth_scale=1.0,
            resolved_stream=resolved_stream,
            state={
                "align_mode": align_mode,
                "device": device,
                "frame_timeout_ms": 100,
                "frame_retry_count": 30,
            },
        )

    def read_frame_bundle(self, runtime: CameraRuntime) -> FrameBundle:
        _require_sdk()
        timeout_ms = int(runtime.state.get("frame_timeout_ms", 100))
        retry_count = int(runtime.state.get("frame_retry_count", 30))
        frames = None
        color_frame = None
        depth_frame = None
        for _ in range(retry_count):
            frames = runtime.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                continue
            if runtime.aligner is not None:
                frames = runtime.aligner.process(frames)
                if not frames:
                    continue
            if hasattr(frames, "as_frame_set"):
                frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame and depth_frame:
                break

        if not frames:
            raise RuntimeError(f"Timed out waiting for Orbbec frames from {runtime.info.serial_number}.")
        if not color_frame or not depth_frame:
            raise RuntimeError(f"Missing color or depth frame from Orbbec pipeline {runtime.info.serial_number}.")

        color = _reshape_color_frame(color_frame)
        if color is None:
            color_format = _enum_name(color_frame.get_format())
            raise RuntimeError(f"Unsupported Orbbec color format '{color_format}'.")

        depth_height = int(depth_frame.get_height())
        depth_width = int(depth_frame.get_width())
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_height, depth_width))
        # Orbbec examples treat depth_frame.get_depth_scale() output as millimeters.
        depth_scale = float(depth_frame.get_depth_scale()) * 0.001
        runtime.depth_scale = depth_scale

        color_profile = color_frame.get_stream_profile()
        depth_profile = depth_frame.get_stream_profile()
        resolved_stream = runtime.resolved_stream.updated(
            actual_fps=int(color_profile.as_video_stream_profile().get_fps()),
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
            timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_index()),
            depth_scale=depth_scale,
            color_intrinsics=intrinsics_to_dict(color_profile),
            depth_intrinsics=intrinsics_to_dict(depth_profile),
            resolved_stream=resolved_stream,
        )

    def close_runtime(self, runtime: CameraRuntime) -> None:
        try:
            runtime.pipeline.stop()
        except Exception:
            pass


BACKEND = OrbbecBackend()
