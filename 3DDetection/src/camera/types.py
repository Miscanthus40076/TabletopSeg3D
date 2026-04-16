from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    name: str
    serial_number: str
    firmware_version: str = ""
    vendor: str = ""
    product_line: str = ""
    usb_type_descriptor: str = ""


@dataclass(frozen=True)
class StreamRequest:
    width: int
    height: int
    fps: int
    align_to_color: bool = True


@dataclass(frozen=True)
class ResolvedStreamConfig:
    requested_width: int
    requested_height: int
    requested_fps: int
    actual_color_size: tuple[int, int]
    actual_depth_size: tuple[int, int]
    actual_fps: int

    def updated(
        self,
        *,
        actual_color_size: tuple[int, int] | None = None,
        actual_depth_size: tuple[int, int] | None = None,
        actual_fps: int | None = None,
    ) -> "ResolvedStreamConfig":
        return replace(
            self,
            actual_color_size=self.actual_color_size if actual_color_size is None else actual_color_size,
            actual_depth_size=self.actual_depth_size if actual_depth_size is None else actual_depth_size,
            actual_fps=self.actual_fps if actual_fps is None else int(actual_fps),
        )


@dataclass
class CameraRuntime:
    backend_name: str
    info: DeviceInfo
    pipeline: Any
    aligner: Any | None = None
    depth_scale: float = 1.0
    resolved_stream: ResolvedStreamConfig | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameBundle:
    backend: str
    serial_number: str
    device_name: str
    firmware_version: str
    product_line: str
    usb_type_descriptor: str
    color: Any
    depth: Any
    timestamp_ms: float
    frame_number: int
    depth_scale: float
    color_intrinsics: dict[str, Any]
    depth_intrinsics: dict[str, Any]
    resolved_stream: ResolvedStreamConfig
