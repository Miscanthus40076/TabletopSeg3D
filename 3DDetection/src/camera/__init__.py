"""Camera backend registry and shared capture types."""

from .factory import (
    available_backend_names,
    close_runtime,
    enumerate_devices,
    get_backend,
    open_runtime,
    read_frame_bundle,
    select_device,
    stop_runtimes,
)
from .types import CameraRuntime, DeviceInfo, FrameBundle, ResolvedStreamConfig, StreamRequest

__all__ = [
    "CameraRuntime",
    "DeviceInfo",
    "FrameBundle",
    "ResolvedStreamConfig",
    "StreamRequest",
    "available_backend_names",
    "close_runtime",
    "enumerate_devices",
    "get_backend",
    "open_runtime",
    "read_frame_bundle",
    "select_device",
    "stop_runtimes",
]
