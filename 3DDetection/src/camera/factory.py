from __future__ import annotations

from importlib import import_module
from typing import Any

from .types import CameraRuntime, DeviceInfo, FrameBundle, StreamRequest

REGISTERED_BACKENDS: dict[str, str] = {
    "realsense": "camera.realsense_backend",
    "orbbec": "camera.orbbec_backend",
}


def _load_backend(name: str) -> Any:
    module_path = REGISTERED_BACKENDS.get(name)
    if module_path is None:
        supported = ", ".join(sorted(REGISTERED_BACKENDS))
        raise RuntimeError(f"Unsupported camera backend '{name}'. Supported backends: {supported}.")
    module = import_module(module_path)
    return module.BACKEND


def available_backend_names() -> list[str]:
    names: list[str] = []
    for name in REGISTERED_BACKENDS:
        backend = _load_backend(name)
        if backend.is_available():
            names.append(name)
    return names


def get_backend(name: str):
    backend = _load_backend(name)
    if not backend.is_available():
        reason = backend.unavailable_reason()
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(f"Camera backend '{name}' is unavailable{suffix}.")
    return backend


def _candidate_backends(name: str):
    if name == "auto":
        return [get_backend(candidate) for candidate in available_backend_names()]
    return [get_backend(name)]


def enumerate_devices(name: str = "auto") -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for backend in _candidate_backends(name):
        devices.extend(backend.enumerate_devices())
    return devices


def select_device(
    devices: list[DeviceInfo],
    serial: str = "",
    backend_name: str = "auto",
) -> DeviceInfo:
    if not devices:
        target = "camera devices" if backend_name == "auto" else f"{backend_name} camera devices"
        raise RuntimeError(f"No {target} found.")

    if serial:
        matches = [device for device in devices if device.serial_number == serial]
        if not matches:
            raise RuntimeError(f"Requested serial '{serial}' was not found.")
        if len(matches) > 1:
            matches_by_backend = sorted({device.backend for device in matches})
            raise RuntimeError(
                f"Serial '{serial}' is ambiguous across backends {matches_by_backend}. "
                "Pass --camera-backend explicitly."
            )
        return matches[0]

    if len(devices) == 1:
        return devices[0]

    device_list = ", ".join(f"{device.backend}:{device.serial_number}" for device in devices)
    raise RuntimeError(
        "Multiple camera devices are available. Pass --serial to choose one. "
        f"Detected devices: {device_list}."
    )


def open_runtime(device_info: DeviceInfo, stream_request: StreamRequest) -> CameraRuntime:
    backend = get_backend(device_info.backend)
    return backend.open_runtime(device_info, stream_request)


def read_frame_bundle(runtime: CameraRuntime) -> FrameBundle:
    backend = get_backend(runtime.backend_name)
    return backend.read_frame_bundle(runtime)


def close_runtime(runtime: CameraRuntime) -> None:
    backend = get_backend(runtime.backend_name)
    backend.close_runtime(runtime)


def stop_runtimes(runtimes: list[CameraRuntime]) -> None:
    for runtime in runtimes:
        close_runtime(runtime)
