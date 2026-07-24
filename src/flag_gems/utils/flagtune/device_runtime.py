"""Provide the only device-operation boundary used by FlagGems FlagTune.

FlagTune scheduling and execution consume :class:`DeviceRuntime` instead of
calling ``torch.cuda`` or interpreting visibility variables themselves.  Each
registered Triton backend owns its PyTorch API namespace, visibility variables,
native architecture, synchronization, and device selection behavior.

Detection is strict: unknown backends, missing APIs, unavailable devices, and
metadata failures raise a device-boundary exception before benchmark work is
scheduled.  No NVIDIA or A100 fallback metadata is permitted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from triton.flagtune.device import (
    DeviceDescriptor,
    DeviceProbeError,
    probe_flagtune_device,
)
from triton.flagtune.identity import gpu_metadata


class DeviceUnavailableError(DeviceProbeError):
    """Report a supported backend that has no usable process-visible device."""


@dataclass(frozen=True)
class _RuntimeBackend:
    visibility_variables: tuple[str, ...]


_RUNTIME_BACKENDS = {
    "cuda": _RuntimeBackend(("CUDA_VISIBLE_DEVICES",)),
    # PyTorch ROCm uses torch.cuda while launchers may use any of these names.
    "hip": _RuntimeBackend(
        ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    ),
}


class DeviceRuntime:
    """Execute device operations for one explicitly supported Triton backend."""

    def __init__(self, descriptor: DeviceDescriptor, torch_module: Any) -> None:
        backend = _RUNTIME_BACKENDS.get(descriptor.backend)
        if backend is None:
            # FlagTree normally rejects this first; keep the execution boundary
            # strict if a synthetic descriptor is supplied by a caller or test.
            raise DeviceProbeError(
                f"FlagGems FlagTune has no runtime adapter for backend "
                f"{descriptor.backend!r}"
            )
        device_api = getattr(torch_module, descriptor.torch_device_type, None)
        if device_api is None:
            raise DeviceProbeError(
                f"backend {descriptor.backend!r} requires torch."
                f"{descriptor.torch_device_type}, but that API is unavailable"
            )
        self.descriptor = descriptor
        self._backend = backend
        self._torch_module = torch_module
        self._device_api = device_api

    @property
    def backend(self) -> str:
        return self.descriptor.backend

    @property
    def visibility_variables(self) -> tuple[str, ...]:
        return self._backend.visibility_variables

    def device_count(self) -> int:
        """Return the process-visible accelerator count or fail explicitly."""
        try:
            count = int(self._device_api.device_count())
        except Exception as exc:
            raise DeviceProbeError(
                f"cannot query visible {self.backend!r} devices: {exc}"
            ) from exc
        if count <= 0:
            raise DeviceUnavailableError(
                f"FlagTune found backend {self.backend!r} but no visible devices"
            )
        return count

    def is_available(self) -> bool:
        """Return backend availability without suppressing API failures."""
        try:
            return bool(self._device_api.is_available())
        except Exception as exc:
            raise DeviceProbeError(
                f"cannot query availability for backend {self.backend!r}: {exc}"
            ) from exc

    def set_device(self, index: int) -> None:
        """Select one process-visible logical device."""
        try:
            self._device_api.set_device(int(index))
        except Exception as exc:
            raise DeviceProbeError(
                f"cannot select {self.backend!r} device {index}: {exc}"
            ) from exc

    def synchronize(self) -> None:
        """Synchronize the selected device through its registered PyTorch API."""
        try:
            self._device_api.synchronize()
        except Exception as exc:
            raise DeviceProbeError(
                f"cannot synchronize backend {self.backend!r}: {exc}"
            ) from exc

    def dtype(self, name: str) -> Any:
        """Resolve one validated dtype name from the lazily imported torch module."""
        try:
            return getattr(self._torch_module, name)
        except AttributeError as exc:
            raise DeviceProbeError(
                f"installed torch has no dtype {name!r}"
            ) from exc

    def make_tensor(
        self,
        factory_name: str,
        shape: Sequence[int],
        *,
        dtype: Any,
        device_index: int = 0,
    ) -> Any:
        """Allocate one tensor through an allowlisted torch factory."""
        try:
            factory = getattr(self._torch_module, factory_name)
            device = self._torch_module.device(
                self.descriptor.torch_device_type, int(device_index)
            )
            return factory(tuple(shape), device=device, dtype=dtype)
        except Exception as exc:
            raise DeviceProbeError(
                f"cannot allocate {factory_name} tensor on backend "
                f"{self.backend!r}: {exc}"
            ) from exc

    def visible_device_tokens(
        self,
        visible_count: int | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Map process-visible ordinals back to launcher tokens.

        The first non-empty backend-specific visibility variable is
        authoritative.  A short value is rejected before workers launch.
        """
        count = self.device_count() if visible_count is None else int(visible_count)
        source = os.environ if environ is None else environ
        raw = ""
        variable = ""
        for name in self.visibility_variables:
            candidate = str(source.get(name, "")).strip()
            if candidate:
                raw = candidate
                variable = name
                break
        if not raw:
            return [str(index) for index in range(count)]
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) < count:
            raise DeviceProbeError(
                f"{variable} contains {len(tokens)} entries but backend "
                f"{self.backend!r} reports {count} visible devices"
            )
        return tokens[:count]

    def apply_worker_visibility(
        self, environment: MutableMapping[str, str], token: str
    ) -> None:
        """Expose exactly one launcher token using backend-owned variables."""
        primary, *aliases = self.visibility_variables
        environment[primary] = str(token)
        for name in aliases:
            environment.pop(name, None)

    def metadata(self, device_index: int = 0) -> dict[str, Any]:
        """Return canonical model metadata for one process-visible device."""
        descriptor = probe_flagtune_device(device_index)
        if descriptor.backend != self.backend:
            raise DeviceProbeError(
                f"active backend changed from {self.backend!r} to "
                f"{descriptor.backend!r} during device probing"
            )
        return dict(
            gpu_metadata(
                backend=descriptor.backend,
                vendor=descriptor.vendor,
                device_name=descriptor.device_name,
                architecture=descriptor.architecture,
            )
        )


@dataclass(frozen=True)
class FlagTuneEnvironment:
    """Hold one validated runtime plus every process-visible device."""

    runtime: DeviceRuntime
    devices: tuple[DeviceDescriptor, ...]

    @property
    def device_count(self) -> int:
        return len(self.devices)


def probe_flagtune_environment() -> FlagTuneEnvironment:
    """Validate the active backend and all process-visible devices at one boundary."""
    descriptor = probe_flagtune_device()
    try:
        import torch
    except ImportError as exc:
        raise DeviceProbeError(
            "FlagTune device probing requires PyTorch"
        ) from exc
    runtime = DeviceRuntime(descriptor, torch)
    if not runtime.is_available():
        raise DeviceUnavailableError(
            f"FlagTune backend {descriptor.backend!r} is not available in "
            "the installed PyTorch runtime"
        )
    count = runtime.device_count()
    devices = tuple(probe_flagtune_device(index) for index in range(count))
    mismatched = [
        item.backend for item in devices if item.backend != descriptor.backend
    ]
    if mismatched:
        raise DeviceProbeError(
            f"visible devices do not share backend {descriptor.backend!r}: "
            f"{mismatched}"
        )
    return FlagTuneEnvironment(runtime=runtime, devices=devices)
