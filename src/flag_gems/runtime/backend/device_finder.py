# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch  # noqa: F401

from .. import backend, error
from ..common import _VENDOR_TORCH_ATTR, vendors


def _privateuse1_backend_name():
    """Name the running PyTorch build gave to its PrivateUse1 backend."""
    try:
        return str(torch._C._get_privateuse1_backend_name())
    except Exception:
        return None


def resolve_guard_device_name(declared_name: str) -> str:
    """Return the device name that tensors on this build actually report.

    Ops guard their inputs with ``x.device.type != flag_gems.device``, so
    ``flag_gems.device`` has to name what tensors report, not what the vendor
    descriptor assumes. Most vendors expose their accelerator as
    ``torch.<declared_name>``, but some PyTorch builds register the same
    hardware as a PrivateUse1 backend under a different name (Iluvatar's
    ``torch_fl`` registers ``flagos``); there every guarded op would reject
    every tensor the platform can hand it (issue #7135).

    ``GEMS_DEVICE_NAME`` / ``FLAGGEMS_DEVICE_NAME`` force the name when the
    probe cannot decide.
    """
    override = os.environ.get("GEMS_DEVICE_NAME") or os.environ.get(
        "FLAGGEMS_DEVICE_NAME"
    )
    if override:
        return override.strip().lower()

    module = getattr(torch, declared_name, None)
    if module is not None:
        try:
            is_available = getattr(module, "is_available", None)
            if not callable(is_available) or is_available():
                return declared_name
        except Exception:
            return declared_name

    registered = _privateuse1_backend_name()
    if registered and registered not in ("privateuseone", declared_name):
        if getattr(torch, registered, None) is not None:
            return registered
    return declared_name


# A singleton class to manage device context.
class DeviceDetector:
    """Singleton class to manage device context."""

    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(DeviceDetector, cls).__new__(cls)
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, vendor_name=None):
        if not hasattr(self, "initialized"):
            self.initialized = True
            # A list of all available vendor names.
            self.vendor_list = vendors.get_all_vendors().keys()
            # A dataclass instance, get the vendor information based on the provided or default vendor name.
            self.info = self.get_vendor(vendor_name)
            # vendor_name is like 'nvidia', device_name is like 'cuda'.
            self.vendor_name = self.info.vendor_name
            # ``name`` backs ``flag_gems.device`` and is compared against
            # ``Tensor.device.type`` by every guarded op, so it follows the
            # live build rather than the descriptor (issue #7135).
            self.name = resolve_guard_device_name(self.info.device_name)
            self.vendor = vendors.get_all_vendors()[self.vendor_name]
            # The dispatch key stays tied to the descriptor: torch registers
            # PrivateUse1 kernels under the ``PrivateUse1`` key, never under
            # the renamed backend name.
            self.dispatch_key = (
                self.info.device_name.upper()
                if self.info.dispatch_key is None
                else self.info.dispatch_key
            )
            self.device_count = backend.gen_torch_device_object(
                self.vendor_name
            ).device_count()
            self.support_fp64 = self.info.fp64_enabled
            self.support_bf16 = self.info.bf16_enabled
            self.support_int64 = self.info.int64_enabled
            self.support_fp8 = self.info.fp8_enabled
            self.support_complex = self.info.complex_enabled

    def get_vendor(self, vendor_name=None) -> tuple:
        # Try to get the vendor name from a quick special command like 'torch.mlu'.
        vendor_from_env = self._get_vendor_from_env()
        if vendor_from_env:
            return backend.get_vendor_info(vendor_from_env)

        vendor_name = self._get_vendor_from_quick_cmd()
        if vendor_name:
            return backend.get_vendor_info(vendor_name)
        try:
            # Obtaining a vendor_info from the methods provided by torch or triton, but is not currently implemented.
            return self._get_vendor_from_lib()
        except Exception:
            return self._get_vendor_from_sys()

    def _get_vendor_from_quick_cmd(self):
        try:
            import torch_npu

            torch_module = torch_npu
        except ImportError:
            torch_module = torch

        try:
            import torch_txda  # noqa: F401 — trigger txda backend registration on torch
        except ImportError:
            pass

        for vendor_name, attr in _VENDOR_TORCH_ATTR.items():
            if hasattr(torch_module, attr):
                return str(vendor_name)

        if hasattr(torch_module, "cuda") and hasattr(
            torch_module.cuda, "get_device_properties"
        ):
            try:
                prop = torch_module.cuda.get_device_properties(0)
                if "NVIDIA" in prop.name.upper():
                    return "nvidia"
            except Exception:
                return False

        return False

    def _get_vendor_from_env(self):
        if "PPU_SDK" in os.environ.keys():
            return "thead"

        env_keys = (
            "GEMS_VENDOR",
            "FLAGGEMS_VENDOR",
            "GEMS_BACKEND",
            "FLAGGEMS_BACKEND",
        )
        for key in env_keys:
            if key in os.environ:
                return str(os.environ.get(key).lower())

        return False

    def _get_vendor_from_sys(self):
        vendor_infos = backend.get_vendor_infos()

        def check_vendor(info):
            try:
                cmd_args = shlex.split(info.device_query_cmd)
                result = subprocess.run(cmd_args, capture_output=True, text=True)
                return info if result.returncode == 0 else None
            except Exception:
                return None

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(check_vendor, info): info for info in vendor_infos
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    return result

        error.device_not_found()

    def get_vendor_name(self):
        return self.vendor_name

    def _get_vendor_from_lib(self):
        # Reserve the associated interface for triton or torch
        # although they are not implemented yet.
        # try:
        #     return triton.get_vendor_info()
        # except Exception:
        #     return torch.get_vendor_info()
        raise RuntimeError("The method is not implemented")
