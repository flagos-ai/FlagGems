import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch  # noqa: F401

from .. import backend, error
from ..common import (
    _VENDOR_TORCH_ATTR,
    UNSUPPORT_BF16,
    UNSUPPORT_FP64,
    UNSUPPORT_INT64,
    vendors,
)


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
            self.name = self.info.device_name
            self.vendor = vendors.get_all_vendors()[self.vendor_name]
            self.dispatch_key = (
                self.name.upper()
                if self.info.dispatch_key is None
                else self.info.dispatch_key
            )
            self.device_count = backend.gen_torch_device_object(
                self.vendor_name
            ).device_count()
            self.support_fp64 = self.vendor not in UNSUPPORT_FP64
            self.support_bf16 = self.vendor not in UNSUPPORT_BF16
            self.support_int64 = self.vendor not in UNSUPPORT_INT64

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
        for vendor_name, attr in _VENDOR_TORCH_ATTR.items():
            if hasattr(torch, attr):
                return vendor_name
        try:
            import torch_npu

            for vendor_name, attr in _VENDOR_TORCH_ATTR.items():
                if hasattr(torch_npu, attr):
                    return vendor_name
        except ImportError:
            pass
        return None

    def _get_vendor_from_env(self):
        vendor = os.environ.get("GEMS_VENDOR")
        return vendor if vendor in self.vendor_list else None

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
