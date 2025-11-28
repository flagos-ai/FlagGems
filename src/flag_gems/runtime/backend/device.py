import os
import subprocess
import threading
from queue import Queue

import torch  # noqa: F401

from .. import backend, error
from ..commom_utils import vendors

UNSUPPORT_FP64 = [
    vendors.CAMBRICON,
    vendors.ILUVATAR,
    vendors.KUNLUNXIN,
    vendors.MTHREADS,
    vendors.AIPU,
    vendors.ASCEND,
]
UNSUPPORT_BF16 = [
    vendors.AIPU,
]
UNSUPPORT_INT64 = [
    vendors.AIPU,
]


# A singleton class to manage device context.
class DeviceDetector(object):
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(DeviceDetector, cls).__new__(cls)
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
        if vendor_from_env is not None:
            return backend.get_vendor_info(vendor_from_env)

        vendor_name = self._get_vendor_from_quick_cmd()
        if vendor_name is not None:
            return backend.get_vendor_info(vendor_name)
        try:
            # Obtaining a vendor_info from the methods provided by torch or triton, but is not currently implemented.
            return self._get_vendor_from_lib()
        except Exception:
            return self._get_vendor_from_sys()

    def _get_vendor_from_quick_cmd(self):
        cmd = {
            "cambricon": "mlu",
            "mthreads": "musa",
            "iluvatar": "corex",
            "ascend": "npu",
        }
        for vendor_name, flag in cmd.items():
            if hasattr(torch, flag):
                return vendor_name
        try:
            import torch_npu

            for vendor_name, flag in cmd.items():
                if hasattr(torch_npu, flag):
                    return vendor_name
        except:  # noqa: E722
            pass
        return None

    def _get_vendor_from_env(self):
        device_from_evn = os.environ.get("GEMS_VENDOR")
        return None if device_from_evn not in self.vendor_list else device_from_evn

    def _get_vendor_from_sys(self):
        vendor_infos = backend.get_vendor_infos()
        result_single_info = Queue()

        def runcmd(single_info):
            device_query_cmd = single_info.device_query_cmd
            try:
                result = subprocess.run(
                    [device_query_cmd], capture_output=True, text=True
                )
                if result.returncode == 0:
                    result_single_info.put(single_info)
            except:  # noqa: E722
                pass

        threads = []
        for single_info in vendor_infos:
            # Get the vendor information by running system commands.
            thread = threading.Thread(target=runcmd, args=(single_info,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        if result_single_info.empty():
            error.device_not_found()
        else:
            return result_single_info.get()

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
