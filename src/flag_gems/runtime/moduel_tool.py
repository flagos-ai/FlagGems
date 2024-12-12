import triton

from . import backend
from .backend.device import DeviceDetector

device = DeviceDetector()
backend.set_torch_backend_device_fn(device.vendor_name)
tl_extra_module = None
if tl_extra_module is None:
    try:
        backend.set_tl_extra_backend_module(device.vendor_name)
        tl_extra_module = backend.get_tl_extra_backend_module()
    except ImportError:
        try:
            tl_extra_module = triton.language.math
        except ImportError:
            tl_extra_module = triton.language.libdevice
