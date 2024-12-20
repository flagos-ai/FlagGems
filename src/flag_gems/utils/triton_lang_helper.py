import triton

from ..runtime import backend
from ..runtime.backend.device import DeviceDetector

"""
    To be compatible with different versions of math libraries
    tl_extra_shim will be selected to a specific library.
    And the "triton.language.extra" module is only available in
    Triton 2.2 and later versions.
"""

device = DeviceDetector()
backend.set_torch_backend_device_fn(device.vendor_name)
tl_extra_shim = None
if tl_extra_shim is None:
    try:
        backend.set_tl_extra_backend_module(device.vendor_name)
        tl_extra_shim = backend.get_tl_extra_backend_module()
    except ImportError:
        try:
            tl_extra_shim = triton.language.math
        except ImportError:
            tl_extra_shim = triton.language.libdevice
