from torch_musa import current_device, get_device_capability

from .isin import isin
from .unique import _unique2

__all__ = [
    "_unique2",
    "isin",
]

if get_device_capability(current_device())[0] >= 3:
    from .addmm import addmm
    from .bmm import bmm
    from .gelu import gelu
    from .mm import mm
    from .tanh import tanh

    __all__ += ["gelu"]
    __all__ += ["tanh"]
    __all__ += ["mm"]
    __all__ += ["addmm"]
    __all__ += ["bmm"]
