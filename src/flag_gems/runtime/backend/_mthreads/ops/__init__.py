from torch_musa import current_device, get_device_capability

from .isin import isin
from .unique import _unique2

__all__ = [
    "_unique2",
    "isin",
]

if get_device_capability(current_device())[0] >= 3:
    __all__.append("gelu")
    __all__.append("tanh")
