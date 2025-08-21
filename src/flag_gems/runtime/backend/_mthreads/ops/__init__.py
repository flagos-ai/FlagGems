from torch_musa import current_device, get_device_capability

from .attention import (
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
)
from .dropout import dropout, dropout_backward
from .ones import ones
from .ones_like import ones_like
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "flash_attn_varlen_func",
    "scaled_dot_product_attention",
    "flash_attention_forward",
    "rand",
    "rand_like",
    "dropout",
    "dropout_backward",
    "ones",
    "ones_like",
    "randn",
    "randn_like",
    "zeros",
    "zeros_like",
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
