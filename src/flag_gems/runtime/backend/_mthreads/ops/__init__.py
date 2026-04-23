from torch_musa import current_device, get_device_capability

from .all import all, all_dim, all_dims
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .argmin import argmin
from .batch_norm import batch_norm, batch_norm_backward
from .celu import celu
from .conv2d import conv2d
from .dropout import dropout, dropout_backward
from .gather import gather, gather_backward
from .index_put import index_put, index_put_
from .log import log
from .max import max, max_dim
from .min import min, min_dim
from .ones import ones
from .ones_like import ones_like
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .resolve_conj import resolve_conj
from .sort import sort, sort_stable
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "rand",
    "rand_like",
    "dropout",
    "dropout_backward",
    "celu",
    # "celu_",
    "ones",
    "ones_like",
    "randn",
    "randn_like",
    "zeros",
    "zero_",
    "zeros_like",
    "log",
    "sort",
    "arange",
    "arange_start",
    "sort_stable",
    "randperm",
    "conv2d",
    "all",
    "all_dim",
    "all_dims",
    "any",
    "any_dim",
    "any_dims",
    "argmin",
    "prod",
    "prod_dim",
    "min",
    "min_dim",
    "max",
    "max_dim",
    "batch_norm",
    "batch_norm_backward",
    "gather",
    "gather_backward",
    "index_put",
    "index_put_",
    "resolve_conj",
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
