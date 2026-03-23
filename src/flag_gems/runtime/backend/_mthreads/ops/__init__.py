from torch_musa import current_device, get_device_capability

from .all import all, all_dim, all_dims
from .amax import amax
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .argmin import argmin
from .batch_norm import batch_norm, batch_norm_backward
from .celu import celu
from .conv2d import conv2d
from .dropout import dropout, dropout_backward
from .gather import gather, gather_backward
from .index_add import index_add, index_add_
from .index_put import index_put, index_put_
from .index_select import index_select
from .log import log
from .log_softmax import log_softmax, log_softmax_backward
from .max import max, max_dim
from .min import min, min_dim
from .normal import normal_
from .ones import ones
from .ones_like import ones_like
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_conj import resolve_conj
from .sort import sort, sort_stable
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "amax",
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
    "log_softmax",
    "log_softmax_backward",
    "sort",
    "arange",
    "arange_start",
    "sort_stable",
    "randperm",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
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
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "resolve_conj",
    "normal_",
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
