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
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_conj import resolve_conj
from .sort import sort, sort_stable
from .tile import tile
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "amax",
    "all",
    "all_dim",
    "all_dims",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "argmin",
    "batch_norm",
    "batch_norm_backward",
    "celu",
    # "celu_",
    "conv2d",
    "dropout",
    "dropout_backward",
    "gather",
    "gather_backward",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "log",
    "log_softmax",
    "log_softmax_backward",
    "max",
    "max_dim",
    "min",
    "min_dim",
    "normal_",
    "one_hot",
    "ones",
    "ones_like",
    "prod",
    "prod_dim",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randperm",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "resolve_conj",
    "sort",
    "sort_stable",
    "tile",
    "zero_",
    "zeros",
    "zeros_like",
]

if get_device_capability(current_device())[0] >= 3:
    from .addmm import addmm  # noqa: F401
    from .bmm import bmm  # noqa: F401
    from .gelu import gelu  # noqa: F401
    from .mm import mm  # noqa: F401
    from .tanh import tanh  # noqa: F401

    __all__.extend(
        [
            "addmm",
            "bmm",
            "gelu",
            "mm",
            "tanh",
        ]
    )
