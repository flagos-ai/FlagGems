# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch_musa import current_device, get_device_capability

from .all import all, all_dim, all_dims
from .amax import amax
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .argmin import argmin
from .batch_norm import batch_norm, batch_norm_backward
from .bucketize import bucketize
from .celu import celu
from .conv2d import conv2d
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .dropout import dropout, dropout_backward
from .erfinv import erfinv
from .erfinv_ import erfinv_
from .feature_dropout import feature_dropout_
from .flip import flip
from .fmod_ import fmod_, fmod_scalar_, fmod_tensor_
from .gather import gather, gather_backward
from .histc import histc
from .im2col import im2col
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .linalg_cholesky import linalg_cholesky
from .linear import linear
from .log import log
from .log10 import log10, log10_, log10_out
from .log_normal_ import log_normal_
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .max import max, max_dim
from .median import median, median_dim, median_dim_values, median_out
from .min import min, min_dim
from .mish import mish, mish_
from .mode import mode
from .mul import mul, mul_
from .nonzero_numpy import nonzero_numpy
from .norm import norm, norm_scalar, norm_scalaropt_dim
from .normal import normal_
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .pad import constant_pad_nd, pad
from .permute_copy import permute_copy
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reflection_pad3d_backward import reflection_pad3d_backward
from .renorm_ import renorm_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_conj import resolve_conj
from .round_ import round_
from .softplus_backward import softplus_backward
from .sort import sort, sort_stable
from .special_gammainc import special_gammainc
from .tile import tile
from .trunc import trunc, trunc_
from .unique import _unique2
from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
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
    "bucketize",
    "celu",
    # "celu_",
    "conv2d",
    "dropout",
    "dropout_backward",
    "erfinv",
    "erfinv_",
    "feature_dropout_",
    "flip",
    "fmod_",
    "fmod_scalar_",
    "fmod_tensor_",
    "gather",
    "gather_backward",
    "histc",
    "im2col",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "_index_put_impl_",
    "index_select",
    "linalg_cholesky",
    "linear",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log_normal_",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "max",
    "max_dim",
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "min",
    "min_dim",
    "mish",
    "mish_",
    "mode",
    "mul",
    "mul_",
    "nonzero_numpy",
    "norm",
    "norm_scalar",
    "norm_scalaropt_dim",
    "normal_",
    "one_hot",
    "ones",
    "ones_like",
    "constant_pad_nd",
    "pad",
    "permute_copy",
    "prod",
    "prod_dim",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randperm",
    "reflection_pad3d_backward",
    "renorm_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "resolve_conj",
    "round_",
    "softplus_backward",
    "sort",
    "sort_stable",
    "special_gammainc",
    "tile",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "div_mode",
    "div_mode_",
    "floor_divide",
    "floor_divide_",
    "_unique2",
    "trunc",
    "trunc_",
    "w8a8_block_fp8_matmul",
    "zero_",
    "zeros",
    "zeros_like",
]


if get_device_capability(current_device())[0] >= 3:
    from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out  # noqa: F401
    from .baddbmm import baddbmm, baddbmm_out  # noqa: F401
    from .bmm import bmm  # noqa: F401
    from .gelu import gelu  # noqa: F401
    from .mm import mm  # noqa: F401
    from .tanh import tanh  # noqa: F401

    __all__.extend(
        [
            "addmm",
            "addmm_dtype",
            "addmm_dtype_out",
            "addmm_out",
            "baddbmm",
            "baddbmm_out",
            "bmm",
            "gelu",
            "mm",
            "tanh",
        ]
    )
