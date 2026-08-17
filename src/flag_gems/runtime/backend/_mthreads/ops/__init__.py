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
from .bucketize import bucketize, bucketize_kernel
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
from .erfinv import erfinv, erfinv_kernel
from .erfinv_ import erfinv_
from .flip import flip
from .fmod_ import (
    fmod_,
    fmod_inplace_scalar_kernel,
    fmod_inplace_tensor_kernel,
    fmod_scalar_,
    fmod_tensor_,
)
from .gather import gather, gather_backward
from .histc import histc, histc_kernel, histc_local_reduce_kernel
from .im2col import im2col, im2col_kernel
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_, index_copy_kernel
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .linalg_cholesky import cholesky_kernel, linalg_cholesky
from .log import log
from .log10 import log10, log10_, log10_out
from .log_normal_ import log_normal_, log_normal_kernel, pair_uniform_to_normal
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .max import max, max_dim
from .median import (
    median,
    median_dim,
    median_dim_values,
    median_out,
    median_sort_select_kernel,
)
from .min import min, min_dim
from .mish import mish, mish_, mish_kernel
from .mode import mode
from .mul import mul, mul_
from .nonzero_numpy import (
    nonzero_count_kernel,
    nonzero_fill_kernel,
    nonzero_numpy,
    nonzero_single_kernel,
)
from .norm import (
    norm,
    norm_finalize_kernel,
    norm_partial_kernel,
    norm_scalar,
    norm_scalaropt_dim,
)
from .normal import normal_
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .pad import constant_pad_nd
from .permute_copy import permute_copy
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reflection_pad3d_backward import (
    reflection_pad3d_backward,
    reflection_pad3d_backward_kernel,
)
from .renorm_ import renorm_, renorm_kernel, renorm_kernel_single_pass
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_conj import resolve_conj
from .round_ import round_, round_inplace_kernel
from .softplus_backward import softplus_backward, softplus_backward_kernel
from .sort import sort, sort_stable
from .special_gammainc import gammainc_kernel, special_gammainc
from .tile import tile
from .trunc import trunc, trunc_, trunc_kernel
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
    "celu",
    # "celu_",
    "conv2d",
    "dropout",
    "dropout_backward",
    "flip",
    "gather",
    "gather_backward",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "_index_put_impl_",
    "index_select",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "max",
    "max_dim",
    "min",
    "min_dim",
    "mode",
    "mul",
    "mul_",
    "normal_",
    "one_hot",
    "ones",
    "ones_like",
    "constant_pad_nd",
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
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "div_mode",
    "div_mode_",
    "floor_divide",
    "floor_divide_",
    "_unique2",
    "w8a8_block_fp8_matmul",
    "zero_",
    "zeros",
    "zeros_like",
    "bucketize",
    "bucketize_kernel",
    "cholesky_kernel",
    "erfinv",
    "erfinv_",
    "erfinv_kernel",
    "fmod_",
    "fmod_inplace_scalar_kernel",
    "fmod_inplace_tensor_kernel",
    "fmod_scalar_",
    "fmod_tensor_",
    "gammainc_kernel",
    "histc",
    "histc_kernel",
    "histc_local_reduce_kernel",
    "im2col",
    "im2col_kernel",
    "index_copy",
    "index_copy_",
    "index_copy_kernel",
    "linalg_cholesky",
    "log_normal_",
    "log_normal_kernel",
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "median_sort_select_kernel",
    "mish",
    "mish_",
    "mish_kernel",
    "nonzero_count_kernel",
    "nonzero_fill_kernel",
    "nonzero_numpy",
    "nonzero_single_kernel",
    "norm",
    "norm_finalize_kernel",
    "norm_partial_kernel",
    "norm_scalar",
    "norm_scalaropt_dim",
    "pair_uniform_to_normal",
    "permute_copy",
    "reflection_pad3d_backward",
    "reflection_pad3d_backward_kernel",
    "renorm_",
    "renorm_kernel",
    "renorm_kernel_single_pass",
    "round_",
    "round_inplace_kernel",
    "softplus_backward",
    "softplus_backward_kernel",
    "special_gammainc",
    "trunc",
    "trunc_",
    "trunc_kernel",
]


if get_device_capability(current_device())[0] >= 3:
    from .addmm import addmm, addmm_dtype, addmm_dtype_out  # noqa: F401
    from .baddbmm import baddbmm  # noqa: F401
    from .bmm import bmm  # noqa: F401
    from .gelu import gelu  # noqa: F401
    from .mm import mm  # noqa: F401
    from .tanh import tanh  # noqa: F401

    __all__.extend(
        [
            "addmm",
            "addmm_dtype",
            "addmm_dtype_out",
            "baddbmm",
            "bmm",
            "gelu",
            "mm",
            "tanh",
        ]
    )
