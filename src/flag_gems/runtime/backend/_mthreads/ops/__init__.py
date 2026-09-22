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

from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from ._conj import _conj
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._masked_scale import _masked_scale
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from ._upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from .adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addmv_ import addmv_
from .all import all, all_dim, all_dims
from .amax import amax
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arctan2 import arctan2
from .arctan2_ import arctan2_
from .arctan_ import arctan_
from .argmin import argmin
from .argsort import argsort
from .atan2 import atan2
from .atan2_ import atan2_
from .avg_pool2d import avg_pool2d_backward
from .batch_norm import batch_norm, batch_norm_backward
from .bucketize import bucketize
from .celu import celu
from .channel_shuffle import channel_shuffle
from .cholesky_inverse import cholesky_inverse
from .clip_ import clip_
from .conv2d import conv2d
from .conv_transpose1d import conv_transpose1d, conv_transpose1d_output_size
from .conv_transpose2d import conv_transpose2d
from .copy import copy_
from .cudnn_convolution import cudnn_convolution
from .deg2rad_ import deg2rad_
from .diagonal_scatter import diagonal_scatter
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
from .expand_copy import expand_copy
from .feature_dropout import feature_dropout, feature_dropout_
from .fix import fix
from .flip import flip
from .fmod_ import fmod_, fmod_scalar_, fmod_tensor_
from .gather import gather, gather_backward
from .gcd_ import gcd_
from .grid_sampler_3d_backward import grid_sampler_3d_backward
from .histc import histc
from .im2col import im2col
from .index_add import index_add, index_add_
from .index_copy import index_copy
from .index_copy_ import index_copy_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .int_mm import int_mm, int_mm_out
from .isin import isin
from .kthvalue import kthvalue
from .lcm import lcm
from .lcm_ import lcm_
from .lift_fresh import lift_fresh
from .lift_out import lift_out
from .linalg_cholesky import linalg_cholesky
from .linalg_ldl_factor_ex import ldl_factor_ex
from .linalg_ldl_solve import linalg_ldl_solve
from .linear import linear
from .log import log
from .log2_ import log2_
from .log10 import log10, log10_, log10_out
from .log_normal_ import log_normal_
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .max import max, max_dim
from .max_pool2d_with_indices_backward import max_pool2d_with_indices_backward
from .median import median, median_dim, median_dim_values, median_out
from .min import min, min_dim
from .mish import mish
from .mish_ import mish_
from .mode import mode
from .mul import mul, mul_
from .mvlgamma import mvlgamma
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nonzero_numpy import nonzero_numpy
from .norm import norm, norm_scalar, norm_scalaropt_dim
from .normal import normal_
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .ormqr import ormqr
from .pad import constant_pad_nd, pad
from .permute_copy import permute_copy
from .polar import polar
from .prod import prod, prod_dim
from .quantile import quantile
from .rad2deg_ import rad2deg_
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
from .replication_pad2d import replication_pad2d
from .resolve_conj import resolve_conj
from .rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8
from .round_ import round_
from .scaled_grouped_mm import scaled_grouped_mm
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .silu_backward import silu_backward
from .softplus_backward import softplus_backward
from .sort import sort, sort_stable
from .special_bessel_j0 import special_bessel_j0
from .special_bessel_j1 import special_bessel_j1
from .special_bessel_y0 import special_bessel_y0
from .special_erfcx import special_erfcx
from .special_gammainc import special_gammainc
from .special_multigammaln import special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .square_ import square_
from .tile import tile
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .trunc import trunc, trunc_
from .unique import _unique2
from .upsample_bilinear2d import upsample_bilinear2d
from .upsample_linear1d_backward import upsample_linear1d_backward
from .var import var, var_correction, var_dim
from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "_adaptive_avg_pool2d_backward",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_conj",
    "_functional_sym_constrain_range_for_size",
    "_index_put_impl_",
    "_masked_scale",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unique2",
    "_upsample_bilinear2d_aa",
    "_upsample_nearest_exact2d_backward",
    "adaptive_max_pool3d_backward",
    "addmv_",
    "all",
    "all_dim",
    "all_dims",
    "amax",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arctan2",
    "arctan2_",
    "arctan_",
    "argmin",
    "argsort",
    "atan2",
    "atan2_",
    "avg_pool2d_backward",
    "batch_norm",
    "batch_norm_backward",
    "bucketize",
    "celu",
    "channel_shuffle",
    "cholesky_inverse",
    "clip_",
    "constant_pad_nd",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose1d_output_size",
    "conv_transpose2d",
    "copy_",
    "cudnn_convolution",
    "deg2rad_",
    "diagonal_scatter",
    "div_mode",
    "div_mode_",
    "dropout",
    "dropout_backward",
    "erfinv",
    "erfinv_",
    "expand_copy",
    "feature_dropout",
    "feature_dropout_",
    "fix",
    "flip",
    "floor_divide",
    "floor_divide_",
    "fmod_",
    "fmod_scalar_",
    "fmod_tensor_",
    "gather",
    "gather_backward",
    "gcd_",
    "grid_sampler_3d_backward",
    "histc",
    "im2col",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_select",
    "int_mm",
    "int_mm_out",
    "isin",
    "kthvalue",
    "lcm",
    "lcm_",
    "ldl_factor_ex",
    "lift_fresh",
    "lift_out",
    "linalg_cholesky",
    "linalg_ldl_solve",
    "linear",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log2_",
    "log_normal_",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "max",
    "max_dim",
    "max_pool2d_with_indices_backward",
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
    "mvlgamma",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nonzero_numpy",
    "norm",
    "norm_scalar",
    "norm_scalaropt_dim",
    "normal_",
    "one_hot",
    "ones",
    "ones_like",
    "ormqr",
    "pad",
    "permute_copy",
    "polar",
    "prod",
    "prod_dim",
    "quantile",
    "rad2deg_",
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
    "replication_pad2d",
    "resolve_conj",
    "rms_norm_w8a16_fp8",
    "round_",
    "scaled_grouped_mm",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "silu_backward",
    "softplus_backward",
    "sort",
    "sort_stable",
    "special_bessel_j0",
    "special_bessel_j1",
    "special_bessel_y0",
    "special_erfcx",
    "special_gammainc",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "square_",
    "tile",
    "topk_w8a16_fp8",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc",
    "trunc_",
    "upsample_bilinear2d",
    "upsample_linear1d_backward",
    "var",
    "var_correction",
    "var_dim",
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
            "atan2",
            "kthvalue",
            "mvlgamma",
            "atan2_",
            "_upsample_nearest_exact2d_backward",
            "_upsample_bilinear2d_aa",
            "ldl_factor_ex",
            "_thnn_fused_lstm_cell_backward_impl",
            "lift_out",
            "lcm_",
            "lift_fresh",
            "addmv_",
            "ormqr",
            "cholesky_inverse",
            "square_",
            "deg2rad_",
            "clip_",
            "silu_backward",
            "log2_",
            "fix",
            "lcm",
            "rad2deg_",
            "arctan_",
            "_masked_scale",
            "_functional_sym_constrain_range_for_size",
            "diagonal_scatter",
            "_amp_foreach_non_finite_check_and_unscale_",
            "adaptive_max_pool3d_backward",
            "_amp_foreach_non_finite_check_and_unscale_",
            "_functional_sym_constrain_range_for_size",
            "_masked_scale",
            "_thnn_fused_lstm_cell_backward_impl",
            "_upsample_bilinear2d_aa",
            "_upsample_nearest_exact2d_backward",
            "adaptive_max_pool3d_backward",
            "addmv_",
            "arctan_",
            "atan2",
            "atan2_",
            "cholesky_inverse",
            "clip_",
            "deg2rad_",
            "diagonal_scatter",
            "fix",
            "kthvalue",
            "lcm",
            "lcm_",
            "lift_fresh",
            "lift_out",
            "ldl_factor_ex",
            "log2_",
            "mvlgamma",
            "ormqr",
            "rad2deg_",
            "silu_backward",
            "square_",
        ]
    )

if get_device_capability(current_device()) >= (3, 1):
    from .mm_w8a8_fp8 import mm_w8a8_fp8, mm_w8a8_fp8_out  # noqa: F401

    __all__.extend(["mm_w8a8_fp8", "mm_w8a8_fp8_out"])
