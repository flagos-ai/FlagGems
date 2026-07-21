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

__all__ = []

from .abs import abs, abs_
from .add import add, add_
from .addmm import addmm
from .addmv import addmv, addmv_out
from .all import all, all_dim, all_dims
from .amax import amax
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start  # noqa: F401
from .argmax import argmax
from .argmin import argmin
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import bitwise_left_shift, bitwise_left_shift_
from .bitwise_not import bitwise_not, bitwise_not_  # noqa: F401
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_
from .bitwise_xor import (
    bitwise_xor_scalar,
    bitwise_xor_scalar_,
    bitwise_xor_scalar_tensor,
    bitwise_xor_tensor,
    bitwise_xor_tensor_,
)
from .bmm import bmm, bmm_out
from .cat import cat, cat_out
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .clamp_min import clamp_min, clamp_min_
from .clip import clip, clip_
from .conj_physical import conj_physical
from .contiguous import contiguous
from .copy import copy, copy_
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .div import (
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
    trunc_divide,
    trunc_divide_,
)
from .dropout import dropout
from .elu import elu
from .embedding import embedding
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1 import expm1, expm1_, expm1_out
from .exponential_ import exponential_
from .eye import eye
from .eye_m import eye_m
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .flip import flip
from .full import full
from .full_like import full_like
from .gather import gather
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .index import index
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .linear import linear
from .linspace import linspace
from .log import log
from .log10 import log10, log10_, log10_out
from .log_sigmoid import log_sigmoid
from .log_softmax import log_softmax
from .logaddexp import logaddexp
from .logical_and import logical_and
from .logical_not import logical_not
from .logical_or import logical_or
from .logical_xor import logical_xor
from .lt import lt, lt_scalar
from .masked_fill import masked_fill, masked_fill_
from .masked_select import masked_select
from .max import max, max_dim
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .nan_to_num import nan_to_num
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .nllloss import nll_loss_backward, nll_loss_forward
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .normal import normal_float_tensor, normal_tensor_float, normal_tensor_tensor
from .ones import ones  # noqa: F401
from .ones_like import ones_like
from .outer import outer
from .pad import pad
from .per_token_group_quant_fp8 import per_token_group_quant_fp8
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prod import prod, prod_dim
from .rand import rand
from .rand_like import rand_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .relu import relu, relu_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad3d import replication_pad3d
from .rsqrt import rsqrt, rsqrt_
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .select_scatter import select_scatter
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .softmax import softmax, softmax_backward
from .softplus import softplus
from .sort import sort, sort_stable
from .sqrt import sqrt, sqrt_
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .tril import tril, tril_, tril_out
from .triu import triu
from .uniform import uniform_
from .unique import _unique2, simple_unique_flat, sorted_indices_unique_flat
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .var_mean import var_mean
from .vector_norm import vector_norm
from .vstack import vstack
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "mean_dim",
    "mean",
    "zeros",
    "zero_",
    "scatter",
    "scatter_",
    "sort",
    "sort_stable",
    "cat",
    "addmm",
    "bmm",
    "bmm_out",
    "mm",
    "mv",
    "arange",
    "embedding",
    "multinomial",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "repeat_interleave_self_int",
    "argmax",
    "argmin",
    "exponential_",
    "gather",
    "gt",
    "gt_scalar",
    "index_select",
    "index",
    "isin",
    "max",
    "max_dim",
    "min",
    "min_dim",
    "sum",
    "sum_out",
    "sum_dim_out",
    "sum_dim",
    "full",
    "abs",
    "abs_",
    "add",
    "add_",
    "angle",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_not",
    "bitwise_not",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_xor_scalar",
    "bitwise_xor_scalar_",
    "bitwise_xor_scalar_tensor",
    "bitwise_xor_tensor",
    "bitwise_xor_tensor_",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "copy",
    "copy_",
    "cos",
    "cos_",
    "count_nonzero",
    "diag",
    "diag_embed",
    "true_divide",
    "true_divide_",
    "trunc_divide_",
    "trunc_divide",
    "floor_divide",
    "floor_divide_",
    "remainder",
    "remainder_",
    "elu",
    "equal",
    "eq_scalar",
    "eq",
    "erf",
    "erf_",
    "exp",
    "exp_",
    "exp_out",
    "exp2",
    "exp2_",
    "expm1",
    "expm1_",
    "expm1_out",
    "fill_scalar",
    "fill_scalar_",
    "fill_tensor",
    "fill_tensor_",
    "fill_scalar_out",
    "fill_tensor_out",
    "full_like",
    "flip",
    "ge",
    "ge_scalar",
    "gelu_backward",
    "gelu_",
    "gelu",
    "glu",
    "isclose",
    "allclose",
    "isfinite",
    "isinf",
    "isnan",
    "le_scalar",
    "le",
    "lerp_tensor_",
    "lerp_tensor",
    "lerp_scalar",
    "lerp_scalar_",
    "log_sigmoid",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt_scalar",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "mul_",
    "nan_to_num",
    "ne_scalar",
    "ne",
    "neg",
    "neg_",
    "nonzero_numpy",
    "normal_tensor_tensor",
    "normal_tensor_float",
    "normal_float_tensor",
    "per_token_group_quant_fp8",
    "polar",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_scalar",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "repeat",
    "rsqrt",
    "rsqrt_",
    "sigmoid_backward",
    "sigmoid_",
    "sigmoid",
    "silu_backward",
    "silu",
    "silu_",
    "sin",
    "sin_",
    "sub",
    "sub_",
    "tanh_backward",
    "tanh",
    "tanh_",
    "threshold_backward",
    "threshold",
    "trace",
    "tile",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "where_self_out",
    "where_self",
    "where_scalar_self",
    "where_scalar_other",
    "contiguous",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_right_shift",
    "bitwise_right_shift_",
    "outer",
    "diagonal_backward",
    "topk",
    "eye",
    "eye_m",
    "pad",
    "log_softmax",
    "count_nonzero",
    "linspace",
    "var_mean",
    "slice_backward",
    "slice_scatter",
    "select_scatter",
    "ones_like",
    "prod",
    "prod_dim",
    "zeros_like",
    "rand",
    "randn",
    "rand_like",
    "randn_like",
    "randperm",
    "normed_cumsum",
    "cumsum",
    "cumsum_out",
    "nonzero",
    "nonzero_numpy",
    "uniform_",
    "cummin",
    "simple_unique_flat",
    "_unique2",
    "sorted_indices_unique_flat",
    "dropout",
    "cummax",
    "_index_put_impl_",
    "index_put",
    "index_put_",
    "vstack",
    "all",
    "all_dim",
    "all_dims",
    "amax",
    "group_norm",
    "group_norm_backward",
    "layer_norm",
    "layer_norm_backward",
    "to_copy",
    "any",
    "any_dim",
    "any_dims",
    "amax",
    "nll_loss_forward",
    "nll_loss_backward",
    "vector_norm",
    "tril",
    "tril_",
    "tril_out",
    "triu",
    "_upsample_bicubic2d_aa",
    "softmax",
    "softmax_backward",
    "replication_pad3d",
    "softplus",
    "sqrt",
    "sqrt_",
    "addmv",
    "addmv_out",
    "cat_out",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "conj_physical",
    "cosh",
    "cosh_",
    "cosh_out",
    "linear",
    "scatter_add_",
]
