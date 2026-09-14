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

from ._unsafe_masked_index import _unsafe_masked_index
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
from .baddbmm import baddbmm
from .bincount import bincount
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
from .cauchy import cauchy, cauchy_
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .clamp_min import clamp_min, clamp_min_
from .clip import clip, clip_
from .concatenate import concatenate
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
from .embedding import embedding, embedding_backward
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1 import expm1, expm1_, expm1_out
from .exponential_ import exponential_
from .eye import eye
from .eye_m import eye_m
from .feature_dropout import feature_dropout, feature_dropout_
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
from .gather import gather, gather_backward
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hstack import hstack
from .index import index
from .index_add import index_add, index_add_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .kron import kron
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
from .max_pool2d_with_indices import max_pool2d_backward, max_pool2d_with_indices
from .max_pool3d_with_indices import max_pool3d_backward, max_pool3d_with_indices
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, router_gemm
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .nan_to_num import nan_to_num
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .nllloss import (
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
)
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .one_hot import one_hot
from .ones import ones  # noqa: F401
from .ones_like import ones_like
from .outer import outer
from .pad import pad
from .per_token_group_quant_fp8 import per_token_group_quant_fp8
from .poisson import poisson
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
from .randint_like import randint_like
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
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
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
from .unique_consecutive import unique_consecutive
from .unique_dim import unique_dim
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
    "_index_put_impl_",
    "_unique2",
    "_unsafe_masked_index",
    "_upsample_bicubic2d_aa",
    "abs",
    "abs_",
    "add",
    "add_",
    "addmm",
    "addmv",
    "addmv_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "amax",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "argmax",
    "argmin",
    "baddbmm",
    "bincount",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_not",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_right_shift",
    "bitwise_right_shift_",
    "bitwise_xor_scalar",
    "bitwise_xor_scalar_",
    "bitwise_xor_scalar_tensor",
    "bitwise_xor_tensor",
    "bitwise_xor_tensor_",
    "bmm",
    "bmm_out",
    "cat",
    "cat_out",
    "cauchy",
    "cauchy_",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "concatenate",
    "conj_physical",
    "contiguous",
    "copy",
    "copy_",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cummax",
    "cummin",
    "cumsum",
    "cumsum_out",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "dropout",
    "elu",
    "embedding",
    "embedding_backward",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erf_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "exp_out",
    "expm1",
    "expm1_",
    "expm1_out",
    "exponential_",
    "eye",
    "eye_m",
    "feature_dropout",
    "feature_dropout_",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "flip",
    "floor_divide",
    "floor_divide_",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "glu",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "hstack",
    "index",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "kron",
    "layer_norm",
    "layer_norm_backward",
    "le",
    "le_scalar",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "linear",
    "linspace",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log_sigmoid",
    "log_softmax",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt",
    "lt_scalar",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "maximum",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mul",
    "mul_",
    "multinomial",
    "mv",
    "nan_to_num",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nonzero",
    "nonzero_numpy",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "one_hot",
    "ones_like",
    "outer",
    "pad",
    "per_token_group_quant_fp8",
    "poisson",
    "polar",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prod",
    "prod_dim",
    "rand",
    "rand_like",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "remainder",
    "remainder_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad3d",
    "router_gemm",
    "rsqrt",
    "rsqrt_",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "select_scatter",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "silu",
    "silu_",
    "silu_backward",
    "simple_unique_flat",
    "sin",
    "sin_",
    "slice_backward",
    "slice_scatter",
    "softmax",
    "softmax_backward",
    "softplus",
    "sort",
    "sort_stable",
    "sorted_indices_unique_flat",
    "sqrt",
    "sqrt_",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "tanh",
    "tanh_",
    "tanh_backward",
    "threshold",
    "threshold_backward",
    "tile",
    "to_copy",
    "topk",
    "trace",
    "tril",
    "tril_",
    "tril_out",
    "triu",
    "true_divide",
    "true_divide_",
    "trunc_divide",
    "trunc_divide_",
    "uniform_",
    "unique_consecutive",
    "unique_dim",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "var_mean",
    "vector_norm",
    "vstack",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "zero_",
    "zeros",
    "zeros_like",
]
