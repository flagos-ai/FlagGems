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

from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._safe_softmax import _safe_softmax  # noqa: F401
from ._upsample_nearest_exact1d import (
    _upsample_nearest_exact1d,
    _upsample_nearest_exact1d_out,
    _upsample_nearest_exact1d_vec,
)
from .abs import abs, abs_
from .absolute import absolute
from .add import add, add_
from .addcmul import addcmul
from .addmm import addmm, addmm_out
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arctanh_ import arctanh_
from .argmax import argmax
from .argmin import argmin
from .atan import atan, atan_
from .attention import (
    ScaleDotProductAttention,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import bitwise_left_shift, bitwise_left_shift_
from .bitwise_not import bitwise_not, bitwise_not_
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
from .ceil import ceil, ceil_
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .concatenate import concatenate
from .contiguous import contiguous
from .copy import copy, copy_
from .cos import cos, cos_
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .digamma_ import digamma_
from .div import (
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    trunc_divide,
    trunc_divide_,
)
from .dropout import dropout
from .elu import elu, elu_
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .exp import exp, exp_
from .exp2 import exp2, exp2_
from .exponential_ import exponential_
from .eye import eye
from .eye_m import eye_m
from .feature_dropout import feature_dropout, feature_dropout_
from .fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
from .flip import flip
from .floor_ import floor_
from .fmin import fmin, fmin_out
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu
from .grid_sample import grid_sample
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_, i0_out
from .index import index
from .index_add import index_add, index_add_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linear import linear
from .linspace import linspace
from .log import log
from .log_sigmoid import log_sigmoid
from .log_softmax import log_softmax
from .logical_and import logical_and
from .logical_not import logical_not
from .logical_or import logical_or
from .logical_xor import logical_xor
from .logit import logit, logit_, logit_out
from .logspace import logspace
from .lt import lt, lt_scalar
from .masked_fill import masked_fill, masked_fill_
from .masked_select import masked_select
from .max import max, max_dim
from .max_pool2d_with_indices import max_pool2d_backward, max_pool2d_with_indices
from .maximum import maximum
from .mean import mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .nan_to_num import nan_to_num
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .pad import pad
from .poisson import poisson
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu import prelu
from .quantile import quantile
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .relu import relu, relu_
from .relu6 import relu6
from .remainder import remainder, remainder_
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad1d import replication_pad1d, replication_pad1d_out
from .replication_pad3d import replication_pad3d
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .select_scatter import select_scatter
from .selu import selu
from .selu_ import selu_
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinh_ import sinh_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .smooth_l1_loss import smooth_l1_loss, smooth_l1_loss_backward, smooth_l1_loss_out
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .softmax import softmax_backward
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .sqrt import sqrt, sqrt_
from .stack import stack
from .sub import sub, sub_
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_dtype
from .topk import topk
from .tril import tril, tril_, tril_out
from .uniform import uniform_
from .unique import (
    _unique2,
    simple_unique_flat,
    sorted_indices_unique_flat,
    sorted_quick_unique_flat,
)
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vector_norm import vector_norm
from .vstack import vstack
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .zero import zero, zero_, zero_out
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "one_hot",
    "argmin",
    "avg_pool2d",
    "avg_pool2d_backward",
    "count_nonzero",
    "mean_dim",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "scatter",
    "scatter_",
    "topk",
    "topk_backward",
    "ScaleDotProductAttention",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scatter_add_",
    "sort",
    "sort_stable",
    "cat",
    "concatenate",
    "alias_copy",
    "alias_copy_out",
    "mm",
    "true_divide",
    "true_divide_",
    "trunc_divide_",
    "trunc_divide",
    "floor_divide",
    "floor_divide_",
    "remainder",
    "remainder_",
    "add",
    "add_",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "equal",
    "eq_scalar",
    "eq",
    "ge",
    "ge_scalar",
    "gt",
    "gt_scalar",
    "le_scalar",
    "le",
    "lt_scalar",
    "lt",
    "mul",
    "mul_",
    "ne_scalar",
    "ne",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_scalar",
    "maximum",
    "min",
    "min_dim",
    "minimum",
    "sub",
    "sub_",
    "where_self_out",
    "where_self",
    "where_scalar_self",
    "where_scalar_other",
    "isclose",
    "allclose",
    "logical_and",
    "logical_or",
    "logical_xor",
    "threshold_backward",
    "threshold",
    "polar",
    "lerp_tensor_",
    "lerp_tensor",
    "lerp_scalar",
    "lerp_scalar_",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "fill_scalar",
    "fill_scalar_",
    "fill_tensor",
    "fill_tensor_",
    "pad",
    "eye",
    "normed_cumsum",
    "cumsum",
    "cumsum_out",
    "multinomial",
    "isfinite",
    "bitwise_xor_scalar",
    "bitwise_xor_scalar_",
    "bitwise_xor_scalar_tensor",
    "bitwise_xor_tensor",
    "bitwise_xor_tensor_",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_right_shift",
    "bitwise_right_shift_",
    "log_softmax",
    "argmax",
    "sorted_quick_unique_flat",
    "sorted_indices_unique_flat",
    "simple_unique_flat",
    "_unique2",
    "upsample_nearest2d",
    "upsample_nearest1d",
    "max",
    "max_dim",
    "rms_norm",
    "cummin",
    "index_select",
    "vector_norm",
    "cummax",
    "copy",
    "copy_",
    "contiguous",
    "eye_m",
    "dropout",
    "feature_dropout",
    "feature_dropout_",
    "_functional_sym_constrain_range_for_size",
    "index",
    "index_add",
    "index_add_",
    "bmm",
    "bmm_out",
    "diag_embed",
    "diagonal_backward",
    "digamma_",
    "flip",
    "floor_",
    "fmin",
    "fmin_out",
    "gcd",
    "gcd_out",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
    "histc",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "poisson",
    "abs",
    "abs_",
    "absolute",
    "addcmul",
    "addmm",
    "addmm_out",
    "angle",
    "bitwise_not",
    "bitwise_not_",
    "cos",
    "cos_",
    "diag_embed",
    "elu",
    "elu_",
    "erf",
    "erf_",
    "exp",
    "exp_",
    "exp2",
    "exp2_",
    "full",
    "gelu",
    "gelu_",
    "gelu_backward",
    "glu",
    "isin",
    "isinf",
    "isnan",
    "log",
    "log_sigmoid",
    "logical_not",
    "mse_loss",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "nan_to_num",
    "neg",
    "neg_",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "relu6",
    "rrelu_with_noise_backward",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "rsqrt",
    "rsqrt_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinh_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "to_dtype",
    "full_like",
    "resolve_neg",
    "linspace",
    "arange",
    "arange_start",
    "arctanh_",
    "slice_scatter",
    "select_scatter",
    "ones",
    "ones_like",
    "zeros_like",
    "grid_sample",
    "group_norm",
    "group_norm_backward",
    "_upsample_bicubic2d_aa",
    "gather",
    "gather_backward",
    "randn_like",
    "randn",
    "exponential_",
    "logspace",
    "replication_pad3d",
    "replication_pad1d",
    "replication_pad1d_out",
    "max_pool2d_with_indices",
    "max_pool2d_backward",
    "upsample_linear1d",
    "var",
    "var_dim",
    "var_correction",
    "var_mean",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "softmax_backward",
    "ceil",
    "ceil_",
    "sqrt",
    "sqrt_",
    "celu",
    "celu_",
    "tile",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "softplus",
    "softshrink",
    "softshrink_out",
    "atan",
    "atan_",
    "hstack",
    "vstack",
    "uniform_",
    "all",
    "all_dim",
    "all_dims",
    "any",
    "any_dim",
    "any_dims",
    "tril",
    "tril_",
    "tril_out",
    "selu",
    "selu_",
    "hypot",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
    "round",
    "round_",
    "round_out",
    "sgn_",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "stack",
    "logit",
    "logit_",
    "logit_out",
    "randperm",
    "slice_backward",
    "quantile",
    "_upsample_nearest_exact1d",
    "_upsample_nearest_exact1d_out",
    "_upsample_nearest_exact1d_vec",
    "_saft_softmax",
    "layer_norm",
    "layer_norm_backward",
    "linear",
    "cat_out",
    "prelu",
]
