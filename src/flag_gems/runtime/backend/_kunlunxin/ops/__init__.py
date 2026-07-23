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

from ._euclidean_dist import _euclidean_dist
from ._functional_sym_constrain_range import _functional_sym_constrain_range
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._is_all_true import _is_all_true
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_, addcdiv_out
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_out
from .addmv import addmv, addmv_out
from .addr import addr
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .amax import amax
from .amin import amin, amin_
from .aminmax import aminmax
from .angle import angle
from .any import any, any_dim, any_dims
from .apply_repetition_penalties import apply_repetition_penalties
from .arange import arange, arange_start
from .arccos import arccos, arccos_
from .arcsin import arcsin, arcsin_, arcsin_out
from .arctan import arctan, arctan_
from .argmax import argmax
from .argmin import argmin
from .as_strided_copy import as_strided_copy, as_strided_copy_out
from .asin import asin, asin_
from .atan import atan, atan_
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .baddbmm import baddbmm
from .batch_norm import batch_norm, batch_norm_backward
from .bernoulli_ import bernoulli_
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import bitwise_left_shift
from .bitwise_not import bitwise_not, bitwise_not_
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .bitwise_right_shift import bitwise_right_shift
from .bmm import bmm, bmm_out
from .broadcast_to import broadcast_to
from .cat import cat, cat_out
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import (
    clamp,
    clamp_,
    clamp_max,
    clamp_max_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
)
from .clip import clip, clip_
from .concatenate import concatenate
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .cos import cos, cos_
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .deg2rad import deg2rad, deg2rad_, deg2rad_out
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .digamma_ import digamma_
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .dot import dot
from .dropout import dropout, dropout_backward
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward
from .eq import eq, eq_scalar
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
from .floor import floor, floor_, floor_out
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .ge import ge, ge_scalar, greater_equal_
from .gelu import gelu, gelu_, gelu_backward
from .get_scheduler_metadata import get_scheduler_metadata
from .glu import glu, glu_backward
from .greater import greater, greater_out, greater_scalar, greater_scalar_out
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hadamard_transform import hadamard_transform
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hstack import hstack
from .index import index
from .index_add import index_add, index_add_
from .index_put import index_put, index_put_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .kron import kron
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .less_equal import less_equal, less_equal_scalar
from .lift_fresh_copy import lift_fresh_copy
from .linspace import linspace
from .log import log
from .log1p import log1p, log1p_
from .log_sigmoid import log_sigmoid
from .log_softmax import log_softmax, log_softmax_backward
from .logaddexp2 import logaddexp2, logaddexp2_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not, logical_not_
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor, logical_xor_
from .logspace import logspace
from .logsumexp import logsumexp
from .lt import lt, lt_, lt_scalar, lt_scalar_
from .masked_fill import masked_fill, masked_fill_
from .masked_scatter import masked_scatter, masked_scatter_
from .masked_select import masked_select
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max import max, max_dim
from .max_pool2d_with_indices import max_pool2d_backward, max_pool2d_with_indices
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .multiply_ import multiply_
from .mv import mv, mv_cluster
from .nan_to_num import nan_to_num
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .narrow_copy import narrow_copy
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .negative import negative
from .new_full import new_full
from .new_ones import new_ones
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
from .not_equal import not_equal, not_equal_scalar
from .ones import ones
from .ones_like import ones_like
from .pad import constant_pad_nd, pad
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .permute_copy import permute_copy
from .pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu import prelu
from .prod import prod, prod_dim
from .quantile import quantile
from .rad2deg import rad2deg, rad2deg_
from .rand import rand
from .rand_like import rand_like
from .randint_like import randint_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad1d import reflection_pad1d, reflection_pad1d_out
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from .relu import relu, relu_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resize import resize, resize_
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .rnn_relu import rnn_relu
from .rot90 import rot90
from .round import round, round_, round_out
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub, rsub_scalar, rsub_tensor
from .safe_softmax import _safe_softmax
from .scaled_softmax import scaled_softmax_backward, scaled_softmax_forward
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .select_scatter import select_scatter
from .selu import selu, selu_
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .signbit import signbit, signbit_out
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .soft_margin_loss_backward import soft_margin_loss_backward
from .softmax import softmax, softmax_backward
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_log_softmax import special_log_softmax
from .special_logsumexp import special_logsumexp
from .sqrt import sqrt, sqrt_
from .stack import stack
from .std import std
from .sub import sub, sub_, subtract_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .tril import tril, tril_, tril_out
from .triu import triu, triu_
from .trunc import trunc, trunc_
from .uniform import uniform_
from .unique import _unique2
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_trilinear3d import upsample_trilinear3d
from .var_mean import var_mean
from .vdot import vdot
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack import vstack
from .weightnorm import weight_norm_interface, weight_norm_interface_backward
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .xlogy import (
    xlogy,
    xlogy_out,
    xlogy_scalar_tensor,
    xlogy_scalar_tensor_out,
    xlogy_tensor_scalar,
    xlogy_tensor_scalar_out,
)
from .zero import zero, zero_, zero_out
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_euclidean_dist",
    "_is_all_true",
    "_thnn_fused_lstm_cell_backward_impl",
    "_conv_depthwise2d",
    "_safe_softmax",
    "digamma_",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "soft_margin_loss_backward",
    "special_log_softmax",
    "special_logsumexp",
    "softshrink",
    "softshrink_out",
    "_unique2",
    "_upsample_bicubic2d_aa",
    "apply_repetition_penalties",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcdiv_out",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_out",
    "addmv",
    "addmv_out",
    "addr",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "amax",
    "amin",
    "amin_",
    "aminmax",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arccos",
    "arccos_",
    "arcsin",
    "arcsin_",
    "arcsin_out",
    "arctan",
    "arctan_",
    "argmax",
    "argmin",
    "as_strided_copy",
    "as_strided_copy_out",
    "asin",
    "asin_",
    "atan",
    "atan_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "baddbmm",
    "batch_norm",
    "batch_norm_backward",
    "bernoulli_",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_right_shift",
    "bmm",
    "bmm_out",
    "broadcast_to",
    "cat",
    "cat_out",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_tensor",
    "clamp_tensor_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "concatenate",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "copy",
    "copy_",
    "copysign",
    "copysign_out",
    "cos",
    "cos_",
    "count_nonzero",
    "cummax",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumsum",
    "cumsum_out",
    "deg2rad",
    "deg2rad_",
    "deg2rad_out",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "div_mode",
    "div_mode_",
    "dot",
    "dropout",
    "dropout_backward",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
    "eq",
    "eq_scalar",
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
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flip",
    "floor",
    "floor_",
    "floor_out",
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
    "get_scheduler_metadata",
    "glu",
    "glu_backward",
    "greater",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "greater_equal_",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "hstack",
    "hadamard_transform",
    "hardsigmoid",
    "hardsigmoid_out",
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
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "le",
    "le_scalar",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "less_equal",
    "less_equal_scalar",
    "lift_fresh_copy",
    "linspace",
    "log",
    "log1p",
    "log1p_",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_backward",
    "logaddexp2",
    "logaddexp2_out",
    "logsumexp",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "logspace",
    "lt",
    "lt_",
    "lt_scalar",
    "lt_scalar_",
    "matmul_bf16",
    "matmul_int8",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_select",
    "max",
    "max_dim",
    "maximum",
    "max_pool2d_with_indices",
    "max_pool2d_backward",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mm_out",
    "mse_loss",
    "mul",
    "mul_",
    "multinomial",
    "multiply_",
    "mv",
    "mv_cluster",
    "nan_to_num",
    "narrow_copy",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "new_full",
    "new_ones",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
    "not_equal",
    "not_equal_scalar",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nonzero",
    "nonzero_numpy",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normal_",
    "normed_cumsum",
    "ones",
    "ones_like",
    "pad",
    "per_token_group_quant_fp8",
    "pixel_unshuffle",
    "pixel_unshuffle_out",
    "permute_copy",
    "polar",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "prod",
    "prod_dim",
    "quantile",
    "rad2deg",
    "rad2deg_",
    "rand",
    "rand_like",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "reflection_pad1d",
    "reflection_pad1d_out",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "relu",
    "relu_",
    "remainder",
    "remainder_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "resize",
    "resize_",
    "resolve_conj",
    "resolve_neg",
    "rot90",
    "round",
    "round_",
    "round_out",
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "rnn_relu",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "rsub_scalar",
    "rsub_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
    "scatter",
    "scatter_",
    "scatter_add_",
    "select_scatter",
    "selu",
    "selu_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "signbit",
    "signbit_out",
    "sgn_",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "slice_backward",
    "slice_scatter",
    "softmax",
    "softmax_backward",
    "softplus",
    "sort",
    "sort_stable",
    "sqrt",
    "sqrt_",
    "stack",
    "std",
    "sub",
    "sub_",
    "subtract_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "ScaleDotProductAttention",
    "SUPPORTED_FP8_DTYPE",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "threshold",
    "threshold_",
    "threshold_backward",
    "tile",
    "to_copy",
    "topk",
    "trace",
    "tril",
    "tril_",
    "tril_out",
    "triu",
    "triu_",
    "true_divide",
    "true_divide_out",
    "trunc",
    "trunc_",
    "true_divide_",
    "uniform_",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_trilinear3d",
    "var_mean",
    "vdot",
    "vector_norm",
    "view_copy",
    "vstack",
    "weight_norm_interface",
    "weight_norm_interface_backward",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "xlogy",
    "xlogy_out",
    "xlogy_scalar_tensor",
    "xlogy_scalar_tensor_out",
    "xlogy_tensor_scalar",
    "xlogy_tensor_scalar_out",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]
