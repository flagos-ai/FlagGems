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

from ._aminmax import _aminmax, _aminmax_out
from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from ._functional_sym_constrain_range import (
    _functional_sym_constrain_range,
)
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._native_batch_norm_legit_functional import (
    _native_batch_norm_legit_functional,
)
from .linalg_cholesky import linalg_cholesky
from .cholesky import cholesky
from ._euclidean_dist import _euclidean_dist
from ._foreach_zero import _foreach_zero, _foreach_zero_
from ._foreach_log2_ import _foreach_log2_
from .conv3d import conv3d
from .conv2d__opt import conv2d
from .conv1d__opt import conv1d
from .atan2 import atan2, atan2_out
from .lgamma_ import lgamma, lgamma_
from ._safe_softmax import _safe_softmax
from ._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention
from ._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from ._upsample_nearest_exact1d import (
    _upsample_nearest_exact1d,
    _upsample_nearest_exact1d_out,
    _upsample_nearest_exact1d_vec,
)
from ._upsample_nearest_exact2d import (
    _upsample_nearest_exact2d,
    _upsample_nearest_exact2d_out,
    _upsample_nearest_exact2d_vec,
)
from ._upsample_nearest_exact3d import (
    _upsample_nearest_exact3d,
    _upsample_nearest_exact3d_out,
    _upsample_nearest_exact3d_vec,
)
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .acosh import acosh, acosh_
from .add import add, add_
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_out
from .addmm_dtype import addmm_dtype, addmm_dtype_out
from .addmv import addmv, addmv_out
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .rsub import rsub_tensor
from .rsub__opt import rsub_scalar
from .not_equal import not_equal
from .not_equal_scalar__opt import not_equal_scalar
from .greater_equal import greater_equal, greater_equal_scalar, greater_equal_
from .trunc_ import trunc, trunc_
from .copysign import copysign, copysign_out
from .greater import greater, greater_out
from .greater_scalar__opt import greater_scalar
from .greater_scalar_out__opt import greater_scalar_out
from .copysign_ import copysign_
from .logical_xor_ import logical_xor_
from .prod import prod, prod_dim
from .rad2deg import rad2deg, rad2deg_
from .clamp_max import clamp_max
from .clamp_max__opt import clamp_max_
from .gt import gt_tensor_
from .gt_scalar__opt import gt_scalar_

from .logical_and import logical_and_
from .ceil import ceil_out
from .amin import amin, amin_
from .aminmax import aminmax
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arccos import arccos
from .arccosh_ import arccosh_
from .arcsin import arcsin, arcsin_, arcsin_out
from .arcsinh import arcsinh, arcsinh_, arcsinh_out
from .arctan_ import arctan, arctan_
from .arctanh_ import arctanh_
from .argmax__opt import argmax
from .argmin import argmin
from .argsort import argsort
from .asin import asin_
from .asinh import asinh, asinh_, asinh_out
from .atan import atan, atan_
from .atanh import atanh, atanh_
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .baddbmm import baddbmm, baddbmm_out
from .binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits,
    binary_cross_entropy_with_logits_,
)
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
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .bitwise_or__opt import bitwise_or_scalar_
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
from .ceil import ceil, ceil_
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .concat__opt import concat
from .concatenate import concatenate
from .constant_pad_nd import constant_pad_nd
from .contiguous import contiguous
from ._is_all_true import _is_all_true
from .copy import copy, copy_
from .cos import cos, cos_
from .cosh_out__opt import cosh_out
from .cosh__opt import cosh
from .cosh___opt import cosh_
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cummaxmin_backward__opt import cummaxmin_backward
from .bucketize__opt import bucketize
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .deg2rad import deg2rad, deg2rad_, deg2rad_out
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .digamma import digamma, digamma_out
from .digamma_ import digamma_
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    trunc_divide,
    trunc_divide_,
)
from .dropout import dropout, dropout_backward, native_dropout_backward
from .dot import dot
from .elu import elu, elu_
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .erfinv_ import erfinv, erfinv_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .exponential import exponential
from .exponential_ import exponential_
from .square_out import square_out
from .square__opt import square_

from .square import square
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
from .flash_attention_backward_gcu400__opt import (
    scaled_dot_product_efficient_attention_backward,
)
from .frac_ import frac, frac_
from .flip import flip
from .floor_ import floor_
from .floor__opt import floor
from .clip__opt import clip
from .clip___opt import clip_
from .expm1_out__opt import expm1_out
from .expm1__opt import expm1
from .expm1___opt import expm1_
from .floor_out__opt import floor_out

from .fmax import fmax, fmax_out
from .fmin import fmin, fmin_out
from .fmod import fmod_, fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .fractional_max_pool2d import fractional_max_pool2d, fractional_max_pool2d_backward
from .fractional_max_pool3d_backward import fractional_max_pool3d_backward
from .full import full
from .full_like__opt import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu, glu_backward
from .grid_sample import grid_sample
from .group_mm import group_mm
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
from .repeat import repeat
from .top_k_per_row_decode import top_k_per_row_decode
from .per_token_group_quant_fp8 import per_token_group_quant_fp8
from .rwkv_mm_sparsity import rwkv_mm_sparsity
from .linalg_ldl_solve import linalg_ldl_solve
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_, i0_out
from .index import index
from .index_add import index_add, index_add_
from .index_copy import index_copy, index_copy_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .isneginf import isneginf, isneginf_out
from .kron import kron
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linalg_ldl_factor import ldl_factor as linalg_ldl_factor

# Also bind the bare name: the aten patcher (op_registrar._resolve_live_override)
# looks the override up as flag_gems.<generic function __name__>, which for
# aten::linalg_ldl_factor is "ldl_factor".  Without this alias, use_gems() would
# keep registering the generic flag_gems.ops implementation instead of this one.
from .linalg_ldl_factor import ldl_factor  # noqa: F401
from .linalg_polar import linalg_polar
from .linear import linear
from .linspace import linspace
from .log import log
from .log1p import log1p_, log1p_out
from .log1p__opt import log1p
from .log2 import log2, log2_
from .log10 import log10, log10_, log10_out
from .log_sigmoid import log_sigmoid
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .logaddexp import logaddexp, logaddexp_out
from .logical_and import logical_and
from .logical_not import logical_not
from .less_equal import less_equal
from .less_equal_scalar__opt import less_equal_scalar
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logit import logit, logit_, logit_out
from .logspace import logspace
from .lt import lt, lt_scalar
from .margin_ranking_loss import margin_ranking_loss
from .masked_fill import masked_fill, masked_fill_
from .masked_select import masked_select
from .max import max, max_dim
from .max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
    max_pool2d_with_indices_backward,
)
from .maximum import maximum
from .mean import mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out, router_gemm
from .mse_loss import mse_loss
from .mul import mul, mul_
from .multinomial import multinomial
from .multiply_ import multiply_
from .mv import mv
from .nan_to_num import nan_to_num
from .native_layer_norm import native_layer_norm
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .negative import negative, negative_, negative_out
from .nllloss import (
    nll_loss_backward,
    nll_loss_forward,
    nll_loss2d_backward,
    nll_loss2d_forward,
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
from .ones import ones
from .ones_like import ones_like
from .pad__opt import pad

from .poisson__opt import poisson
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu__opt import prelu
from .quantile import quantile
from .randint import randint
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
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
from .rrelu_with_noise_backward__opt import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add, scatter_add_
from .scatter_reduce__opt import scatter_reduce
from .segment_reduce import (
    _segment_reduce_backward,
    _segment_reduce_backward_out,
    segment_reduce,
    segment_reduce_out,
)
from .select_scatter import select_scatter
from .selu import selu
from .selu_ import selu_
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .silu import silu, silu_, silu_backward
from .signbit__opt import signbit
from .sin import sin, sin_
from .sinh_ import sinh_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .smooth_l1_loss import smooth_l1_loss, smooth_l1_loss_backward, smooth_l1_loss_out
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .softmax import softmax, softmax_backward
from .softmax_out__opt import softmax_out
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .special_bessel_j1 import special_bessel_j1
from .special_modified_bessel_k0 import (
    special_modified_bessel_k0,
    special_modified_bessel_k0_out,
)
from .sqrt import sqrt, sqrt_
from .stack import stack
from .std import std
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .svd import svd
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
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
from .upsample_bicubic2d import upsample_bicubic2d
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack__opt import vstack
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .new_zeros import new_zeros, new_zeros_out
from .zero import zero, zero_, zero_out
from .zeros import zeros
from .zeros_like import zeros_like

from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d import adaptive_max_pool3d

__all__ = [
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "_aminmax",
    "_aminmax_out",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_is_all_true",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_index_put_impl_",
    "_native_batch_norm_legit_functional",
    "_safe_softmax",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_fused_attention_overrideable",
    "_unique2",
    "_upsample_bicubic2d_aa",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "linalg_polar",
    "svd",
    "_upsample_nearest_exact1d",
    "_upsample_nearest_exact1d_out",
    "_upsample_nearest_exact1d_vec",
    "_upsample_nearest_exact2d",
    "_upsample_nearest_exact2d_out",
    "_upsample_nearest_exact2d_vec",
    "_upsample_nearest_exact3d",
    "_upsample_nearest_exact3d_out",
    "_upsample_nearest_exact3d_vec",
    "abs",
    "abs_",
    "absolute",
    "add",
    "add_",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_out",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmv",
    "addmv_out",
    "alias_copy",
    "alias_copy_out",
    "acos",
    "acosh",
    "acosh_",
    "all",
    "arcsin",
    "arcsin_",
    "arcsin_out",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "rsub_tensor",
    "rsub_scalar",
    "not_equal",
    "not_equal_scalar",
    "greater_equal",
    "greater_equal_scalar",
    "greater_equal_",
    "trunc",
    "trunc_",
    "copysign",
    "copysign_out",
    "copysign_scalar",
    "gt_tensor",
    "gt_tensor_",
    "greater",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "copysign_",
    "logical_xor_",
    "logical_and_",
    "prod",
    "prod_dim",
    "rad2deg",
    "rad2deg_",
    "clamp_max",
    "clamp_max_",
    "ceil_out",
    "amin",
    "amin_",
    "aminmax",
    "all_dim",
    "all_dims",
    "allclose",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arccos",
    "arccosh_",
    "atanh",
    "atanh_",
    "isneginf",
    "isneginf_out",
    "asin_",
    "asinh",
    "asinh_",
    "asinh_out",
    "arctan",
    "arctan_",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "atan",
    "atan_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "baddbmm",
    "baddbmm_out",
    "binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits_",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_not",
    "bitwise_not_",
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
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "concat",
    "concatenate",
    "constant_pad_nd",
    "index_copy",
    "index_copy_",
    "contiguous",
    "copy",
    "copy_",
    "cos",
    "cos_",
    "cosh_out",
    "cosh_",
    "cosh",
    "count_nonzero",
    "cummax",
    "cummin",
    "cummaxmin_backward",
    "bucketize",
    "cumsum",
    "cumsum_out",
    "deg2rad",
    "deg2rad_",
    "deg2rad_out",
    "diag_embed",
    "diagonal_backward",
    "digamma_",
    "digamma",
    "digamma_out",
    "digamma_",
    "div_mode",
    "div_mode_",
    "dropout",
    "dot",
    "dropout_backward",
    "native_dropout_backward",
    "elu",
    "elu_",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erfinv",
    "erfinv_",
    "erf_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "exp_out",
    "exponential",
    "exponential_",
    "square_out",
    "square",
    "square_",
    "eye",
    "eye_m",
    "fmax",
    "fmax_out",
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
    "fmod_",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "frac",
    "frac_",
    "flip",
    "floor",
    "clip",
    "clip_",
    "expm1",
    "expm1_",
    "expm1_out",
    "floor_",
    "floor_out",
    "floor_divide",
    "floor_divide_",
    "fmin",
    "fmin_out",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "fractional_max_pool3d_backward",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "gcd",
    "gcd_out",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "glu",
    "glu_backward",
    "grid_sample",
    "group_mm",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar_",
    "gt_scalar",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
    "repeat",
    "top_k_per_row_decode",
    "per_token_group_quant_fp8",
    "rwkv_mm_sparsity",
    "linalg_ldl_solve",
    "histc",
    "hstack",
    "hypot",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
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
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linear",
    "linspace",
    "linalg_cholesky",
    "cholesky",
    "_euclidean_dist",
    "_foreach_log2_",
    "_foreach_zero",
    "_foreach_zero_",
    "conv3d",
    "conv2d",
    "conv1d",
    "atan2",
    "atan2_out",
    "lgamma",
    "lgamma_",
    "log",
    "log1p",
    "log1p_",
    "log1p_out",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_out",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "logaddexp",
    "logaddexp_out",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_or_",
    "less_equal",
    "less_equal_scalar",
    "log2",
    "log2_",
    "log10",
    "log10_",
    "log10_out",
    "linalg_ldl_factor",
    "ldl_factor",
    "logical_xor",
    "logit",
    "logit_",
    "logit_out",
    "logspace",
    "lt",
    "lt_scalar",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "maximum",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "router_gemm",
    "mse_loss",
    "mul",
    "mul_",
    "multinomial",
    "multiply_",
    "mv",
    "nan_to_num",
    "native_layer_norm",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "negative_out",
    "nonzero",
    "nonzero_numpy",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "one_hot",
    "ones",
    "ones_like",
    "pad",
    "poisson",
    "polar",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "quantile",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad1d",
    "replication_pad1d_out",
    "replication_pad3d",
    "resolve_neg",
    "rms_norm",
    "round",
    "round_",
    "round_out",
    "rrelu_with_noise_backward",
    "rsqrt",
    "rsqrt_",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "segment_reduce",
    "segment_reduce_out",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_efficient_attention_backward",
    "ScaleDotProductAttention",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "select_scatter",
    "selu",
    "selu_",
    "sgn_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "silu",
    "silu_",
    "silu_backward",
    "signbit",
    "simple_unique_flat",
    "sin",
    "sin_",
    "sinh_",
    "slice_backward",
    "slice_scatter",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "softmax",
    "softmax_backward",
    "softmax_out",
    "softplus",
    "softshrink",
    "softshrink_out",
    "sort",
    "sort_stable",
    "sorted_indices_unique_flat",
    "sorted_quick_unique_flat",
    "special_chebyshev_polynomial_u",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_bessel_j1",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "sqrt",
    "sqrt_",
    "stack",
    "std",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "threshold",
    "threshold_backward",
    "tile",
    "to_dtype",
    "topk",
    "tril",
    "tril_",
    "tril_out",
    "true_divide",
    "true_divide_",
    "trunc_divide",
    "trunc_divide_",
    "uniform_",
    "upsample_bicubic2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "view_copy",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vector_norm",
    "vstack",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "new_zeros",
    "new_zeros_out",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]
