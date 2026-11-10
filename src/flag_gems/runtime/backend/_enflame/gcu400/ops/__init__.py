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
from ._cummin_helper import _cummin_helper
from ._euclidean_dist import _euclidean_dist
from ._foreach_add__List import _foreach_add__List
from ._foreach_addcdiv__Scalar import _foreach_addcdiv__Scalar
from ._foreach_ceil import _foreach_ceil, _foreach_ceil_
from ._foreach_clamp_min import _foreach_clamp_min__List
from ._foreach_cos_ import _foreach_cos_
from ._foreach_div import _foreach_div__ScalarList
from ._foreach_exp import _foreach_exp, _foreach_exp_
from ._foreach_lerp__List import _foreach_lerp__List
from ._foreach_log2 import _foreach_log2
from ._foreach_log2_ import _foreach_log2_
from ._foreach_log10 import _foreach_log10
from ._foreach_maximum_List import _foreach_maximum_List
from ._foreach_minimum import (
    _foreach_minimum__Scalar,
    _foreach_minimum__ScalarList,
    _foreach_minimum_Scalar,
    _foreach_minimum_ScalarList,
)
from ._foreach_mul import (
    _foreach_mul__List,
    _foreach_mul__Scalar,
    _foreach_mul__ScalarList,
    _foreach_mul__Tensor,
    _foreach_mul_List,
    _foreach_mul_Scalar,
    _foreach_mul_ScalarList,
    _foreach_mul_Tensor,
)
from ._foreach_neg import _foreach_neg
from ._foreach_pow import _foreach_pow__List
from ._foreach_sign import _foreach_sign, _foreach_sign_
from ._foreach_sin import _foreach_sin
from ._foreach_sqrt import _foreach_sqrt
from ._foreach_sqrt_ import _foreach_sqrt_
from ._foreach_sub import (
    _foreach_sub__Scalar,
    _foreach_sub__ScalarList,
    _foreach_sub_Scalar,
    _foreach_sub_ScalarList,
)
from ._foreach_tan import _foreach_tan
from ._foreach_zero import _foreach_zero, _foreach_zero_
from ._functional_sym_constrain_range import _functional_sym_constrain_range
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._is_all_true import _is_all_true
from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from ._safe_softmax import _safe_softmax
from ._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention
from ._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
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
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d import adaptive_max_pool3d
from .add import add, add_
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_out
from .addmm_dtype import addmm_dtype, addmm_dtype_out
from .addmv import addmv, addmv_out
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
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
from .atan2 import atan2, atan2_out
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
from .batch_norm_backward_reduce import batch_norm_backward_reduce
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
from .bucketize__opt import bucketize
from .cat import cat, cat_out
from .cauchy import cauchy, cauchy_
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .cholesky import cholesky
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .clamp_max import clamp_max
from .clamp_max__opt import clamp_max_
from .clip___opt import clip_
from .clip__opt import clip
from .combinations import combinations
from .concat__opt import concat
from .concatenate import concatenate
from .constant_pad_nd import constant_pad_nd
from .contiguous import contiguous
from .conv1d__opt import conv1d
from .conv2d__opt import conv2d
from .conv3d import conv3d
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .copysign_ import copysign_
from .cos import cos, cos_
from .cosh___opt import cosh_
from .cosh__opt import cosh
from .cosh_out__opt import cosh_out
from .count_nonzero import count_nonzero
from .cudnn_rnn_backward import cudnn_rnn_backward
from .cummax import cummax
from .cummaxmin_backward__opt import cummaxmin_backward
from .cummin import cummin
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
from .dot import dot
from .dropout import dropout, dropout_backward, native_dropout_backward
from .elu import elu, elu_
from .embedding_bag import _embedding_bag, embedding_bag
from .embedding_bag_backward import _embedding_bag_backward
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .erfinv_ import erfinv, erfinv_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1___opt import expm1_
from .expm1__opt import expm1
from .expm1_out__opt import expm1_out
from .exponential import exponential
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
from .flash_attention_backward_gcu400__opt import (
    scaled_dot_product_efficient_attention_backward,
)
from .flip import flip
from .floor_ import floor_
from .floor__opt import floor
from .floor_out__opt import floor_out
from .fmax import fmax, fmax_out
from .fmin import fmin, fmin_out
from .fmod import fmod_, fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .frac_ import frac, frac_
from .fractional_max_pool2d import fractional_max_pool2d, fractional_max_pool2d_backward
from .fractional_max_pool3d_backward import fractional_max_pool3d_backward
from .full__opt import full
from .full_like__opt import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu, glu_backward
from .greater import greater, greater_out
from .greater_equal import greater_equal, greater_equal_, greater_equal_scalar
from .greater_scalar__opt import greater_scalar
from .greater_scalar_out__opt import greater_scalar_out
from .grid_sample import grid_sample
from .group_mm import group_mm
from .groupnorm import group_norm, group_norm_backward
from .gru import gru_data
from .gt import gt, gt_scalar, gt_tensor_
from .gt_scalar__opt import gt_scalar_
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
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
from .less_equal import less_equal
from .less_equal_scalar__opt import less_equal_scalar
from .lgamma_ import lgamma, lgamma_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linalg_cholesky import linalg_cholesky

# Also bind the bare name: the aten patcher (op_registrar._resolve_live_override)
# looks the override up as flag_gems.<generic function __name__>, which for
# aten::linalg_ldl_factor is "ldl_factor".  Without this alias, use_gems() would
# keep registering the generic flag_gems.ops implementation instead of this one.
from .linalg_ldl_factor import ldl_factor  # noqa: F401
from .linalg_ldl_factor import ldl_factor as linalg_ldl_factor
from .linalg_ldl_solve import linalg_ldl_solve
from .linalg_matrix_rank import linalg_matrix_rank
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
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logical_xor_ import logical_xor_
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
from .multi_margin_loss_out import multi_margin_loss_out
from .multinomial import multinomial
from .multiply_ import multiply_
from .mv import mv
from .nan_to_num import nan_to_num
from .native_layer_norm import native_layer_norm
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .negative import negative, negative_, negative_out
from .nested_compute_contiguous_strides_offsets import (
    nested_compute_contiguous_strides_offsets,
)
from .new_zeros import new_zeros, new_zeros_out
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
from .not_equal import not_equal
from .not_equal_scalar__opt import not_equal_scalar
from .one_hot import one_hot
from .ones import ones
from .ones_like import ones_like
from .pad__opt import pad
from .per_token_group_quant_fp8 import per_token_group_quant_fp8
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
from .prod import prod, prod_dim
from .quantile import quantile
from .rad2deg import rad2deg, rad2deg_
from .randint import randint
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from .relu import relu, relu_
from .relu6 import relu6
from .remainder import remainder, remainder_
from .repeat import repeat
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
from .rsub import rsub_tensor
from .rsub__opt import rsub_scalar
from .rwkv_mm_sparsity import rwkv_mm_sparsity
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
from .signbit__opt import signbit
from .silu import silu, silu_, silu_backward
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
from .special_bessel_j1 import special_bessel_j1
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .special_modified_bessel_k0 import (
    special_modified_bessel_k0,
    special_modified_bessel_k0_out,
)
from .special_scaled_modified_bessel_k0 import (
    special_scaled_modified_bessel_k0,
    special_scaled_modified_bessel_k0_out,
)
from .special_xlogy import special_xlogy, special_xlogy_
from .sqrt import sqrt, sqrt_
from .square import square
from .square__opt import square_
from .square_out import square_out
from .stack import stack
from .std import std
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .svd import svd
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .thnn_fused_gru_cell_backward import _thnn_fused_gru_cell_backward_out
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_dtype
from .top_k_per_row_decode import top_k_per_row_decode
from .topk import topk
from .tril import tril, tril_, tril_out
from .trunc_ import trunc, trunc_
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
from .upsample_nearest1d_backward import (
    upsample_nearest1d_backward,
    upsample_nearest1d_backward_grad_input,
)
from .upsample_nearest2d import upsample_nearest2d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack__opt import vstack
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .zero import zero, zero_, zero_out
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "_aminmax",
    "_aminmax_out",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_cummin_helper",
    "_embedding_bag",
    "_embedding_bag_backward",
    "_euclidean_dist",
    "_foreach_add__List",
    "_foreach_addcdiv__Scalar",
    "_foreach_ceil",
    "_foreach_ceil_",
    "_foreach_clamp_min__List",
    "_foreach_cos_",
    "_foreach_div__ScalarList",
    "_foreach_exp",
    "_foreach_exp_",
    "_foreach_lerp__List",
    "_foreach_log10",
    "_foreach_log2",
    "_foreach_log2_",
    "_foreach_maximum_List",
    "_foreach_minimum__Scalar",
    "_foreach_minimum__ScalarList",
    "_foreach_minimum_Scalar",
    "_foreach_minimum_ScalarList",
    "_foreach_mul__List",
    "_foreach_mul__Scalar",
    "_foreach_mul__ScalarList",
    "_foreach_mul__Tensor",
    "_foreach_mul_List",
    "_foreach_mul_Scalar",
    "_foreach_mul_ScalarList",
    "_foreach_mul_Tensor",
    "_foreach_neg",
    "_foreach_pow__List",
    "_foreach_sign",
    "_foreach_sign_",
    "_foreach_sin",
    "_foreach_sqrt",
    "_foreach_sqrt_",
    "_foreach_sub__Scalar",
    "_foreach_sub__ScalarList",
    "_foreach_sub_Scalar",
    "_foreach_sub_ScalarList",
    "_foreach_tan",
    "_foreach_zero",
    "_foreach_zero_",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_index_put_impl_",
    "_is_all_true",
    "_native_batch_norm_legit_functional",
    "_safe_softmax",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_fused_attention_overrideable",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_thnn_fused_gru_cell_backward_out",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unique2",
    "_upsample_bicubic2d_aa",
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
    "acos",
    "acosh",
    "acosh_",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "add",
    "add_",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv",
    "addmv_out",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
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
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsin_out",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctan",
    "arctan_",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "asin_",
    "asinh",
    "asinh_",
    "asinh_out",
    "atan",
    "atan2",
    "atan2_out",
    "atan_",
    "atanh",
    "atanh_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "baddbmm",
    "baddbmm_out",
    "batch_norm_backward_reduce",
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
    "bucketize",
    "cat",
    "cat_out",
    "cauchy",
    "cauchy_",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "cholesky",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "combinations",
    "concat",
    "concatenate",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "copy",
    "copy_",
    "copysign",
    "copysign_",
    "copysign_out",
    "copysign_scalar",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cudnn_rnn_backward",
    "cummax",
    "cummaxmin_backward",
    "cummin",
    "cumsum",
    "cumsum_out",
    "deg2rad",
    "deg2rad_",
    "deg2rad_out",
    "diag_embed",
    "diagonal_backward",
    "digamma",
    "digamma_",
    "digamma_out",
    "div_mode",
    "div_mode_",
    "dot",
    "dropout",
    "dropout_backward",
    "elu",
    "elu_",
    "embedding_bag",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erf_",
    "erfinv",
    "erfinv_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "exp_out",
    "expm1",
    "expm1_",
    "expm1_out",
    "exponential",
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
    "floor_divide",
    "floor_divide_",
    "floor_out",
    "fmax",
    "fmax_out",
    "fmin",
    "fmin_out",
    "fmod_",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "frac",
    "frac_",
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
    "greater",
    "greater_equal",
    "greater_equal_",
    "greater_equal_scalar",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "grid_sample",
    "group_mm",
    "group_norm",
    "group_norm_backward",
    "gru_data",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor",
    "gt_tensor_",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
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
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_select",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "kron",
    "layer_norm",
    "layer_norm_backward",
    "ldl_factor",
    "le",
    "le_scalar",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "less_equal",
    "less_equal_scalar",
    "lgamma",
    "lgamma_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linalg_cholesky",
    "linalg_ldl_factor",
    "linalg_ldl_solve",
    "linalg_matrix_rank",
    "linalg_polar",
    "linear",
    "linspace",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log1p",
    "log1p_",
    "log1p_out",
    "log2",
    "log2_",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logaddexp",
    "logaddexp_out",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
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
    "mse_loss",
    "mul",
    "mul_",
    "multi_margin_loss_out",
    "multinomial",
    "multiply_",
    "mv",
    "nan_to_num",
    "native_dropout_backward",
    "native_layer_norm",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "negative_out",
    "nested_compute_contiguous_strides_offsets",
    "new_zeros",
    "new_zeros_out",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nonzero",
    "nonzero_numpy",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "not_equal",
    "not_equal_scalar",
    "one_hot",
    "ones",
    "ones_like",
    "pad",
    "per_token_group_quant_fp8",
    "poisson",
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
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "repeat",
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
    "router_gemm",
    "rrelu_with_noise_backward",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "rwkv_mm_sparsity",
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
    "segment_reduce",
    "segment_reduce_out",
    "select_scatter",
    "selu",
    "selu_",
    "sgn_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "signbit",
    "silu",
    "silu_",
    "silu_backward",
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
    "special_bessel_j1",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_scaled_modified_bessel_k0",
    "special_scaled_modified_bessel_k0_out",
    "special_xlogy",
    "special_xlogy_",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "stack",
    "std",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "svd",
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
    "top_k_per_row_decode",
    "topk",
    "tril",
    "tril_",
    "tril_out",
    "true_divide",
    "true_divide_",
    "trunc",
    "trunc_",
    "trunc_divide",
    "trunc_divide_",
    "uniform_",
    "upsample_bicubic2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest1d_backward",
    "upsample_nearest1d_backward_grad_input",
    "upsample_nearest2d",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vector_norm",
    "view_copy",
    "vstack",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]

# Enflame has no vendor FFT, so point the STFT path at the aten::fft-backed
# implementation.  Imported and executed here (not at package import) so the
# shared flag_gems.ops.stft module already exists to be patched.
from ..stft_fft import install as _install_enflame_fft

_install_enflame_fft()
del _install_enflame_fft
