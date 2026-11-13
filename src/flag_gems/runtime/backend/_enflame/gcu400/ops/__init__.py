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

# Import order is significant: the trailing ``# noqa: F811`` rebindings
# intentionally override earlier imports. Reordering them would change
# which implementation is registered, so isort must leave this file alone.
# isort: skip_file

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
from .upsample_nearest1d_backward import (
    upsample_nearest1d_backward,
    upsample_nearest1d_backward_grad_input,
)
from .upsample_nearest_exact2d_backward import (
    _upsample_nearest_exact2d_backward,
)
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
from ._functional_sym_constrain_range import (
    _functional_sym_constrain_range,
)
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._is_all_true import _is_all_true
from ._native_batch_norm_legit_functional import (
    _native_batch_norm_legit_functional,
)
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
from .bincount import bincount
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
from .copysign_ import copysign_

# gcu400 copies of the generic implementations, carrying this backend's
# work-arounds (masked-load lane leaks, grid limits, strided-store tails,
# NaN degree conversion).  Keeping them here leaves the shared ops untouched.
from .conv_tbc import conv_tbc, conv_tbc_out
from .cumulative_trapezoid import cumulative_trapezoid
from .dstack import dstack
from .index_fill import index_fill, index_fill_
from .ldexp_ import ldexp_
from .pairwise_distance import pairwise_distance
from .special_laguerre_polynomial_l import (
    special_laguerre_polynomial_l,
    special_laguerre_polynomial_l_out,
)
from .thnn_conv2d import thnn_conv2d
from .cos import cos, cos_
from .cosh___opt import cosh_
from .cosh__opt import cosh
from .cosh_out__opt import cosh_out
from .count_nonzero import count_nonzero
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
from ._sparse_semi_structured_mm import _sparse_semi_structured_mm
from .logcumsumexp_out import logcumsumexp_out
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
from .glu_backward_jvp import glu_backward_jvp
from .greater import greater, greater_out
from .greater_equal import greater_equal, greater_equal_, greater_equal_scalar
from .greater_scalar__opt import greater_scalar
from .greater_scalar_out__opt import greater_scalar_out
from .grid_sample import grid_sample
from .group_mm import group_mm
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar, gt_tensor_
from .gt_scalar__opt import gt_scalar_
from .hadamard_transform import (
    hadamard_transform,
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)
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
from .isreal import isreal
from .kron import kron
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .less_equal import less_equal, less_equal_scalar
from .lgamma_ import lgamma, lgamma_
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linalg_cholesky import linalg_cholesky

# Also bind the bare name: the aten patcher (op_registrar._resolve_live_override)
# looks the override up as flag_gems.<generic function __name__>, which for
# aten::linalg_ldl_factor is "ldl_factor".  Without this alias, use_gems() would
# keep registering the generic flag_gems.ops implementation instead of this one.
from .linalg_ldl_factor import ldl_factor  # noqa: F401
from .linalg_ldl_factor import ldl_factor as linalg_ldl_factor
from .linalg_inv_ex import linalg_inv_ex
from .linalg_matmul import linalg_matmul
from .linalg_matrix_rank import linalg_matrix_rank
from .linalg_solve_ex import linalg_solve_ex
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
from .masked_select_backward import masked_select_backward
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
from .not_equal__opt import not_equal
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
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
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
from .slow_conv_dilated2d import slow_conv_dilated2d, slow_conv_dilated2d_out
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
from .embedding_bag_backward import _embedding_bag_backward
from .embedding_bag import _embedding_bag, embedding_bag
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
from .te_rmsnorm import te_rmsnorm_bwd
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
from .upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest1d import upsample_nearest1d
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
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from .cudnn_rnn_backward import cudnn_rnn_backward
from .thnn_fused_gru_cell_backward import _thnn_fused_gru_cell_backward_out

# 以下算子原本只有公共实现，已复制到 gcu400 后端作为专用实现，
# 后续优化只需改动本目录文件，不再原地修改公共代码。
from ._dirichlet_grad import _dirichlet_grad
from ._fused_sgd_ import _fused_sgd_, _fused_sgd__tensor_lr
from ._histogramdd_from_bin_cts import (
    _histogramdd_from_bin_cts,
    _histogramdd_from_bin_cts_out,
)
from ._logcumsumexp import _logcumsumexp, _logcumsumexp_out
from ._lu_with_info import _lu_with_info
from .adaptive_avg_pool1d import adaptive_avg_pool1d
from .cast_Byte import cast_Byte
from .convolution_overrideable import (
    convolution_overrideable,
    convolution_overrideable_out,
)
from .convert_indices_from_coo_to_csr import (
    _convert_indices_from_coo_to_csr,
    _convert_indices_from_coo_to_csr_out,
)
from .dstack import run as dstack  # noqa: F811
from .kaiser_window import (
    kaiser_window,
    kaiser_window_beta,
    kaiser_window_periodic,
)
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_norm import linalg_norm
from .lu_solve import lu_solve, lu_solve_out
from .nanmean import nanmean, nanmean_out
from .polygamma import polygamma, polygamma_, polygamma_out
from .put import put, put_out
from .scaled_mm import scaled_mm, scaled_mm_out
from .slogdet import slogdet
from .trapz import trapz

# foreach 系列的公共实现位于 codegen 共享模块，这里只导出本批目标函数，
# 避免覆盖 gcu400 上已有的其它 foreach 专用实现。
from ._addmm_activation import _addmm_activation, _addmm_activation_out
from ._foreach_addcmul_scalarlist import _foreach_addcmul__ScalarList
from ._foreach_binary import _foreach_addcdiv__Tensor
from ._foreach_pow_scalarlist import _foreach_pow_ScalarList
from ._foreach_unary import _foreach_round, _foreach_trunc
from .as_strided_scatter import as_strided_scatter
from .ctc_loss import ctc_loss

# 以下算子在 _FULL_CONFIG 中仍指向公共实现，已把公共实现复制到本目录，
# 逐一注册为 gcu400 专属实现；后续优化只改动本目录文件，不原地修改公共代码。
from .__ilshift__ import __ilshift__
from .__rshift__ import __rshift__
from .__xor__ import xor, xor_, xor_scalar, xor_scalar_, xor_scalar_tensor
from ._adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from ._adaptive_avg_pool3d import (
    _adaptive_avg_pool3d as adaptive_avg_pool3d,
    _adaptive_avg_pool3d_out as adaptive_avg_pool3d_out,
)
from ._batch_norm_impl_index import batch_norm_impl_index as _batch_norm_impl_index
from ._batch_norm_impl_index_backward import _batch_norm_impl_index_backward
from ._batch_norm_with_update import _batch_norm_with_update
from ._cholesky_solve_helper import _cholesky_solve_helper
from ._chunk_cat import chunk_cat as _chunk_cat
from ._coalesce import _coalesce, _coalesce_out
from ._coalesced_ import _coalesced_
from ._conj import _conj
from ._conj_copy import _conj_copy, _conj_copy_out
from ._conj_physical import _conj_physical, _conj_physical_out
from ._convolution_double_backward import _convolution_double_backward
from ._convolution_mode import _convolution_mode
from ._copy_from_and_resize import _copy_from_and_resize, _copy_from_and_resize_out
from ._cslt_sparse_mm import _cslt_sparse_mm
from ._ctc_loss import _ctc_loss, _ctc_loss_out
from ._ctc_loss_backward import _ctc_loss_backward, _ctc_loss_backward_out
from ._cummax_helper import _cummax_helper
from ._dimI import _dimI
from ._embedding_bag_dense_backward import _embedding_bag_dense_backward
from ._embedding_bag_sparse_backward import _embedding_bag_sparse_backward
from ._fake_quantize_learnable_per_channel_affine import (
    _fake_quantize_learnable_per_channel_affine,
)
from ._fake_quantize_learnable_per_channel_affine_backward import (
    _fake_quantize_learnable_per_channel_affine_backward,
)
from ._fake_quantize_learnable_per_tensor_affine import (
    _fake_quantize_learnable_per_tensor_affine,
)
from ._fake_quantize_learnable_per_tensor_affine_backward import (
    _fake_quantize_learnable_per_tensor_affine_backward,
)
from ._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
)
from ._fill_mem_eff_dropout_mask_ import _fill_mem_eff_dropout_mask_
from ._flash_attention_forward import _flash_attention_forward
from ._foreach_binary import (
    _foreach_add_List,
    _foreach_add_Scalar,
    _foreach_add_ScalarList,
    _foreach_add_Tensor,
    _foreach_add__Scalar,
    _foreach_add__ScalarList,
    _foreach_add__Tensor,
    _foreach_addcdiv_Scalar,
    _foreach_addcdiv_ScalarList,
    _foreach_addcdiv_Tensor,
    _foreach_addcdiv__ScalarList,
    _foreach_addcmul_Scalar,
    _foreach_addcmul_ScalarList,
    _foreach_addcmul_Tensor,
    _foreach_addcmul__Scalar,
    _foreach_addcmul__Tensor,
    _foreach_clamp_max_List,
    _foreach_clamp_max_Scalar,
    _foreach_clamp_max_ScalarList,
    _foreach_clamp_max__List,
    _foreach_clamp_max__Scalar,
    _foreach_clamp_max__ScalarList,
    _foreach_clamp_min_List,
    _foreach_clamp_min_Scalar,
    _foreach_clamp_min_ScalarList,
    _foreach_clamp_min__Scalar,
    _foreach_clamp_min__ScalarList,
    _foreach_copy,
    _foreach_copy_,
    _foreach_div_List,
    _foreach_div_Scalar,
    _foreach_div_ScalarList,
    _foreach_div_Tensor,
    _foreach_div__List,
    _foreach_div__Scalar,
    _foreach_div__Tensor,
    _foreach_lerp_List,
    _foreach_lerp_Scalar,
    _foreach_lerp_ScalarList,
    _foreach_lerp__Scalar,
    _foreach_lerp__ScalarList,
    _foreach_maximum_Scalar,
    _foreach_maximum_ScalarList,
    _foreach_maximum__List,
    _foreach_maximum__Scalar,
    _foreach_maximum__ScalarList,
    _foreach_minimum_List,
    _foreach_minimum__List,
    _foreach_pow_List,
    _foreach_pow_Scalar,
    _foreach_pow_ScalarAndTensor,
    _foreach_pow__Scalar,
    _foreach_pow__ScalarList,
    _foreach_sub_List,
    _foreach_sub__List,
)
from ._foreach_reduction import _foreach_max, _foreach_norm
from ._foreach_unary import (
    _foreach_abs,
    _foreach_abs_,
    _foreach_acos,
    _foreach_acos_,
    _foreach_asin,
    _foreach_asin_,
    _foreach_atan,
    _foreach_atan_,
    _foreach_cos,
    _foreach_cosh,
    _foreach_cosh_,
    _foreach_erf,
    _foreach_erf_,
    _foreach_erfc,
    _foreach_erfc_,
    _foreach_expm1,
    _foreach_expm1_,
    _foreach_floor,
    _foreach_floor_,
    _foreach_frac,
    _foreach_frac_,
    _foreach_lgamma,
    _foreach_lgamma_,
    _foreach_log,
    _foreach_log10_,
    _foreach_log1p,
    _foreach_log1p_,
    _foreach_log_,
    _foreach_neg_,
    _foreach_reciprocal,
    _foreach_reciprocal_,
    _foreach_round_,
    _foreach_rsqrt,
    _foreach_rsqrt_,
    _foreach_sigmoid,
    _foreach_sigmoid_,
    _foreach_sin_,
    _foreach_sinh,
    _foreach_sinh_,
    _foreach_tan_,
    _foreach_tanh,
    _foreach_tanh_,
    _foreach_trunc_,
)

# floor 使用独立的一维融合 kernel，覆盖公共 unary 执行器的同名函数。
from ._foreach_floor_opt import _foreach_floor, _foreach_floor_  # noqa: F811

# lgamma 的 libdevice extern 在大形状高并发下触发设备 SIP abort（P16），
# 换独立的纯 Triton Lanczos flat kernel，覆盖公共 unary 执行器的同名函数。
from ._foreach_lgamma import _foreach_lgamma, _foreach_lgamma_  # noqa: F811
from ._fused_adamw_ import _fused_adamw_, _fused_adamw__tensor_lr
from ._fused_rms_norm import _fused_rms_norm, _fused_rms_norm_backward
from ._histogramdd_from_bin_tensors import (
    _histogramdd_from_bin_tensors,
    _histogramdd_from_bin_tensors_out,
)
from ._indices_copy import _indices_copy, _indices_copy_out
from ._is_any_true import _is_any_true
from ._linalg_det import _linalg_det
from ._linalg_eigh import _linalg_eigh
from ._linalg_solve_ex import _linalg_solve_ex
from ._linalg_svd import _linalg_svd
from ._make_dep_token import _make_dep_token
from ._native_batch_norm_legit import (
    _native_batch_norm_legit,
    _native_batch_norm_legit_no_stats,
    _native_batch_norm_legit_no_stats_out,
    _native_batch_norm_legit_out,
)
from ._neg_view_copy import _neg_view_copy, _neg_view_copy_out
from ._nested_view_from_jagged_copy import _nested_view_from_jagged_copy
from ._nnz import _nnz
from ._pack_padded_sequence import _pack_padded_sequence
from ._pad_circular import _pad_circular
from ._pin_memory import _pin_memory
from ._prelu_kernel_backward import _prelu_kernel_backward
from ._reshape_alias import _reshape_alias
from ._sample_dirichlet import _sample_dirichlet
from ._scaled_dot_product_cudnn_attention import _scaled_dot_product_cudnn_attention
from ._scaled_dot_product_efficient_attention import (
    _scaled_dot_product_efficient_attention,
)
from ._scaled_dot_product_fused_attention_overrideable_backward import (
    scaled_dot_product_fused_attention_overrideable_backward,
)
from ._scaled_grouped_mm_v2 import _scaled_grouped_mm_v2
from ._scaled_mm_v2 import _scaled_mm_v2
from ._slow_conv2d_forward import _slow_conv2d_forward
from ._sobol_engine_ff_ import _sobol_engine_ff_
from ._sobol_engine_initialize_state_ import _sobol_engine_initialize_state_
from ._sobol_engine_scramble_ import _sobol_engine_scramble_
from ._sparse_mm_reduce_impl import _sparse_mm_reduce_impl
from ._sparse_semi_structured_addmm import _sparse_semi_structured_addmm
from ._sparse_semi_structured_tile import _sparse_semi_structured_tile
from ._spsolve import spsolve
from ._stack import _stack
from ._thnn_fused_gru_cell import _thnn_fused_gru_cell, _thnn_fused_gru_cell_out
from ._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from ._thnn_fused_lstm_cell_backward import _thnn_fused_lstm_cell_backward
from ._unsafe_masked_index import _unsafe_masked_index
from ._unsafe_view import _unsafe_view
from ._upsample_lanczos2d_aa_backward import (
    upsample_lanczos2d_aa_backward,
    upsample_lanczos2d_aa_backward_grad_input,
)
from ._upsample_nearest_exact1d_backward import (
    _upsample_nearest_exact1d_backward,
    _upsample_nearest_exact1d_backward_grad_input,
)
from ._upsample_nearest_exact3d_backward import (
    _upsample_nearest_exact3d_backward,
    _upsample_nearest_exact3d_backward_grad_input,
)
from ._validate_sparse_csc_tensor_args import _validate_sparse_csc_tensor_args
from ._values_copy import _values_copy, _values_copy_out
from ._weight_norm import _weight_norm
from ._wrapped_quantized_linear_prepacked import _wrapped_quantized_linear_prepacked
from .absolute__common import absolute_
from .acos_ import acos_
from .adaptive_avg_pool2d import (
    _adaptive_avg_pool2d_out as adaptive_avg_pool2d_out,
    adaptive_avg_pool2d,
)
from .adaptive_max_pool1d import adaptive_max_pool1d
from .adaptive_max_pool2d import adaptive_max_pool2d
from .addcmul__common import addcmul_
from .addmv_ import addmv_
from .affine_grid_generator_backward import affine_grid_generator_backward
from .alias import alias
from .and_scalar import and_scalar
from .arccos import arccos_
from .arctan2 import arctan2, arctan2_
from .arctanh import arctanh, arctanh_out
from .asin import asin
from .batch_norm import batch_norm_backward
from .batch_norm_backward_elemt import batch_norm_backward_elemt
from .batch_norm_elemt import batch_norm_elemt
from .batch_norm_gather_stats import batch_norm_gather_stats
from .batch_norm_gather_stats_with_counts import batch_norm_gather_stats_with_counts
from .batch_norm_stats import batch_norm_stats
from .batch_norm_update_stats import batch_norm_update_stats
from .bilinear import bilinear
from .binary_cross_entropy import binary_cross_entropy, binary_cross_entropy_out
from .cast_Char import cast_Char
from .cdist import _cdist_backward, _cdist_forward, cdist
from .chain_matmul import chain_matmul_out, run as chain_matmul
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .choose_qparams_per_tensor import choose_qparams_per_tensor
from .clamp__common import clamp_min, clamp_min_
from .clone import clone
from .coalesce import coalesce
from .col_indices_copy import col_indices_copy, col_indices_copy_out
from .column_stack import column_stack, column_stack_out
from .conv_depthwise2d import _conv_depthwise2d
from .conv_tbc_backward import conv_tbc_backward
from .conv_transpose1d import conv_transpose1d
from .conv_transpose2d import conv_transpose2d
from .conv_transpose3d import conv_transpose3d
from .convert_indices_from_csr_to_coo import (
    _convert_indices_from_csr_to_coo,
    _convert_indices_from_csr_to_coo_out,
)
from .convolution import convolution
from .convolution_backward import convolution_backward, convolution_backward_out
from .convolution_backward_overrideable import (
    convolution_backward_overrideable,
    convolution_backward_overrideable_out,
)
from .copy_sparse_to_sparse_ import copy_sparse_to_sparse_
from .cov import cov
from .cudnn_attention_backward import cudnn_attention_backward
from .cudnn_attention_forward import cudnn_attention_forward
from .cudnn_batch_norm import cudnn_batch_norm, cudnn_batch_norm_out
from .cudnn_convolution import cudnn_convolution
from .cudnn_convolution_relu import cudnn_convolution_relu, cudnn_convolution_relu_out
from .cudnn_convolution_transpose import cudnn_convolution_transpose
from .cudnn_grid_sampler_backward import (
    cudnn_grid_sampler_backward,
    cudnn_grid_sampler_backward_out,
)
from .cumprod import cumprod, cumprod_, cumprod_backward
from .dense_dim import dense_dim
from .dequantize import dequantize
from .det import det
from .detach_copy import detach_copy
from .diagflat import run as diagflat
from .diagonal_copy import diagonal_copy
from .diff import diff
from .dimV import _dimV as dimV
from .div__common import true_divide_out
from .divide import divide
from .elu__common import elu_backward
from .embedding import embedding, embedding_backward
from .embedding_bag import _embedding_bag_forward_only
from .embedding_dense_backward import embedding_dense_backward
from .embedding_sparse_backward import embedding_sparse_backward
from .empty_affine_quantized import _empty_affine_quantized, _empty_affine_quantized_out
from .empty_per_channel_affine_quantized import _empty_per_channel_affine_quantized
from .expand_as import expand_as
from .expand_copy import expand_copy
from .fake_quantize_per_tensor_affine_cachemask import (
    fake_quantize_per_tensor_affine_cachemask,
    fake_quantize_per_tensor_affine_cachemask_out,
)
from .fake_quantize_per_tensor_affine_cachemask_backward import (
    fake_quantize_per_tensor_affine_cachemask_backward,
)
from .fft_rfft import fft_rfft
from .flash_attention_backward import (
    efficient_attention_backward,
    flash_attention_backward,
    scaled_dot_product_cudnn_attention_backward,
    scaled_dot_product_flash_attention_backward,
)
from .flatten_dense_tensors import flatten_dense_tensors
from .fliplr import fliplr
from .flipud import flipud
from .float_power_ import (
    float_power_scalar_tensor,
    float_power_scalar_tensor_out,
    float_power_tensor_scalar,
    float_power_tensor_scalar_,
    float_power_tensor_scalar_out,
    float_power_tensor_tensor,
    float_power_tensor_tensor_,
    float_power_tensor_tensor_out,
)
from .frobenius_norm import frobenius_norm
from .fused_moving_avg_obs_fake_quant import fused_moving_avg_obs_fake_quant
from .fw_primal_copy import _fw_primal_copy, _fw_primal_copy_out
from .glu_jvp import glu_jvp
from .gradient import gradient
from .greater_ import greater_
from .grid_sampler import grid_sampler
from .grid_sampler_3d import grid_sampler_3d
from .gru import gru, gru_data
from .gru_cell import gru_cell
from .hash_tensor import hash_tensor
from .histogram import histogram_bin_ct, histogram_bins_tensor
from .histogramdd import histogramdd
from .huber_loss import huber_loss, huber_loss_out
from .hypot_ import hypot_
from .index_reduce import index_reduce, index_reduce_, index_reduce_out
from .indices import indices
from .indices_copy import indices_copy, indices_copy_out
from .infinitely_differentiable_gelu_backward import (
    infinitely_differentiable_gelu_backward,
)
from .int_repr import int_repr
from .inverse import inverse
from .ior_tensor import ior_tensor
from .is_same_size import is_same_size
from .isposinf import isposinf
from .istft import istft
from .kthvalue import kthvalue
from .ldexp import ldexp, ldexp_out
from .le_ import le_, le_scalar_
from .leaky_relu__common import leaky_relu_backward
from .less_equal_ import less_equal_, less_equal_scalar_
from .linalg_cholesky_ex import linalg_cholesky_ex
from .linalg_cond import linalg_cond, linalg_cond_p_str
from .linalg_det import linalg_det, linalg_det_out
from .linalg_eig import linalg_eig
from .linalg_eigh import linalg_eigh
from .linalg_lstsq import linalg_lstsq
from .linalg_lu_factor_ex import linalg_lu_factor_ex, linalg_lu_factor_ex_out
from .linalg_lu_solve import linalg_lu_solve
from .linalg_matrix_exp import linalg_matrix_exp, linalg_matrix_exp_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_matrix_rank import (
    linalg_matrix_rank_out,
    linalg_matrix_rank_tol,
    linalg_matrix_rank_tol_out,
)
from .linalg_multi_dot import linalg_multi_dot, linalg_multi_dot_out
from .linalg_pinv import linalg_pinv
from .linalg_polar import linalg_polar_out
from .linalg_qr import linalg_qr, linalg_qr_out
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linalg_svd import linalg_svd
from .linalg_tensorsolve import linalg_tensorsolve
from .linear_backward import linear_backward
from .log_normal import log_normal
from .log_sigmoid_backward import log_sigmoid_backward, log_sigmoid_backward_out
from .log_sigmoid_forward import log_sigmoid_forward
from .logaddexp2 import logaddexp2, logaddexp2_out
from .logcumsumexp import logcumsumexp
from .logical_not__common import logical_not_
from .lstm import lstm
from .lstm_cell import lstm_cell
from .make_per_channel_quantized_tensor import _make_per_channel_quantized_tensor
from .make_per_tensor_quantized_tensor import (
    _make_per_tensor_quantized_tensor,
    _make_per_tensor_quantized_tensor_out,
)
from .masked_scatter_backward import masked_scatter_backward
from .matrix_exp import matrix_exp
from .matrix_power import matrix_power, matrix_power_out
from .mean import mean
from .median import median, median_dim, median_dim_values, median_out
from .miopen_batch_norm import miopen_batch_norm
from .mish import mish, mish_
from .mixed_dtypes_linear import mixed_dtypes_linear
from .mkldnn_rnn_layer import mkldnn_rnn_layer
from .mode import mode
from .msort import msort, msort_out
from .multilabel_margin_loss_forward import multilabel_margin_loss_forward
from .nan_to_num_ import nan_to_num_
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .native_group_norm import native_group_norm
from .native_norm import native_norm
from .new_full import new_full
from .nll_loss2d import nll_loss2d
from .nnpack_spatial_convolution import nnpack_spatial_convolution
from .norm import norm, norm_scalar, norm_scalaropt_dim
from .norm_except_dim import norm_except_dim
from .or_tensor import or_tensor
from .orgqr import orgqr, orgqr_out
from .ormqr import ormqr
from .permute_copy import permute_copy
from .pinverse import pinverse
from .poisson_nll_loss import poisson_nll_loss
from .put_ import put_
from .quantize_per_channel import quantize_per_channel, quantize_per_channel_out
from .quantize_per_tensor import quantize_per_tensor, quantize_per_tensor_out
from .quantized_batch_norm import quantized_batch_norm, quantized_batch_norm_out
from .quantized_gru import quantized_gru_data, quantized_gru_input
from .quantized_lstm import quantized_lstm
from .quantized_max_pool1d import quantized_max_pool1d, quantized_max_pool1d_out
from .quantized_max_pool2d import quantized_max_pool2d, quantized_max_pool2d_out
from .quantized_max_pool3d import quantized_max_pool3d, quantized_max_pool3d_out
from .range import range
from .ravel import ravel
from .real import real
from .relu6_ import relu6_
from .replication_pad2d import replication_pad2d, replication_pad2d_out
from .resize_as_sparse_ import resize_as_sparse_
from .resolve_conj import resolve_conj
from .row_indices_copy import row_indices_copy, row_indices_copy_out
from .row_stack import row_stack, row_stack_out
from .rrelu_with_noise import rrelu_with_noise, rrelu_with_noise_
from .scaled_grouped_mm import scaled_grouped_mm
from .scatter_reduce__common import scatter_reduce_, scatter_reduce_out
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from .sgn import sgn, sgn_out
from .sign import sign, sign_, sign_out
from .sinh import sinh
from .slow_conv_transpose3d import slow_conv_transpose3d
from .smm import smm
from .softmax__common import softmax_backward_out
from .softplus__common import softplus_backward
from .sparse_broadcast_to import sparse_broadcast_to
from .sparse_broadcast_to_copy import sparse_broadcast_to_copy
from .sparse_bsc_tensor import sparse_bsc_tensor
from .sparse_bsr_tensor import sparse_bsr_tensor
from .sparse_compressed_tensor import sparse_compressed_tensor
from .sparse_coo_tensor import sparse_coo_tensor
from .sparse_csr_tensor import sparse_csr_tensor
from .sparse_dim import sparse_dim
from .sparse_resize_ import sparse_resize_
from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out
from .special_bessel_j0 import special_bessel_j0
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_erf import special_erf
from .special_erfc import erfc, erfc_, special_erfc
from .special_erfinv import special_erfinv, special_erfinv_, special_erfinv_out
from .special_exp2 import special_exp2
from .special_expit import special_expit
from .special_expm1 import special_expm1
from .special_gammaincc import special_gammaincc
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_logit import special_logit, special_logit_out
from .special_modified_bessel_i0 import (
    special_modified_bessel_i0,
    special_modified_bessel_i0_out,
)
from .special_psi import special_psi, special_psi_out
from .special_round import special_round, special_round_out
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)
from .split_copy import split_copy
from .split_with_sizes_copy import split_with_sizes_copy
from .squeeze_copy import squeeze_copy
from .sspaddmm import sspaddmm, sspaddmm_out
from .std_mean import (
    std_mean,
    std_mean_correction,
    std_mean_correction_names,
    std_mean_correction_out,
    std_mean_dim,
    std_mean_names_dim,
)
from .stft import stft, stft_center
from .t_ import t_
from .take_along_dim import take_along_dim, take_along_dim_out
from .tensordot import tensordot, tensordot_out
from .thnn_fused_gru_cell_backward__common import _thnn_fused_gru_cell_backward
from .threshold_ import threshold_
from .to__common import to_copy
from .trace import trace
from .transpose_copy import transpose_copy
from .trapezoid import trapezoid_x
from .triangular_solve import triangular_solve, triangular_solve_out
from .triplet_margin_loss import triplet_margin_loss
from .true_divide import true_divide_tensor
from .true_divide_ import true_divide_tensor_
from .underscore_sobol_engine_draw import underscore_sobol_engine_draw
from .unflatten import unflatten
from .uniform__common import uniform
from .upsample_bicubic2d_backward import (
    upsample_bicubic2d_backward,
    upsample_bicubic2d_backward_grad_input,
)
from .upsample_bilinear2d import upsample_bilinear2d
from .upsample_bilinear2d_backward import (
    upsample_bilinear2d_backward,
    upsample_bilinear2d_backward_grad_input,
)
from .upsample_trilinear3d_backward import upsample_trilinear3d_backward
from .values_copy import values_copy, values_copy_out
from .vander import vander
from .vdot import vdot
from .view_as_complex_copy import view_as_complex_copy
from .view_as_real_copy import view_as_real_copy
from .weight_int8pack_mm import weight_int8pack_mm
from .xlogy_ import xlogy_, xlogy_tensor_scalar_

__all__ = [
    "__ilshift__",
    "__rshift__",
    "_adaptive_avg_pool2d_backward",
    "_addmm_activation",
    "_addmm_activation_out",
    "_aminmax",
    "_aminmax_out",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_batch_norm_impl_index",
    "_batch_norm_impl_index_backward",
    "_batch_norm_with_update",
    "_cdist_backward",
    "_cdist_forward",
    "_cholesky_solve_helper",
    "_chunk_cat",
    "_coalesce",
    "_coalesce_out",
    "_coalesced_",
    "_conj",
    "_conj_copy",
    "_conj_copy_out",
    "_conj_physical",
    "_conj_physical_out",
    "_conv_depthwise2d",
    "_convert_indices_from_coo_to_csr",
    "_convert_indices_from_coo_to_csr_out",
    "_convert_indices_from_csr_to_coo",
    "_convert_indices_from_csr_to_coo_out",
    "_convolution_double_backward",
    "_convolution_mode",
    "_copy_from_and_resize",
    "_copy_from_and_resize_out",
    "_cslt_sparse_mm",
    "_ctc_loss",
    "_ctc_loss_backward",
    "_ctc_loss_backward_out",
    "_ctc_loss_out",
    "_cummax_helper",
    "_cummin_helper",
    "_dimI",
    "_dirichlet_grad",
    "_embedding_bag",
    "_embedding_bag_backward",
    "_embedding_bag_dense_backward",
    "_embedding_bag_forward_only",
    "_embedding_bag_sparse_backward",
    "_empty_affine_quantized",
    "_empty_affine_quantized_out",
    "_empty_per_channel_affine_quantized",
    "_euclidean_dist",
    "_fake_quantize_learnable_per_channel_affine",
    "_fake_quantize_learnable_per_channel_affine_backward",
    "_fake_quantize_learnable_per_tensor_affine",
    "_fake_quantize_learnable_per_tensor_affine_backward",
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_fill_mem_eff_dropout_mask_",
    "_flash_attention_forward",
    "_foreach_abs",
    "_foreach_abs_",
    "_foreach_acos",
    "_foreach_acos_",
    "_foreach_add__List",
    "_foreach_add__Scalar",
    "_foreach_add__ScalarList",
    "_foreach_add__Tensor",
    "_foreach_add_List",
    "_foreach_add_Scalar",
    "_foreach_add_ScalarList",
    "_foreach_add_Tensor",
    "_foreach_addcdiv__Scalar",
    "_foreach_addcdiv__ScalarList",
    "_foreach_addcdiv__Tensor",
    "_foreach_addcdiv_Scalar",
    "_foreach_addcdiv_ScalarList",
    "_foreach_addcdiv_Tensor",
    "_foreach_addcmul__Scalar",
    "_foreach_addcmul__ScalarList",
    "_foreach_addcmul__Tensor",
    "_foreach_addcmul_Scalar",
    "_foreach_addcmul_ScalarList",
    "_foreach_addcmul_Tensor",
    "_foreach_asin",
    "_foreach_asin_",
    "_foreach_atan",
    "_foreach_atan_",
    "_foreach_ceil",
    "_foreach_ceil_",
    "_foreach_clamp_max__List",
    "_foreach_clamp_max__Scalar",
    "_foreach_clamp_max__ScalarList",
    "_foreach_clamp_max_List",
    "_foreach_clamp_max_Scalar",
    "_foreach_clamp_max_ScalarList",
    "_foreach_clamp_min__List",
    "_foreach_clamp_min__Scalar",
    "_foreach_clamp_min__ScalarList",
    "_foreach_clamp_min_List",
    "_foreach_clamp_min_Scalar",
    "_foreach_clamp_min_ScalarList",
    "_foreach_copy",
    "_foreach_copy_",
    "_foreach_cos",
    "_foreach_cos_",
    "_foreach_cosh",
    "_foreach_cosh_",
    "_foreach_div__List",
    "_foreach_div__Scalar",
    "_foreach_div__ScalarList",
    "_foreach_div__Tensor",
    "_foreach_div_List",
    "_foreach_div_Scalar",
    "_foreach_div_ScalarList",
    "_foreach_div_Tensor",
    "_foreach_erf",
    "_foreach_erf_",
    "_foreach_erfc",
    "_foreach_erfc_",
    "_foreach_exp",
    "_foreach_exp_",
    "_foreach_expm1",
    "_foreach_expm1_",
    "_foreach_floor",
    "_foreach_floor_",
    "_foreach_frac",
    "_foreach_frac_",
    "_foreach_lerp__List",
    "_foreach_lerp__Scalar",
    "_foreach_lerp__ScalarList",
    "_foreach_lerp_List",
    "_foreach_lerp_Scalar",
    "_foreach_lerp_ScalarList",
    "_foreach_lgamma",
    "_foreach_lgamma_",
    "_foreach_log",
    "_foreach_log10",
    "_foreach_log10_",
    "_foreach_log1p",
    "_foreach_log1p_",
    "_foreach_log2",
    "_foreach_log2_",
    "_foreach_log_",
    "_foreach_max",
    "_foreach_maximum__List",
    "_foreach_maximum__Scalar",
    "_foreach_maximum__ScalarList",
    "_foreach_maximum_List",
    "_foreach_maximum_Scalar",
    "_foreach_maximum_ScalarList",
    "_foreach_minimum__List",
    "_foreach_minimum__Scalar",
    "_foreach_minimum__ScalarList",
    "_foreach_minimum_List",
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
    "_foreach_neg_",
    "_foreach_norm",
    "_foreach_pow__List",
    "_foreach_pow__Scalar",
    "_foreach_pow__ScalarList",
    "_foreach_pow_List",
    "_foreach_pow_Scalar",
    "_foreach_pow_ScalarAndTensor",
    "_foreach_pow_ScalarList",
    "_foreach_reciprocal",
    "_foreach_reciprocal_",
    "_foreach_round",
    "_foreach_round_",
    "_foreach_rsqrt",
    "_foreach_rsqrt_",
    "_foreach_sigmoid",
    "_foreach_sigmoid_",
    "_foreach_sign",
    "_foreach_sign_",
    "_foreach_sin",
    "_foreach_sin_",
    "_foreach_sinh",
    "_foreach_sinh_",
    "_foreach_sqrt",
    "_foreach_sqrt_",
    "_foreach_sub__List",
    "_foreach_sub__Scalar",
    "_foreach_sub__ScalarList",
    "_foreach_sub_List",
    "_foreach_sub_Scalar",
    "_foreach_sub_ScalarList",
    "_foreach_tan",
    "_foreach_tan_",
    "_foreach_tanh",
    "_foreach_tanh_",
    "_foreach_trunc",
    "_foreach_trunc_",
    "_foreach_zero",
    "_foreach_zero_",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_fused_adamw_",
    "_fused_adamw__tensor_lr",
    "_fused_rms_norm",
    "_fused_rms_norm_backward",
    "_fused_sgd_",
    "_fused_sgd__tensor_lr",
    "_fw_primal_copy",
    "_fw_primal_copy_out",
    "_histogramdd_from_bin_cts",
    "_histogramdd_from_bin_cts_out",
    "_histogramdd_from_bin_tensors",
    "_histogramdd_from_bin_tensors_out",
    "_index_put_impl_",
    "_indices_copy",
    "_indices_copy_out",
    "_is_all_true",
    "_is_any_true",
    "_linalg_det",
    "_linalg_eigh",
    "_linalg_solve_ex",
    "_linalg_svd",
    "_logcumsumexp",
    "_logcumsumexp_out",
    "_lu_with_info",
    "_make_dep_token",
    "_make_per_channel_quantized_tensor",
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit_functional",
    "_native_batch_norm_legit_no_stats",
    "_native_batch_norm_legit_no_stats_out",
    "_native_batch_norm_legit_out",
    "_neg_view_copy",
    "_neg_view_copy_out",
    "_nested_view_from_jagged_copy",
    "_nnz",
    "_pack_padded_sequence",
    "_pad_circular",
    "_pin_memory",
    "_prelu_kernel_backward",
    "_reshape_alias",
    "_safe_softmax",
    "_sample_dirichlet",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_fused_attention_overrideable",
    "_scaled_grouped_mm_v2",
    "_scaled_mm_v2",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_slow_conv2d_forward",
    "_sobol_engine_ff_",
    "_sobol_engine_initialize_state_",
    "_sobol_engine_scramble_",
    "_sparse_mm_reduce_impl",
    "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_mm",
    "_sparse_semi_structured_tile",
    "_stack",
    "_thnn_fused_gru_cell",
    "_thnn_fused_gru_cell_backward",
    "_thnn_fused_gru_cell_backward_out",
    "_thnn_fused_gru_cell_out",
    "_thnn_fused_lstm_cell",
    "_thnn_fused_lstm_cell_backward",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unique2",
    "_unsafe_masked_index",
    "_unsafe_view",
    "_upsample_bicubic2d_aa",
    "_upsample_bilinear2d_aa",
    "_upsample_nearest_exact1d",
    "_upsample_nearest_exact1d_backward",
    "_upsample_nearest_exact1d_backward_grad_input",
    "_upsample_nearest_exact1d_out",
    "_upsample_nearest_exact1d_vec",
    "_upsample_nearest_exact2d",
    "_upsample_nearest_exact2d_backward",
    "_upsample_nearest_exact2d_out",
    "_upsample_nearest_exact2d_vec",
    "_upsample_nearest_exact3d",
    "_upsample_nearest_exact3d_backward",
    "_upsample_nearest_exact3d_backward_grad_input",
    "_upsample_nearest_exact3d_out",
    "_upsample_nearest_exact3d_vec",
    "_validate_sparse_csc_tensor_args",
    "_values_copy",
    "_values_copy_out",
    "_weight_norm",
    "_wrapped_quantized_linear_prepacked",
    "abs",
    "abs_",
    "absolute",
    "absolute_",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_out",
    "adaptive_avg_pool3d",
    "adaptive_avg_pool3d_out",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "add",
    "add_",
    "addcmul",
    "addcmul_",
    "addcmul_out",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv",
    "addmv_",
    "addmv_out",
    "affine_grid_generator_backward",
    "alias",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "amin",
    "amin_",
    "aminmax",
    "and_scalar",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arccos",
    "arccos_",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsin_out",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctan",
    "arctan2",
    "arctan2_",
    "arctan_",
    "arctanh",
    "arctanh_",
    "arctanh_out",
    "argmax",
    "argmin",
    "argsort",
    "as_strided_scatter",
    "asin",
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
    "batch_norm_backward",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bilinear",
    "binary_cross_entropy",
    "binary_cross_entropy_out",
    "binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits_",
    "bincount",
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
    "cast_Byte",
    "cast_Char",
    "cat",
    "cat_out",
    "cauchy",
    "cauchy_",
    "cdist",
    "cdiv",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "chain_matmul",
    "chain_matmul_out",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "cholesky_solve_out",
    "choose_qparams_per_tensor",
    "clamp",
    "clamp_",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "clone",
    "coalesce",
    "col_indices_copy",
    "col_indices_copy_out",
    "column_stack",
    "column_stack_out",
    "combinations",
    "concat",
    "concatenate",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_tbc_backward",
    "conv_tbc_out",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "convolution_backward",
    "convolution_backward_out",
    "convolution_backward_overrideable",
    "convolution_backward_overrideable_out",
    "convolution_overrideable",
    "convolution_overrideable_out",
    "copy",
    "copy_",
    "copy_sparse_to_sparse_",
    "copysign_",
    "copysign_scalar",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cov",
    "cross_attention",
    "ctc_loss",
    "cudnn_attention_backward",
    "cudnn_attention_forward",
    "cudnn_batch_norm",
    "cudnn_batch_norm_out",
    "cudnn_convolution",
    "cudnn_convolution_relu",
    "cudnn_convolution_relu_out",
    "cudnn_convolution_transpose",
    "cudnn_grid_sampler_backward",
    "cudnn_grid_sampler_backward_out",
    "cudnn_rnn_backward",
    "cummax",
    "cummaxmin_backward",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumprod_backward",
    "cumsum",
    "cumsum_out",
    "cumulative_trapezoid",
    "deg2rad",
    "deg2rad_",
    "deg2rad_out",
    "dense_dim",
    "dequantize",
    "det",
    "detach_copy",
    "diag_embed",
    "diagflat",
    "diagonal_backward",
    "diagonal_copy",
    "diff",
    "digamma",
    "digamma_",
    "digamma_out",
    "dimV",
    "div_mode",
    "div_mode_",
    "divide",
    "dot",
    "dropout",
    "dropout_backward",
    "dstack",
    "efficient_attention_backward",
    "einsum",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
    "embedding_bag",
    "embedding_dense_backward",
    "embedding_sparse_backward",
    "embedding_spmdm",
    "empty",
    "eq",
    "eq_scalar",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "erfinv_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "exp_out",
    "expand_as",
    "expand_copy",
    "expm1",
    "expm1_",
    "expm1_out",
    "exponential",
    "exponential_",
    "eye",
    "eye_m",
    "fake_quantize_per_tensor_affine_cachemask",
    "fake_quantize_per_tensor_affine_cachemask_backward",
    "fake_quantize_per_tensor_affine_cachemask_out",
    "feature_dropout",
    "feature_dropout_",
    "fft_rfft",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "flash_attention_backward",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flatten_dense_tensors",
    "flip",
    "fliplr",
    "flipud",
    "float_power_scalar_tensor",
    "float_power_scalar_tensor_out",
    "float_power_tensor_scalar",
    "float_power_tensor_scalar_",
    "float_power_tensor_scalar_out",
    "float_power_tensor_tensor",
    "float_power_tensor_tensor_",
    "float_power_tensor_tensor_out",
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
    "fp8_paged_mqa_logits",
    "frac",
    "frac_",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "fractional_max_pool3d_backward",
    "frobenius_norm",
    "full",
    "full_like",
    "fused_moving_avg_obs_fake_quant",
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
    "glu_backward_jvp",
    "glu_jvp",
    "gradient",
    "greater",
    "greater_",
    "greater_equal",
    "greater_equal_",
    "greater_equal_scalar",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "grid_sample",
    "grid_sampler",
    "grid_sampler_3d",
    "group_mm",
    "group_norm",
    "group_norm_backward",
    "gru",
    "gru_cell",
    "gru_data",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor",
    "gt_tensor_",
    "hadamard_transform",
    "hadamard_transform_12N",
    "hadamard_transform_20N",
    "hadamard_transform_28N",
    "hadamard_transform_40N",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
    "hash_tensor",
    "histc",
    "histogram_bin_ct",
    "histogram_bins_tensor",
    "histogramdd",
    "hstack",
    "huber_loss",
    "huber_loss_out",
    "hypot",
    "hypot_",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_reduce_",
    "index_reduce_out",
    "index_select",
    "indices",
    "indices_copy",
    "indices_copy_out",
    "infinitely_differentiable_gelu_backward",
    "int_repr",
    "inverse",
    "ior_tensor",
    "is_same_size",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "isposinf",
    "isreal",
    "istft",
    "kaiser_window",
    "kaiser_window_beta",
    "kaiser_window_periodic",
    "kron",
    "kthvalue",
    "layer_norm",
    "layer_norm_backward",
    "ldexp",
    "ldexp_",
    "ldexp_out",
    "ldl_factor",
    "le",
    "le_",
    "le_scalar",
    "le_scalar_",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_backward",
    "leaky_relu_out",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "less_equal",
    "less_equal_",
    "less_equal_scalar",
    "less_equal_scalar_",
    "lgamma",
    "lgamma_",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linalg_cholesky",
    "linalg_cholesky_ex",
    "linalg_cond",
    "linalg_cond_p_str",
    "linalg_det",
    "linalg_det_out",
    "linalg_eig",
    "linalg_eigh",
    "linalg_inv_ex",
    "linalg_ldl_factor",
    "linalg_lstsq",
    "linalg_lu_factor_ex",
    "linalg_lu_factor_ex_out",
    "linalg_lu_solve",
    "linalg_matmul",
    "linalg_matrix_exp",
    "linalg_matrix_exp_out",
    "linalg_matrix_norm",
    "linalg_matrix_norm_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_matrix_rank",
    "linalg_matrix_rank_out",
    "linalg_matrix_rank_tol",
    "linalg_matrix_rank_tol_out",
    "linalg_multi_dot",
    "linalg_multi_dot_out",
    "linalg_norm",
    "linalg_pinv",
    "linalg_polar",
    "linalg_polar_out",
    "linalg_qr",
    "linalg_qr_out",
    "linalg_solve_ex",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linalg_svd",
    "linalg_tensorsolve",
    "linear",
    "linear_backward",
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
    "log_normal",
    "log_sigmoid",
    "log_sigmoid_backward",
    "log_sigmoid_backward_out",
    "log_sigmoid_forward",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logaddexp",
    "logaddexp2",
    "logaddexp2_out",
    "logaddexp_out",
    "logcumsumexp",
    "logcumsumexp_out",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "logit",
    "logit_",
    "logit_out",
    "logspace",
    "lstm",
    "lstm_cell",
    "lt",
    "lt_scalar",
    "lu_solve",
    "lu_solve_out",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_scatter_backward",
    "masked_select",
    "masked_select_backward",
    "matrix_exp",
    "matrix_power",
    "matrix_power_out",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "maximum",
    "mean",
    "mean_dim",
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "min",
    "min_dim",
    "minimum",
    "miopen_batch_norm",
    "mish",
    "mish_",
    "mixed_dtypes_linear",
    "mkldnn_rnn_layer",
    "mm",
    "mm_out",
    "mode",
    "mse_loss",
    "msort",
    "msort_out",
    "mul",
    "mul_",
    "multi_margin_loss_out",
    "multilabel_margin_loss_forward",
    "multinomial",
    "multiply_",
    "mv",
    "nan_to_num",
    "nan_to_num_",
    "nanmean",
    "nanmean_out",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "native_dropout_backward",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "negative_out",
    "nested_compute_contiguous_strides_offsets",
    "new_full",
    "new_zeros",
    "new_zeros_out",
    "nll_loss2d",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nnpack_spatial_convolution",
    "nonzero",
    "nonzero_numpy",
    "norm",
    "norm_except_dim",
    "norm_scalar",
    "norm_scalaropt_dim",
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
    "or_tensor",
    "orgqr",
    "orgqr_out",
    "ormqr",
    "pad",
    "pairwise_distance",
    "per_token_group_quant_fp8",
    "permute_copy",
    "pinverse",
    "poisson",
    "poisson_nll_loss",
    "polar",
    "polygamma",
    "polygamma_",
    "polygamma_out",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "prod",
    "prod_dim",
    "put",
    "put_",
    "put_out",
    "quantile",
    "quantize_per_channel",
    "quantize_per_channel_out",
    "quantize_per_tensor",
    "quantize_per_tensor_out",
    "quantized_batch_norm",
    "quantized_batch_norm_out",
    "quantized_gru_data",
    "quantized_gru_input",
    "quantized_lstm",
    "quantized_max_pool1d",
    "quantized_max_pool1d_out",
    "quantized_max_pool2d",
    "quantized_max_pool2d_out",
    "quantized_max_pool3d",
    "quantized_max_pool3d_out",
    "rad2deg",
    "rad2deg_",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "range",
    "ravel",
    "real",
    "reciprocal",
    "reciprocal_",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "relu",
    "relu6",
    "relu6_",
    "relu_",
    "remainder",
    "remainder_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad1d",
    "replication_pad1d_out",
    "replication_pad2d",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "replication_pad2d_out",
    "replication_pad3d",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "rms_norm",
    "round",
    "round_",
    "round_out",
    "router_gemm",
    "row_indices_copy",
    "row_indices_copy_out",
    "row_stack",
    "row_stack_out",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "rrelu_with_noise_backward",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "rwkv_mm_sparsity",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_cudnn_attention_backward",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_dot_product_flash_attention_backward",
    "scaled_dot_product_fused_attention_overrideable_backward",
    "scaled_grouped_mm",
    "scaled_mm",
    "scaled_mm_out",
    "ScaleDotProductAttention",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "segment_reduce",
    "segment_reduce_out",
    "select_scatter",
    "selu",
    "selu_",
    "sgn",
    "sgn_",
    "sgn_out",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "sign",
    "sign_",
    "sign_out",
    "signbit",
    "silu",
    "silu_",
    "silu_backward",
    "simple_unique_flat",
    "sin",
    "sin_",
    "sinh",
    "sinh_",
    "slice_backward",
    "slice_scatter",
    "slogdet",
    "slow_conv_dilated2d",
    "slow_conv_dilated2d_out",
    "slow_conv_transpose3d",
    "smm",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "softmax",
    "softmax_backward",
    "softmax_backward_out",
    "softmax_out",
    "softplus",
    "softplus_backward",
    "softshrink",
    "softshrink_out",
    "sort",
    "sort_stable",
    "sorted_indices_unique_flat",
    "sorted_quick_unique_flat",
    "sparse_broadcast_to",
    "sparse_broadcast_to_copy",
    "sparse_bsc_tensor",
    "sparse_bsr_tensor",
    "sparse_compressed_tensor",
    "sparse_coo_tensor",
    "sparse_csr_tensor",
    "sparse_dim",
    "sparse_resize_",
    "sparse_sampled_addmm",
    "sparse_sampled_addmm_out",
    "special_bessel_j0",
    "special_bessel_j1",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_erf",
    "special_erfc",
    "special_erfinv",
    "special_erfinv_",
    "special_erfinv_out",
    "special_exp2",
    "special_expit",
    "special_expm1",
    "special_gammaincc",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_laguerre_polynomial_l",
    "special_laguerre_polynomial_l_out",
    "special_legendre_polynomial_p",
    "special_logit",
    "special_logit_out",
    "special_modified_bessel_i0",
    "special_modified_bessel_i0_out",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_psi",
    "special_psi_out",
    "special_round",
    "special_round_out",
    "special_scaled_modified_bessel_k0",
    "special_scaled_modified_bessel_k0_out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "special_xlogy",
    "special_xlogy_",
    "split_copy",
    "split_with_sizes_copy",
    "spsolve",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "squeeze_copy",
    "sspaddmm",
    "sspaddmm_out",
    "stack",
    "std",
    "std_mean",
    "std_mean_correction",
    "std_mean_correction_names",
    "std_mean_correction_out",
    "std_mean_dim",
    "std_mean_names_dim",
    "stft",
    "stft_center",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "svd",
    "t_",
    "t_copy",
    "t_copy_out",
    "take_along_dim",
    "take_along_dim_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "te_rmsnorm_bwd",
    "tensordot",
    "tensordot_out",
    "thnn_conv2d",
    "threshold",
    "threshold_",
    "threshold_backward",
    "tile",
    "to_copy",
    "to_dtype",
    "top_k_per_row_decode",
    "topk",
    "topk_w8a16_fp8",
    "trace",
    "transpose_copy",
    "trapezoid_x",
    "trapz",
    "triangular_solve",
    "triangular_solve_out",
    "tril",
    "tril_",
    "tril_out",
    "triplet_margin_loss",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "true_divide_tensor",
    "true_divide_tensor_",
    "trunc",
    "trunc_",
    "trunc_divide",
    "trunc_divide_",
    "underscore_sobol_engine_draw",
    "unflatten",
    "uniform",
    "uniform_",
    "upsample_bicubic2d",
    "upsample_bicubic2d_backward",
    "upsample_bicubic2d_backward_grad_input",
    "upsample_bilinear2d",
    "upsample_bilinear2d_backward",
    "upsample_bilinear2d_backward_grad_input",
    "upsample_lanczos2d_aa_backward",
    "upsample_lanczos2d_aa_backward_grad_input",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest1d_backward",
    "upsample_nearest1d_backward_grad_input",
    "upsample_nearest2d",
    "upsample_trilinear3d_backward",
    "values_copy",
    "values_copy_out",
    "vander",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vdot",
    "vector_norm",
    "view_as_complex_copy",
    "view_as_real_copy",
    "view_copy",
    "vstack",
    "weight_int8pack_mm",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "xlogy_",
    "xlogy_tensor_scalar_",
    "xor",
    "xor_",
    "xor_scalar",
    "xor_scalar_",
    "xor_scalar_tensor",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]

# 算子导出名与同名子模块冲突时，Python 会把子模块对象绑定到包属性并覆盖此前
# 导入的函数，注册机制只收集函数对象就会漏掉这些专用实现；在所有 import 结束后
# 重新绑定，使它们恢复为 gcu400 专用函数。
from .div import true_divide, true_divide_  # noqa: F811
from .signbit__opt import signbit  # noqa: F811

# 公共 foreach codegen 实现使用二维 grid，大 shape 下 ``grid.y`` 会超过
# GCU400 的 255 硬件上限；等所有 import 结束后重新绑定为专属一维实现。
from ._foreach_flat_ops import (  # noqa: F811
    _foreach_add__ScalarList,
    _foreach_clamp_max__Scalar,
    _foreach_div__List,
)

# foreach ScalarList 最大值类算子的公共二维 grid 越界，统一覆盖为一维融合实现。
from ._foreach_scalarlist_opt import (  # noqa: F811
    _foreach_clamp_min_ScalarList,
    _foreach_clamp_min__ScalarList,
)

# maximum 的 ScalarList overload 同样规避公共二维 grid 越界。
from ._foreach_maximum_scalarlist_opt import (  # noqa: F811
    _foreach_maximum_ScalarList,
    _foreach_maximum__ScalarList,
)

# Enflame has no vendor FFT, so point the STFT path at the aten::fft-backed
# implementation.  Imported and executed here (not at package import) so the
# shared flag_gems.ops.stft module already exists to be patched.
from ..stft_fft import install as _install_enflame_fft
from .cross_attention import cross_attention
from .embedding_spmdm import embedding_spmdm
from .empty import empty
from .fp8_paged_mqa_logits import cdiv, fp8_paged_mqa_logits
from .topk_w8a16_fp8 import topk_w8a16_fp8

_install_enflame_fft()
del _install_enflame_fft

# CTC 内部入口的测试 / benchmark 直接调用 flag_gems.ops.<name>，而
# SpecOpRegistrar 只把后端特化写进 flag_gems 顶层 globals，ops 模块属性仍指向
# 公共实现（fp64 走通用 kernel 会在 GCU 上触发 PassManager 编译失败）。
# 仿照上面 stft 的既有做法，在后端加载时把 4 个 gcu400 特化实现同步到
# flag_gems.ops，使直接调用路径与 aten dispatch 路径使用同一份实现；
# 不修改任何公共算子源码，也不改测试文件。
import flag_gems.ops as _flag_gems_ops_for_ctc  # noqa: E402

_flag_gems_ops_for_ctc._ctc_loss = _ctc_loss
_flag_gems_ops_for_ctc._ctc_loss_out = _ctc_loss_out
_flag_gems_ops_for_ctc._ctc_loss_backward = _ctc_loss_backward
_flag_gems_ops_for_ctc._ctc_loss_backward_out = _ctc_loss_backward_out
del _flag_gems_ops_for_ctc

# histogram 的测试 / benchmark 直接 `from flag_gems.ops.histogram import ...`
# 调用子模块属性，公共实现是 torch.bucketize host 管线（topsaten bucketize 在
# GCU 上 hard abort，且违反 host 隔离红线）。同样在后端加载时把 gcu400 特化
# 同步到子模块与包属性，使直调路径与 aten dispatch 路径使用同一份实现；
# 不修改任何公共算子源码，也不改测试文件。
import flag_gems.ops as _flag_gems_ops_for_histogram  # noqa: E402
import flag_gems.ops.histogram as _flag_gems_histogram_module  # noqa: E402

_flag_gems_histogram_module.histogram_bin_ct = histogram_bin_ct
_flag_gems_histogram_module.histogram_bins_tensor = histogram_bins_tensor
_flag_gems_ops_for_histogram.histogram_bin_ct = histogram_bin_ct
_flag_gems_ops_for_histogram.histogram_bins_tensor = histogram_bins_tensor
del _flag_gems_histogram_module
del _flag_gems_ops_for_histogram
# tests/test_upsample_bilinear2d_backward.py 也直接调用 flag_gems.ops.<name>，
# 若不在此同步，该文件中的 nonfinite（0 * Inf -> NaN）与 disjoint 用例会绕过
# gcu400 实现、继续跑公共路径。做法与上面的 CTC 一致：只改后端加载行为，
# 不修改公共算子源码，也不改测试文件。
import flag_gems.ops as _flag_gems_ops_for_upsample  # noqa: E402

_flag_gems_ops_for_upsample.upsample_bilinear2d_backward = upsample_bilinear2d_backward
_flag_gems_ops_for_upsample.upsample_bilinear2d_backward_grad_input = (
    upsample_bilinear2d_backward_grad_input
)
del _flag_gems_ops_for_upsample

from . import einsum  # noqa: E402
