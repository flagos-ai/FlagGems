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

from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from ._batch_norm_impl_index import batch_norm_impl_index as _batch_norm_impl_index
from ._batch_norm_no_update import _batch_norm_no_update
from ._dyn_quant_pack_4bit_weight import _dyn_quant_pack_4bit_weight
from ._embedding_bag_dense_backward import _embedding_bag_dense_backward
from ._euclidean_dist import _euclidean_dist
from ._functional_sym_constrain_range import _functional_sym_constrain_range
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._fused_adam import _fused_adam, _fused_adam_
from ._fused_rms_norm import _fused_rms_norm  # noqa: F401
from ._is_all_true import _is_all_true
from ._linalg_eigvals import _linalg_eigvals
from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from ._native_batch_norm_legit_no_training import _native_batch_norm_legit_no_training
from ._nested_view_from_buffer_copy import _nested_view_from_buffer_copy
from ._pdist_backward import _pdist_backward
from ._pdist_forward import _pdist_forward, pdist
from ._prelu_kernel import _prelu_kernel  # noqa: F401
from ._prelu_kernel_backward import _prelu_kernel_backward  # noqa: F401
from ._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from ._masked_scale import _masked_scale  # noqa: F401
from ._unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa  # noqa: F401
from ._upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from ._upsample_nearest_exact3d import _upsample_nearest_exact3d
from .abs import abs, abs_
from .absolute import absolute, absolute_
from .acos import acos, acos_
from .acosh import acosh, acosh_  # noqa: F401
from .adaptive_avg_pool2d import adaptive_avg_pool2d
from .adaptive_max_pool2d import adaptive_max_pool2d
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d import adaptive_max_pool3d
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .add import add, add_
from .addbmm import addbmm, addbmm_  # noqa: F401
from .addcdiv import addcdiv, addcdiv_, addcdiv_out
from .addcmul import addcmul, addcmul_, addcmul_out
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out  # noqa: F401
from .addmm_ import addmm_
from .addmv import addmv, addmv_out
from .addr import addr
from .affine_grid_generator import affine_grid_generator  # noqa: F401
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
from .arccosh import arccosh_  # noqa: F401
from .arcsin import arcsin, arcsin_, arcsin_out
from .arcsinh import arcsinh, arcsinh_, arcsinh_out  # noqa: F401
from .arctan import arctan, arctan_
from .arctan2 import arctan2, arctan2_
from .argmax import argmax
from .argmin import argmin
from .argsort import argsort
from .as_strided_copy import as_strided_copy, as_strided_copy_out
from .as_strided_scatter import as_strided_scatter
from .asin import asin, asin_
from .asinh_ import asinh_  # noqa: F401
from .assert_async import _assert_async
from .atan import atan, atan_
from .atan2 import atan2, atan2_, atan2_out
from .atanh import atanh, atanh_  # noqa: F401
from .attention import (  # noqa: F401
    ScaleDotProductAttention,
    efficient_attention_backward,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
    scaled_dot_product_efficient_attention_backward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .avg_pool3d import avg_pool3d
from .avg_pool3d_backward import avg_pool3d_backward
from .baddbmm import baddbmm, baddbmm_, baddbmm_out
from .batch_norm import batch_norm, batch_norm_backward
from .bernoulli import bernoulli
from .bernoulli_ import bernoulli_
from .binary_cross_entropy import binary_cross_entropy, binary_cross_entropy_out
from .binary_cross_entropy_backward import binary_cross_entropy_backward
from .binary_cross_entropy_with_logits import binary_cross_entropy_with_logits
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
from .bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_  # noqa: F401
from .bitwise_xor import (
    bitwise_xor_scalar,
    bitwise_xor_scalar_,
    bitwise_xor_scalar_tensor,
    bitwise_xor_tensor,
    bitwise_xor_tensor_,
    xor,
    xor_,
    xor_scalar,
    xor_scalar_,
    xor_scalar_tensor,
)
from .bmm import bmm, bmm_out
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .bucketize import bucketize
from .cat import cat, cat_out
from .cdist_backward import _cdist_backward
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .chunk import chunk
from .chunk_cat import chunk_cat  # noqa: F401
from .chunk_cat import chunk_cat as _chunk_cat
from .conj_physical_ import conj_physical_  # noqa: F401
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
from .col2im import col2im
from .concatenate import concatenate
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d
from .conv_transpose2d import conv_transpose2d
from .copy import copy, copy_
from .copysign import copysign, copysign_, copysign_out
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out  # noqa: F401
from .count_nonzero import count_nonzero
from .cudnn_batch_norm import cudnn_batch_norm  # noqa: F401
from .cudnn_batch_norm_backward import cudnn_batch_norm_backward  # noqa: F401
from .cudnn_convolution import cudnn_convolution  # noqa: F401
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .cumsum_ import cumsum_
from .deg2rad import deg2rad, deg2rad_, deg2rad_out
from .dequantize import dequantize
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diagonal_copy import diagonal_copy
from .diff import diff
from .digamma import digamma
from .dist import dist
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
    true_divide_tensor,
)
from .dot import dot
from .dropout import dropout, dropout_backward
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward, embedding_dense_backward
from .eq import eq, eq_, eq_scalar, eq_scalar_
from .erf import erf, erf_, special_erf
from .erfinv import erfinv
from .erfinv_ import erfinv_  # noqa: F401
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
from .floor import floor, floor_, floor_out
from .fmax import fmax, fmax_out
from .fmin import fmin, fmin_out
from .fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .fractional_max_pool2d import fractional_max_pool2d, fractional_max_pool2d_backward
from .full import full
from .full_like import full_like
from .functional_assert_async import _functional_assert_async
from .fused_experts_impl import (
    fused_experts_impl,
    inplace_fused_experts,
    outplace_fused_experts,
)
from .fused_recurrent_gated_delta_rule_fwd import fused_recurrent_gated_delta_rule_fwd
from .gather import gather, gather_backward
from .gcd import gcd, gcd_, gcd_out
from .ge import ge, ge_scalar, greater_equal_
from .gelu import gelu, gelu_, gelu_backward
from .geometric import geometric, geometric_  # noqa: F401
from .get_paged_mqa_logits_metadata import get_paged_mqa_logits_metadata
from .get_scheduler_metadata import get_scheduler_metadata
from .glu import glu, glu_backward
from .greater import greater, greater_out, greater_scalar, greater_scalar_out
from .grid_sample import grid_sample
from .grid_sampler_3d_backward import grid_sampler_3d_backward
from .grouped_mm import group_mm
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar, gt_scalar_, gt_tensor_
from .hadamard_transform import hadamard_transform
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_  # noqa: F401
from .hardtanh_backward import hardtanh_backward  # noqa: F401
from .heaviside_ import heaviside_  # noqa: F401
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_
from .igamma_ import igamma_  # noqa: F401
from .igammac import igammac, igammac_out
from .igammac_ import igammac_
from .im2col import im2col
from .index import index
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_fill import index_fill, index_fill_  # noqa: F401
from .index_put import index_put, index_put_
from .index_put_impl import _index_put_impl_
from .index_reduce import index_reduce_
from .index_select import index_select
from .index_select_backward import index_select_backward
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .isneginf import isneginf, isneginf_out
from .isposinf import isposinf
from .kron import kron
from .kthvalue import kthvalue
from .layernorm import layer_norm, layer_norm_backward
from .lcm import lcm, lcm_
from .le import le, le_, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_backward, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .less_equal import less_equal, less_equal_scalar
from .lgamma import lgamma, lgamma_
from .lift_fresh import lift_fresh  # noqa: F401
from .lift_fresh_copy import lift_fresh_copy
from .linalg_cholesky import linalg_cholesky
from .linalg_cross import linalg_cross, linalg_cross_out
from .linalg_det import linalg_det, linalg_det_out
from .linalg_householder_product import linalg_householder_product
from .linalg_ldl_factor import ldl_factor
from .linalg_ldl_factor_ex import ldl_factor_ex
from .linalg_ldl_solve import linalg_ldl_solve
from .linalg_lstsq import linalg_lstsq
from .linalg_lu import linalg_lu, linalg_lu_out
from .linalg_lu_factor import linalg_lu_factor, linalg_lu_factor_out
from .linalg_lu_factor_ex import linalg_lu_factor_ex, linalg_lu_factor_ex_out
from .linalg_matrix_norm import linalg_matrix_norm
from .linalg_slogdet import linalg_slogdet
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linalg_svd import linalg_svd
from .linear_backward import linear_backward
from .linspace import linspace
from .log import log
from .log1p import log1p, log1p_
from .log2 import log2, log2_
from .log10 import log10, log10_, log10_out  # noqa: F401
from .log_ import log_  # noqa: F401
from .log_sigmoid import log_sigmoid
from .log_sigmoid_backward import log_sigmoid_backward, log_sigmoid_backward_out
from .log_sigmoid_forward import log_sigmoid_forward
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .logaddexp import logaddexp, logaddexp_out
from .logaddexp2 import logaddexp2, logaddexp2_out
from .logcumsumexp import logcumsumexp, logcumsumexp_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not, logical_not_
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor, logical_xor_
from .logspace import logspace
from .logsumexp import logsumexp
from .lt import lt, lt_, lt_scalar, lt_scalar_
from .lu_unpack import lu_unpack, lu_unpack_out
from .margin_ranking_loss import margin_ranking_loss
from .masked_fill import masked_fill, masked_fill_
from .masked_scatter import masked_scatter, masked_scatter_
from .masked_scatter_backward import masked_scatter_backward
from .masked_select import masked_select
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max import max, max_dim
from .max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
    max_pool2d_with_indices_backward,
)
from .max_pool3d_backward import max_pool3d_with_indices_backward
from .max_pool3d_with_indices import max_pool3d_backward, max_pool3d_with_indices
from .max_unpool3d import max_unpool3d  # noqa: F401
from .maximum import maximum
from .mean import mean, mean_dim
from .median import median, median_dim, median_dim_values, median_out
from .min import min, min_dim
from .minimum import minimum
from .miopen_batch_norm_backward import miopen_batch_norm_backward
from .mish import mish, mish_
from .mish_backward import mish_backward
from .mkldnn_rnn_layer import mkldnn_rnn_layer
from .mm import mm, mm_out
from .mode import mode
from .moe_sum import moe_sum
from .mse_loss import mse_loss
from .mse_loss_backward import mse_loss_backward
from .mul import mul, mul_
from .multinomial import multinomial
from .multiply import multiply
from .multiply_ import multiply_
from .mv import mv, mv_cluster
from .mvlgamma import mvlgamma
from .mvlgamma_ import mvlgamma_
from .nan_to_num import nan_to_num
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .narrow_copy import narrow_copy
from .native_batch_norm import native_batch_norm
from .native_group_norm import native_group_norm
from .native_layer_norm import native_layer_norm
from .ne import ne, ne_, ne_scalar, ne_scalar_
from .neg import neg, neg_
from .negative import negative
from .negative_ import negative_
from .new_full import new_full
from .new_ones import new_ones
from .nextafter import nextafter, nextafter_
from .nllloss import (
    nll_loss2d,
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
    nll_loss_nd_backward,
    nll_loss_nd_forward,
)
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .nonzero_static import nonzero_static, nonzero_static_out  # noqa: F401
from .norm import norm, norm_scalar, norm_scalaropt_dim
from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .not_equal import not_equal, not_equal_scalar
from .ones import ones
from .ones_like import ones_like
from .ormqr import ormqr
from .pad import constant_pad_nd, pad
from .pairwise_distance import pairwise_distance
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .permute_copy import permute_copy
from .pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out
from .polar import polar
from .polygamma import polygamma, polygamma_, polygamma_out
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
from .quantized_lstm import quantized_lstm
from .rad2deg import rad2deg, rad2deg_
from .rand import rand
from .rand_like import rand_like
from .randint_like import randint_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .range import range  # noqa: F401
from .reciprocal import reciprocal, reciprocal_
from .reflection_pad1d import reflection_pad1d, reflection_pad1d_out
from .reflection_pad1d_backward import reflection_pad1d_backward
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from .reflection_pad2d_backward import reflection_pad2d_backward
from .reflection_pad3d import reflection_pad3d, reflection_pad3d_out
from .reflection_pad3d_backward import reflection_pad3d_backward
from .relu import relu, relu_
from .relu6 import relu6  # noqa: F401
from .renorm import renorm, renorm_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad1d import replication_pad1d, replication_pad1d_out
from .replication_pad2d import replication_pad2d, replication_pad2d_out
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .replication_pad3d import replication_pad3d  # noqa: F401
from .replication_pad3d_backward import replication_pad3d_backward  # noqa: F401
from .resize import resize, resize_
from .resize_output import _resize_output, _resize_output_
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .rnn_relu import rnn_relu
from .roll import roll
from .rot90 import rot90
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub, rsub_scalar, rsub_tensor
from .safe_softmax import _safe_softmax
from .scalar_tensor import scalar_tensor
from .scaled_mm import scaled_mm, scaled_mm_out
from .scaled_softmax import scaled_softmax_backward, scaled_softmax_forward
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from .segment_reduce import (
    _segment_reduce_backward,
    _segment_reduce_backward_out,
    segment_reduce,
    segment_reduce_out,
)
from .select_backward import select_backward
from .select_scatter import select_scatter
from .selu import selu, selu_
from .sgn import sgn, sgn_out
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .sign import sign, sign_out
from .signbit import signbit, signbit_out
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinc import sinc, sinc_, special_sinc
from .sinh import sinh, sinh_  # noqa: F401
from .slice import slice
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .smooth_l1_loss import smooth_l1_loss, smooth_l1_loss_backward, smooth_l1_loss_out
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .soft_margin_loss_backward import soft_margin_loss_backward
from .softmax import softmax, softmax_backward, softmax_backward_out, softmax_out
from .softplus import softplus, softplus_backward
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .sparse_sampled_addmm import (  # noqa: F401
    sparse_sampled_addmm,
    sparse_sampled_addmm_out,
)
from .special_bessel_j0 import special_bessel_j0
from .special_bessel_j1 import special_bessel_j1
from .special_bessel_y0 import special_bessel_y0
from .special_bessel_y1 import special_bessel_y1
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_digamma import special_digamma
from .special_erfcx import special_erfcx
from .special_erfinv import special_erfinv, special_erfinv_, special_erfinv_out
from .special_exp2 import special_exp2
from .special_gammainc import special_gammainc
from .special_gammaincc import special_gammaincc
from .special_gammaln import special_gammaln, special_gammaln_out
from .special_hermite_polynomial_h import special_hermite_polynomial_h
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out  # noqa: F401
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_log1p import special_log1p_out
from .special_log1p import special_log1p  # noqa: F401  (kunlunxin override)
from .special_logit import special_logit, special_logit_out  # noqa: F401
from .special_log_ndtr import special_log_ndtr, special_log_ndtr_
from .special_log_softmax import special_log_softmax
from .special_logsumexp import special_logsumexp
from .special_modified_bessel_k0 import (
    special_modified_bessel_k0,
    special_modified_bessel_k0_out,
)
from .special_multigammaln import special_multigammaln
from .special_ndtri import special_ndtri
from .special_shifted_chebyshev_polynomial_t import (
    special_shifted_chebyshev_polynomial_t,
)
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)
from .special_shifted_chebyshev_polynomial_v import (
    special_shifted_chebyshev_polynomial_v,
)
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .split_with_sizes_copy import split_with_sizes_copy  # noqa: F401
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
from .squeeze_copy import squeeze_copy  # noqa: F401
from .stack import stack
from .std import std
from .sub import sub, sub_, subtract_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .t_copy import t_copy, t_copy_out
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .te_rmsnorm import te_rmsnorm_bwd
from .threshold import threshold, threshold_, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .trace import trace
from .tril import tril, tril_, tril_out
from .triu import triu, triu_
from .trunc import trunc, trunc_
from .unbind_copy import unbind_copy  # noqa: F401
from .unfold_copy import unfold_copy
from .uniform import uniform_
from .unique import _unique2
from .unique_consecutive import unique_consecutive
from .unique_dim import unique_dim
from .unsafe_chunk import unsafe_chunk
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from .upsample_bilinear2d import upsample_bilinear2d  # noqa: F401
from .upsample_linear1d import upsample_linear1d
from .upsample_linear1d_backward import upsample_linear1d_backward
from .upsample_nearest1d import upsample_nearest1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest3d import upsample_nearest3d
from .upsample_trilinear3d import upsample_trilinear3d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vdot import vdot
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack import vstack
from .weight_norm import _weight_norm
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
    "special_hermite_polynomial_h",
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
    "addbmm",
    "addbmm_",
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
    "chunk",
    "_chunk_cat",
    "chunk_cat",
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
    "linalg_ldl_solve",
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
    "nextafter",
    "nextafter_",
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
    "_resize_output",
    "_resize_output_",
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
    "sinc",
    "sinc_",
    "slice",
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
    "unsafe_chunk",
    "upsample_bilinear2d",
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
    "_functional_assert_async",
    "absolute_",
    "acos_",
    "addcmul_",
    "arctan2",
    "arctan2_",
    "atan2",
    "atan2_",
    "atan2_out",
    "bitwise_xor_scalar",
    "bitwise_xor_scalar_",
    "bitwise_xor_scalar_tensor",
    "bitwise_xor_tensor",
    "bitwise_xor_tensor_",
    "xor",
    "xor_",
    "xor_scalar",
    "xor_scalar_",
    "xor_scalar_tensor",
    "broadcast_tensors",
    "copysign_",
    "diff",
    "eq_",
    "eq_scalar_",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "fmax",
    "fmax_out",
    "fmin",
    "fmin_out",
    "gcd",
    "gcd_",
    "gcd_out",
    "gt_scalar_",
    "gt_tensor_",
    "hypot",
    "isneginf",
    "isneginf_out",
    "isposinf",
    "le_",
    "lcm",
    "lcm_",
    "ne_",
    "ne_scalar_",
    "negative_",
    "rrelu_with_noise_backward",
    "sign",
    "sign_out",
    "sgn",
    "sgn_out",
    "square",
    "square_",
    "square_out",
    "true_divide_tensor",
    "var",
    "var_correction",
    "var_dim",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_assert_async",
    "_batch_norm_impl_index",
    "_batch_norm_no_update",
    "_embedding_bag_dense_backward",
    "_native_batch_norm_legit_functional",
    "_native_batch_norm_legit_no_training",
    "_fused_adam",
    "_fused_adam_",
    "_cdist_backward",
    "_dyn_quant_pack_4bit_weight",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "_pdist_backward",
    "_pdist_forward",
    "pdist",
    "pairwise_distance",
    "bernoulli",
    "binary_cross_entropy",
    "binary_cross_entropy_out",
    "binary_cross_entropy_backward",
    "binary_cross_entropy_with_logits",
    "dequantize",
    "embedding_dense_backward",
    "fused_experts_impl",
    "inplace_fused_experts",
    "outplace_fused_experts",
    "fused_recurrent_gated_delta_rule_fwd",
    "get_paged_mqa_logits_metadata",
    "group_mm",
    "native_group_norm",
    "native_batch_norm",
    "native_layer_norm",
    "leaky_relu_backward",
    "margin_ranking_loss",
    "mish",
    "mish_",
    "mish_backward",
    "miopen_batch_norm_backward",
    "mkldnn_rnn_layer",
    "moe_sum",
    "mse_loss_backward",
    "multiply",
    "nansum",
    "nansum_out",
    "nll_loss_nd_backward",
    "nll_loss_nd_forward",
    "nll_loss2d",
    "quantized_lstm",
    "te_rmsnorm_bwd",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_mm",
    "scaled_mm_out",
    "scalar_tensor",
    "special_sinc",
    "softmax_backward_out",
    "softmax_out",
    "softplus_backward",
    "_weight_norm",
    "digamma",
    "dist",
    "special_gammainc",
    "special_gammaincc",
    "special_gammaln",
    "special_gammaln_out",
    "special_digamma",
    "special_erfcx",
    "special_log_ndtr",
    "special_log_ndtr_",
    "special_multigammaln",
    "special_ndtri",
    "erfinv",
    "special_erf",
    "special_erfinv",
    "special_erfinv_",
    "special_erfinv_out",
    "special_exp2",
    "lgamma",
    "lgamma_",
    "igammac",
    "igammac_",
    "igammac_out",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_bessel_j0",
    "special_bessel_j1",
    "special_bessel_y0",
    "special_bessel_y1",
    "special_i0e",
    "special_i0e_out",
    "special_legendre_polynomial_p",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_log1p_out",
    "special_log1p",
    "special_logit",
    "special_logit_out",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_w",
    "log2",
    "log2_",
    "log_sigmoid_backward",
    "log_sigmoid_backward_out",
    "log_sigmoid_forward",
    "logcumsumexp",
    "logcumsumexp_out",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logaddexp",
    "logaddexp_out",
    "polygamma",
    "polygamma_",
    "polygamma_out",
    "addmm_",
    "baddbmm_",
    "baddbmm_out",
    "cholesky_inverse",
    "cholesky_solve",
    "cholesky_solve_out",
    "linalg_cholesky",
    "linalg_cross",
    "linalg_cross_out",
    "mvlgamma",
    "mvlgamma_",
    "_linalg_eigvals",
    "linalg_det",
    "linalg_det_out",
    "linalg_householder_product",
    "ldl_factor",
    "ldl_factor_ex",
    "linalg_lstsq",
    "linalg_lu",
    "linalg_lu_out",
    "linalg_lu_factor",
    "linalg_lu_factor_out",
    "linalg_lu_factor_ex",
    "linalg_lu_factor_ex_out",
    "linalg_matrix_norm",
    "linalg_slogdet",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linalg_svd",
    "linear_backward",
    "lu_unpack",
    "lu_unpack_out",
    "ormqr",
    "norm",
    "norm_scalar",
    "norm_scalaropt_dim",
    "renorm",
    "renorm_",
    "_unsafe_masked_index_put_accumulate",
    "unique_consecutive",
    "unique_dim",
    "argsort",
    "as_strided_scatter",
    "unfold_copy",
    "bucketize",
    "cumsum_",
    "diagonal_copy",
    "histc",
    "index_copy_",
    "_index_put_impl_",
    "index_reduce_",
    "index_select_backward",
    "kthvalue",
    "masked_scatter_backward",
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "mode",
    "roll",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "segment_reduce",
    "segment_reduce_out",
    "select_backward",
    "_nested_view_from_buffer_copy",
    "_scaled_dot_product_fused_attention_overrideable",
    "_upsample_nearest_exact2d_backward",
    "_upsample_nearest_exact3d",
    "_upsample_bicubic2d_aa_backward",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "adaptive_max_pool3d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "col2im",
    "conv_transpose1d",
    "conv_transpose2d",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "grid_sample",
    "grid_sampler_3d_backward",
    "im2col",
    "max_pool2d_with_indices_backward",
    "max_pool3d_with_indices",
    "max_pool3d_backward",
    "max_pool3d_with_indices_backward",
    "reflection_pad1d_backward",
    "reflection_pad2d_backward",
    "reflection_pad3d",
    "reflection_pad3d_out",
    "reflection_pad3d_backward",
    "replication_pad1d",
    "replication_pad1d_out",
    "replication_pad2d",
    "replication_pad2d_out",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "upsample_linear1d_backward",
    "upsample_nearest3d",
]


def _patch_adaptive_max_pool3d_functional():
    """Route torch.nn.functional.adaptive_max_pool3d to the Gems forward.

    The vendor XDNN wrapper on this stack rejects bfloat16 and returns
    uninitialized index memory for float16/float32, so the corresponding
    ``adaptive_max_pool3d_backward`` (registered for CUDA dispatch) would
    receive garbage indices.  Only the tests/benchmark of this op family call
    the functional, so the patch scope is limited; applied at import time,
    mirroring the ``_sunrise`` vendor monkey patches.  Must stay in this
    module (not ``_kunlunxin/__init__.py``): the package __init__ is loaded
    before ``flag_gems.runtime`` finishes initializing.
    """
    import torch
    import torch.nn.functional as F

    from .adaptive_max_pool3d import adaptive_max_pool3d as _gems_fwd

    if getattr(F.adaptive_max_pool3d, "_flag_gems_kunlunxin_patched", False):
        return

    _orig = F.adaptive_max_pool3d

    def _wrapped(input, output_size, return_indices=False):
        # CPU tensors (the ``--ref cpu`` reference path of the harness) must
        # keep the original functional: the Gems kernels only run on CUDA.
        if input.device.type != "cuda":
            return _orig(input, output_size, return_indices=return_indices)
        return _gems_fwd(input, output_size, return_indices=return_indices)

    _wrapped._flag_gems_kunlunxin_patched = True
    F.adaptive_max_pool3d = _wrapped


_patch_adaptive_max_pool3d_functional()

_adaptive_max_pool2d_aten_lib = None


def _patch_adaptive_max_pool2d_aten():
    """Route ``torch.ops.aten.adaptive_max_pool2d`` (CUDA) to the Gems forward.

    The vendor XDNN wrapper on this stack returns uninitialized index memory
    for float16/float32/bfloat16 (the output values are correct, the indices
    are garbage), so a ``adaptive_max_pool2d_backward`` whose reference
    consumes those indices either faults or disagrees with ATen.  The Gems
    ``adaptive_max_pool2d`` forward kernel computes ATen-exact flat spatial
    indices (``h * W + w``, same window math and tie-break); registering it
    for the CUDA dispatch key makes every ``torch.ops.aten.adaptive_max_pool2d``
    call produce valid indices, both inside and outside ``use_gems()``
    scopes.  The registration is process-global (the Library handle is held in
    this module); the ``use_gems()`` registrar re-registers the same function
    for the same key, so there is no conflict.
    """
    global _adaptive_max_pool2d_aten_lib
    if _adaptive_max_pool2d_aten_lib is not None:
        return

    import torch

    from .adaptive_max_pool2d import adaptive_max_pool2d as _gems_fwd

    _adaptive_max_pool2d_aten_lib = torch.library.Library("aten", "IMPL")
    _adaptive_max_pool2d_aten_lib.impl(
        "adaptive_max_pool2d", _gems_fwd, "CUDA", allow_override=True
    )


_patch_adaptive_max_pool2d_aten()
