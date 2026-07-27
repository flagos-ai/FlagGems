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
from ._batch_norm_no_update import _batch_norm_no_update
from .affine_grid_generator import affine_grid_generator
from ._conj import _conj
from ._euclidean_dist import _euclidean_dist
from ._fused_adam import _fused_adam, _fused_adam_
from ._functional_sym_constrain_range import _functional_sym_constrain_range
from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._native_batch_norm_legit_functional import (
    _native_batch_norm_legit_functional,
)
from ._is_all_true import _is_all_true
from ._pdist_backward import _pdist_backward
from ._pdist_forward import _pdist_forward, pdist
from ._unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from ._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from ._upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from ._upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_, addcdiv_out
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
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
from .range import range
from .arccos import arccos, arccos_
from .arccosh import arccosh_
from .arcsin import arcsin, arcsin_, arcsin_out
from .arcsinh import arcsinh, arcsinh_, arcsinh_out
from .arctan import arctan, arctan_
from .argmax import argmax
from .argmin import argmin
from .argsort import argsort
from .as_strided_copy import as_strided_copy, as_strided_copy_out
from .asin import asin, asin_
from .atan import atan, atan_
from .atan2 import atan2, atan2_out
from .arctan2 import arctan2, arctan2_
from .atanh import atanh, atanh_
from .attention import (
    ScaleDotProductAttention,
    efficient_attention_backward,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
    scaled_dot_product_efficient_attention_backward,
)
from .adaptive_avg_pool2d import adaptive_avg_pool2d
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .avg_pool3d import avg_pool3d
from .avg_pool3d_backward import avg_pool3d_backward
from .baddbmm import baddbmm, baddbmm_out
from .batch_norm import batch_norm, batch_norm_backward
from .bernoulli import bernoulli
from .bernoulli_ import bernoulli_
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
from .bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_
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
from .cholesky_solve import cholesky_solve, cholesky_solve_out
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
from .col2im import col2im
from .concatenate import concatenate
from .contiguous import contiguous
from .conv1d import conv1d
from .conv2d import conv2d
from .conv_transpose1d import conv_transpose1d
from .conv_transpose2d import conv_transpose2d
from .conv3d import conv3d
from .conv_depthwise2d import _conv_depthwise2d
from .cudnn_batch_norm_backward import cudnn_batch_norm_backward
from .cudnn_convolution import cudnn_convolution
from .copy import copy, copy_
from .copysign import copysign, copysign_, copysign_out
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .cummax import cummax
from .cummin import cummin
from .cumprod import cumprod, cumprod_
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .deg2rad import deg2rad, deg2rad_, deg2rad_out
from .dequantize import dequantize
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diff import diff
from .digamma import digamma
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
from .embedding import embedding, embedding_backward, embedding_dense_backward
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
from .fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .log10 import log10, log10_, log10_out
from .fractional_max_pool2d import (
    fractional_max_pool2d,
    fractional_max_pool2d_backward,
)
from .fused_experts_impl import fused_experts_impl, outplace_fused_experts
from .fused_recurrent_gated_delta_rule_fwd import fused_recurrent_gated_delta_rule_fwd
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .get_paged_mqa_logits_metadata import get_paged_mqa_logits_metadata
from .grid_sample import grid_sample
from .gcd import gcd, gcd_, gcd_out
from .ge import ge, ge_scalar, greater_equal_
from .gelu import gelu, gelu_, gelu_backward
from .geometric import geometric, geometric_
from .get_scheduler_metadata import get_scheduler_metadata
from .glu import glu, glu_backward
from .greater import greater, greater_out, greater_scalar, greater_scalar_out
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar, gt_scalar_, gt_tensor_
from .hadamard_transform import hadamard_transform
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .histc import histc
from .hstack import hstack
from .im2col import im2col
from .index import index
from .index_add import index_add, index_add_
from .index_copy_ import index_copy_
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
from .kron import kron
from .kthvalue import kthvalue
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .less_equal import less_equal, less_equal_scalar
from .lift_fresh import lift_fresh
from .lift_fresh_copy import lift_fresh_copy
from .lcm import lcm, lcm_
from .lgamma import lgamma, lgamma_
from .mvlgamma_ import mvlgamma_
from .igamma_ import igamma_
from .igammac_ import igammac_
from .special_modified_bessel_k0 import (
    special_modified_bessel_k0,
    special_modified_bessel_k0_out,
)
from .special_bessel_j1 import special_bessel_j1
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_hermite_polynomial_h import special_hermite_polynomial_h
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)
from .special_shifted_chebyshev_polynomial_v import (
    special_shifted_chebyshev_polynomial_v,
)
from ._linalg_eigvals import _linalg_eigvals
from .linalg_ldl_factor import ldl_factor, ldl_factor_ex
from .linalg_ldl_factor_ex import ldl_factor_ex
from .linalg_ldl_solve import linalg_ldl_solve
from .linalg_lu_factor import linalg_lu_factor, linalg_lu_factor_out
from .linalg_slogdet import linalg_slogdet
from .linear_backward import linear_backward
from .linspace import linspace
from .log import log
from .log_ import log_
from .log1p import log1p, log1p_
from .log2 import log2, log2_
from .log_sigmoid import log_sigmoid
from .logcumsumexp import logcumsumexp, logcumsumexp_out
from .log_softmax import log_softmax, log_softmax_backward, log_softmax_backward_out, log_softmax_out
from .logaddexp2 import logaddexp2, logaddexp2_out
from .margin_ranking_loss import margin_ranking_loss
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
from .max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
    max_pool2d_with_indices_backward,
)
from .max_unpool3d import max_unpool3d
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out
from .mode import mode
from .moe_sum import moe_sum
from .mse_loss import mse_loss
from .mse_loss_backward import mse_loss_backward
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
from .nextafter import nextafter, nextafter_
from .nllloss import (
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
    nll_loss_nd_backward,
    nll_loss_nd_forward,
)
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .nonzero_static import nonzero_static, nonzero_static_out
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
from .replication_pad3d import replication_pad3d
from .replication_pad3d_backward import replication_pad3d_backward
from .reflection_pad1d import reflection_pad1d, reflection_pad1d_out
from .reflection_pad1d_backward import reflection_pad1d_backward
from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from .reflection_pad2d_backward import reflection_pad2d_backward
from .reflection_pad3d import reflection_pad3d, reflection_pad3d_out
from .reflection_pad3d_backward import reflection_pad3d_backward
from .relu import relu, relu_
from .relu6 import relu6
from .renorm import renorm, renorm_
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resize import resize, resize_
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from ._fused_rms_norm import _fused_rms_norm
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .te_rmsnorm import te_rmsnorm_bwd
from .rnn_relu import rnn_relu
from .rot90 import rot90
from .round import round, round_, round_out
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub, rsub_scalar, rsub_tensor
from .safe_softmax import _safe_softmax
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
from .signbit import signbit, signbit_out
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinc import sinc, sinc_, special_sinc
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .split_with_sizes_copy import split_with_sizes_copy
from .squeeze_copy import squeeze_copy
from .unbind_copy import unbind_copy
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .soft_margin_loss_backward import soft_margin_loss_backward
from .softmax import softmax, softmax_backward, softmax_backward_out, softmax_out
from .softplus import softplus, softplus_backward
from .softshrink import softshrink, softshrink_out
from .smooth_l1_loss import (
    smooth_l1_loss,
    smooth_l1_loss_backward,
    smooth_l1_loss_out,
)
from .sort import sort, sort_stable
from .special_log_softmax import special_log_softmax
from .special_logsumexp import special_logsumexp
from .special_gammainc import special_gammainc
from .special_gammaln import special_gammaln, special_gammaln_out
from .special_digamma import special_digamma
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
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
from .unique_consecutive import unique_consecutive
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
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
from .weightnorm import weight_norm_interface, weight_norm_interface_backward
from .weight_norm import _weight_norm
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
from .fmax import fmax, fmax_out
from .special_erfinv import special_erfinv, special_erfinv_, special_erfinv_out

__all__ = [
    "_amp_foreach_non_finite_check_and_unscale_",
    "_batch_norm_no_update",
    "_conj",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_native_batch_norm_legit_functional",
    "_euclidean_dist",
    "_fused_adam",
    "_fused_adam_",
    "_cdist_backward",
    "_is_all_true",
    "_scaled_dot_product_fused_attention_overrideable",
    "_thnn_fused_lstm_cell_backward_impl",
    "_upsample_nearest_exact2d_backward",
    "_conv_depthwise2d",
    "_safe_softmax",
    "digamma",
    "digamma_",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "soft_margin_loss_backward",
    "special_log_softmax",
    "special_logsumexp",
    "special_gammainc",
    "special_gammaln",
    "special_gammaln_out",
    "special_digamma",
    "softshrink",
    "softshrink_out",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "_pdist_backward",
    "_pdist_forward",
    "pdist",
    "_unsafe_masked_index_put_accumulate",
    "_unique2",
    "unique_consecutive",
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
    "arctan2",
    "arctan2_",
    "argmax",
    "argmin",
    "argsort",
    "as_strided_copy",
    "as_strided_copy_out",
    "asin",
    "asin_",
    "atan",
    "atan_",
    "atan2",
    "atan2_out",
    "adaptive_avg_pool2d",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_out",
    "batch_norm",
    "batch_norm_backward",
    "bernoulli",
    "bernoulli_",
    "binary_cross_entropy_with_logits",
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
    "bmm",
    "bmm_out",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cat_out",
    "ceil",
    "cholesky_solve",
    "cholesky_solve_out",
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
    "col2im",
    "concatenate",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "copy",
    "copy_",
    "copysign",
    "copysign_",
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
    "dequantize",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "diff",
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
    "embedding_dense_backward",
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
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "floor_divide",
    "floor_divide_",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "fmax",
    "fmax_out",
    "special_erfinv",
    "special_erfinv_",
    "special_erfinv_out",
    "fused_experts_impl",
    "outplace_fused_experts",
    "fused_recurrent_gated_delta_rule_fwd",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "gcd",
    "gcd_",
    "gcd_out",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "get_paged_mqa_logits_metadata",
    "get_scheduler_metadata",
    "glu",
    "glu_backward",
    "greater",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "greater_equal_",
    "grid_sample",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor_",
    "hstack",
    "hadamard_transform",
    "hardsigmoid",
    "hardsigmoid_out",
    "histc",
    "im2col",
    "index",
    "index_add",
    "index_add_",
    "index_copy_",
    "index_put",
    "index_put_",
    "_index_put_impl_",
    "index_reduce_",
    "index_select",
    "index_select_backward",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "kron",
    "kthvalue",
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
    "lcm",
    "lcm_",
    "lgamma",
    "lgamma_",
    "mvlgamma_",
    "igammac_",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_bessel_j1",
    "special_i0e",
    "special_i0e_out",
    "special_legendre_polynomial_p",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_hermite_polynomial_h",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "special_shifted_chebyshev_polynomial_v",
    "_linalg_eigvals",
    "ldl_factor",
    "ldl_factor_ex",
    "linalg_ldl_solve",
    "linalg_lu_factor",
    "linalg_lu_factor_out",
    "linalg_slogdet",
    "linear_backward",
    "linspace",
    "log",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_sigmoid",
    "logcumsumexp",
    "logcumsumexp_out",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
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
    "margin_ranking_loss",
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
    "max_pool2d_with_indices_backward",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mm_out",
    "mode",
    "moe_sum",
    "mse_loss",
    "mse_loss_backward",
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
    "nll_loss_nd_backward",
    "nll_loss_nd_forward",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nonzero",
    "nonzero_numpy",
    "norm",
    "norm_scalar",
    "norm_scalaropt_dim",
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
    "reflection_pad1d_backward",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "reflection_pad2d_backward",
    "reflection_pad3d",
    "reflection_pad3d_out",
    "reflection_pad3d_backward",
    "relu",
    "relu_",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
    "rrelu_with_noise_backward",
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
    "te_rmsnorm_bwd",
    "rnn_relu",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "rsub_scalar",
    "rsub_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_mm",
    "scaled_mm_out",
    "scaled_softmax_backward",
    "scaled_softmax_forward",
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
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "segment_reduce",
    "segment_reduce_out",
    "select_backward",
    "select_scatter",
    "selu",
    "selu_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "signbit",
    "signbit_out",
    "sgn",
    "sgn_out",
    "sgn_",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "special_sinc",
    "slice_backward",
    "slice_scatter",
    "softmax",
    "softmax_backward",
    "softmax_backward_out",
    "softmax_out",
    "softplus",
    "softplus_backward",
    "sort",
    "sort_stable",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
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
    "upsample_linear1d_backward",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_trilinear3d",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vdot",
    "vector_norm",
    "view_copy",
    "vstack",
    "_weight_norm",
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
