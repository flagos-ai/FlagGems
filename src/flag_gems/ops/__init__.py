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

from flag_gems.ops.__ilshift__ import __ilshift__
from flag_gems.ops.__irshift__ import __irshift__
from flag_gems.ops.__lshift__ import __lshift__
from flag_gems.ops.__xor__ import (  # noqa: F401
    xor,
    xor_,
    xor_scalar,
    xor_scalar_,
    xor_scalar_tensor,
)
from flag_gems.ops._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from flag_gems.ops._batch_norm_no_update import _batch_norm_no_update
from flag_gems.ops._conj import _conj
from flag_gems.ops._embedding_bag_dense_backward import _embedding_bag_dense_backward
from flag_gems.ops._euclidean_dist import _euclidean_dist
from flag_gems.ops._functional_sym_constrain_range import (
    _functional_sym_constrain_range,
)
from flag_gems.ops._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from flag_gems.ops._fused_adam import _fused_adam, _fused_adam_
from flag_gems.ops._is_all_true import _is_all_true
from flag_gems.ops._jagged_to_padded_dense_forward import (
    _jagged_to_padded_dense_forward,
)
from flag_gems.ops._linalg_eigvals import _linalg_eigvals
from flag_gems.ops._masked_scale import _masked_scale
from flag_gems.ops._nested_view_from_buffer_copy import _nested_view_from_buffer_copy
from flag_gems.ops._pdist_backward import _pdist_backward
from flag_gems.ops._prelu_kernel import _prelu_kernel
from flag_gems.ops._prelu_kernel_backward import _prelu_kernel_backward
from flag_gems.ops._resize_output import _resize_output
from flag_gems.ops._safe_softmax import _safe_softmax
from flag_gems.ops._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from flag_gems.ops._sparse_semi_structured_mm import _sparse_semi_structured_mm
from flag_gems.ops._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from flag_gems.ops._thnn_fused_lstm_cell_backward_impl import (
    _thnn_fused_lstm_cell_backward_impl,
)
from flag_gems.ops._unsafe_masked_index import _unsafe_masked_index
from flag_gems.ops._unsafe_masked_index_put_accumulate import (
    _unsafe_masked_index_put_accumulate,
)
from flag_gems.ops._unsafe_view import _unsafe_view
from flag_gems.ops._upsample_bilinear2d_aa import _upsample_bilinear2d_aa
from flag_gems.ops._upsample_nearest_exact1d import _upsample_nearest_exact1d
from flag_gems.ops._upsample_nearest_exact2d_backward import (
    _upsample_nearest_exact2d_backward,
)
from flag_gems.ops._upsample_nearest_exact3d import _upsample_nearest_exact3d
from flag_gems.ops.abs import abs, abs_
from flag_gems.ops.absolute import absolute
from flag_gems.ops.acos import acos
from flag_gems.ops.acosh import acosh, acosh_
from flag_gems.ops.adaptive_avg_pool2d import adaptive_avg_pool2d
from flag_gems.ops.adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from flag_gems.ops.add import add, add_
from flag_gems.ops.addcdiv import addcdiv, addcdiv_, addcdiv_out
from flag_gems.ops.addcmul import addcmul, addcmul_, addcmul_out
from flag_gems.ops.addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from flag_gems.ops.addmm_ import addmm_
from flag_gems.ops.addmv import addmv, addmv_out
from flag_gems.ops.addr import addr
from flag_gems.ops.affine_grid_generator import affine_grid_generator
from flag_gems.ops.alias_copy import alias_copy, alias_copy_out
from flag_gems.ops.all import all, all_dim, all_dims
from flag_gems.ops.alpha_dropout import alpha_dropout
from flag_gems.ops.amax import amax
from flag_gems.ops.amin import amin, amin_
from flag_gems.ops.aminmax import aminmax
from flag_gems.ops.angle import angle
from flag_gems.ops.any import any, any_dim, any_dims
from flag_gems.ops.arange import arange, arange_start
from flag_gems.ops.arccos import arccos, arccos_
from flag_gems.ops.arcsin import arcsin, arcsin_, arcsin_out
from flag_gems.ops.arcsinh import arcsinh, arcsinh_out
from flag_gems.ops.arcsinh_ import arcsinh_
from flag_gems.ops.arctan_ import arctan, arctan_
from flag_gems.ops.arctanh_ import arctanh_
from flag_gems.ops.argmax import argmax
from flag_gems.ops.argmin import argmin
from flag_gems.ops.argsort import argsort
from flag_gems.ops.as_strided_copy import as_strided_copy, as_strided_copy_out
from flag_gems.ops.asin import asin, asin_
from flag_gems.ops.asinh import asinh, asinh_out
from flag_gems.ops.asinh_ import asinh_
from flag_gems.ops.assert_async import _assert_async
from flag_gems.ops.atan import atan, atan_
from flag_gems.ops.atan2 import atan2, atan2_out
from flag_gems.ops.atanh import atanh, atanh_
from flag_gems.ops.attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from flag_gems.ops.avg_pool2d import avg_pool2d, avg_pool2d_backward
from flag_gems.ops.avg_pool3d import avg_pool3d, avg_pool3d_backward
from flag_gems.ops.baddbmm import baddbmm, baddbmm_out
from flag_gems.ops.batch_norm import batch_norm, batch_norm_backward
from flag_gems.ops.bernoulli import bernoulli
from flag_gems.ops.bernoulli_ import bernoulli_
from flag_gems.ops.binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits,
)
from flag_gems.ops.bincount import bincount
from flag_gems.ops.bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from flag_gems.ops.bitwise_left_shift import bitwise_left_shift, bitwise_left_shift_
from flag_gems.ops.bitwise_not import bitwise_not, bitwise_not_
from flag_gems.ops.bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from flag_gems.ops.bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_
from flag_gems.ops.bitwise_xor import (
    bitwise_xor_scalar,
    bitwise_xor_scalar_,
    bitwise_xor_scalar_tensor,
    bitwise_xor_tensor,
    bitwise_xor_tensor_,
)
from flag_gems.ops.bmm import bmm, bmm_out
from flag_gems.ops.broadcast_tensors import broadcast_tensors
from flag_gems.ops.broadcast_to import broadcast_to
from flag_gems.ops.bucketize import bucketize
from flag_gems.ops.cat import cat, cat_out
from flag_gems.ops.cauchy import cauchy, cauchy_
from flag_gems.ops.cdist_backward import _cdist_backward
from flag_gems.ops.ceil import ceil, ceil_, ceil_out
from flag_gems.ops.celu import celu, celu_
from flag_gems.ops.channel_shuffle import channel_shuffle
from flag_gems.ops.clamp import (
    clamp,
    clamp_,
    clamp_min,
    clamp_min_,
    clamp_tensor,
    clamp_tensor_,
)
from flag_gems.ops.clamp_max import clamp_max, clamp_max_  # noqa: F401
from flag_gems.ops.clip import clip, clip_
from flag_gems.ops.col2im import col2im
from flag_gems.ops.concat import concat
from flag_gems.ops.concatenate import concatenate
from flag_gems.ops.conj_physical import conj_physical
from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.conv1d import conv1d
from flag_gems.ops.conv2d import conv2d
from flag_gems.ops.conv3d import conv3d
from flag_gems.ops.conv_depthwise2d import _conv_depthwise2d
from flag_gems.ops.conv_transpose1d import conv_transpose1d
from flag_gems.ops.conv_transpose2d import conv_transpose2d
from flag_gems.ops.copy import copy, copy_
from flag_gems.ops.copysign import copysign, copysign_out
from flag_gems.ops.copysign_ import copysign_
from flag_gems.ops.cos import cos, cos_
from flag_gems.ops.cosh import cosh, cosh_, cosh_out
from flag_gems.ops.count_nonzero import count_nonzero
from flag_gems.ops.ctc_loss import ctc_loss
from flag_gems.ops.cudnn_batch_norm_backward import cudnn_batch_norm_backward
from flag_gems.ops.cudnn_convolution import cudnn_convolution
from flag_gems.ops.cummax import cummax
from flag_gems.ops.cummin import cummin
from flag_gems.ops.cumprod import cumprod, cumprod_
from flag_gems.ops.cumsum import cumsum, cumsum_out, normed_cumsum
from flag_gems.ops.deg2rad import deg2rad, deg2rad_, deg2rad_out
from flag_gems.ops.dequantize import dequantize
from flag_gems.ops.diag import diag
from flag_gems.ops.diag_embed import diag_embed
from flag_gems.ops.diagonal import diagonal_backward
from flag_gems.ops.diagonal_copy import diagonal_copy
from flag_gems.ops.diagonal_scatter import diagonal_scatter
from flag_gems.ops.diff import diff
from flag_gems.ops.digamma_ import digamma, digamma_
from flag_gems.ops.div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from flag_gems.ops.dot import dot
from flag_gems.ops.dropout import dropout, dropout_backward
from flag_gems.ops.elu import elu, elu_, elu_backward
from flag_gems.ops.embedding import embedding, embedding_backward
from flag_gems.ops.embedding_dense_backward import embedding_dense_backward
from flag_gems.ops.empty import empty
from flag_gems.ops.eq import eq, eq_scalar, equal
from flag_gems.ops.erf import erf, erf_
from flag_gems.ops.erfinv_ import erfinv, erfinv_
from flag_gems.ops.exp import exp, exp_, exp_out
from flag_gems.ops.exp2 import exp2, exp2_
from flag_gems.ops.expand import expand, expand_
from flag_gems.ops.expm1 import expm1, expm1_, expm1_out
from flag_gems.ops.exponential_ import exponential_
from flag_gems.ops.eye import eye
from flag_gems.ops.eye_m import eye_m
from flag_gems.ops.feature_dropout import feature_dropout, feature_dropout_
from flag_gems.ops.fft import fft
from flag_gems.ops.fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from flag_gems.ops.fix import fix
from flag_gems.ops.flash_attention_backward import (
    efficient_attention_backward,
    flash_attention_backward,
    scaled_dot_product_cudnn_attention_backward,
    scaled_dot_product_efficient_attention_backward,
    scaled_dot_product_flash_attention_backward,
)
from flag_gems.ops.flip import flip
from flag_gems.ops.floor import floor, floor_out
from flag_gems.ops.floor_ import floor_
from flag_gems.ops.fmax import fmax, fmax_out
from flag_gems.ops.fmin import fmin, fmin_out
from flag_gems.ops.fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from flag_gems.ops.fmod_ import fmod_
from flag_gems.ops.fp8_matmul import fp8_matmul
from flag_gems.ops.fp8_mqa_logits import fp8_mqa_logits
from flag_gems.ops.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from flag_gems.ops.frac_ import frac, frac_
from flag_gems.ops.fractional_max_pool2d import (
    fractional_max_pool2d,
    fractional_max_pool2d_backward,
)
from flag_gems.ops.full import full
from flag_gems.ops.full_like import full_like
from flag_gems.ops.gather import gather, gather_backward
from flag_gems.ops.gather_block_quantized import gather_block_quantized
from flag_gems.ops.gcd import gcd, gcd_out
from flag_gems.ops.gcd_ import gcd_  # noqa: F401
from flag_gems.ops.ge import ge, ge_scalar
from flag_gems.ops.gelu import gelu, gelu_, gelu_backward
from flag_gems.ops.geometric import geometric, geometric_
from flag_gems.ops.get_paged_mqa_logits_metadata import get_paged_mqa_logits_metadata
from flag_gems.ops.get_scheduler_metadata import get_scheduler_metadata
from flag_gems.ops.glu import glu, glu_backward
from flag_gems.ops.greater import (
    greater,
    greater_out,
    greater_scalar,
    greater_scalar_out,
)
from flag_gems.ops.greater_equal import greater_equal_
from flag_gems.ops.grid_sample import grid_sample
from flag_gems.ops.group_gemm import group_mm
from flag_gems.ops.groupnorm import group_norm, group_norm_backward
from flag_gems.ops.gt import gt, gt_scalar, gt_scalar_, gt_tensor_
from flag_gems.ops.hadamard_transform import (
    hadamard_transform,
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)
from flag_gems.ops.hardsigmoid import hardsigmoid, hardsigmoid_out
from flag_gems.ops.hardswish_ import hardswish_
from flag_gems.ops.histc import histc
from flag_gems.ops.hstack import hstack
from flag_gems.ops.hypot import hypot, hypot_out
from flag_gems.ops.i0 import i0, i0_out
from flag_gems.ops.i0_ import i0_
from flag_gems.ops.igammac_ import igammac_
from flag_gems.ops.im2col import im2col
from flag_gems.ops.index import index
from flag_gems.ops.index_add import index_add, index_add_
from flag_gems.ops.index_copy_ import index_copy, index_copy_
from flag_gems.ops.index_put import _index_put_impl_, index_put, index_put_
from flag_gems.ops.index_reduce import index_reduce_
from flag_gems.ops.index_select import index_select
from flag_gems.ops.index_select_backward import index_select_backward
from flag_gems.ops.is_nonzero import is_nonzero
from flag_gems.ops.isclose import allclose, isclose
from flag_gems.ops.isfinite import isfinite
from flag_gems.ops.isin import isin
from flag_gems.ops.isinf import isinf
from flag_gems.ops.isnan import isnan
from flag_gems.ops.isneginf import isneginf, isneginf_out
from flag_gems.ops.kron import kron
from flag_gems.ops.kthvalue import kthvalue
from flag_gems.ops.layernorm import layer_norm, layer_norm_backward
from flag_gems.ops.lcm import lcm, lcm_
from flag_gems.ops.ldl_factor_ex import ldl_factor_ex
from flag_gems.ops.le import le, le_scalar
from flag_gems.ops.leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from flag_gems.ops.lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from flag_gems.ops.less_equal import less_equal, less_equal_scalar
from flag_gems.ops.lgamma_ import lgamma, lgamma_
from flag_gems.ops.lift import lift, lift_out
from flag_gems.ops.lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from flag_gems.ops.linalg_cholesky import linalg_cholesky
from flag_gems.ops.linalg_ldl_factor import ldl_factor
from flag_gems.ops.linalg_slogdet import linalg_slogdet
from flag_gems.ops.linalg_svdvals import linalg_svdvals
from flag_gems.ops.linear import linear
from flag_gems.ops.linear_backward import linear_backward
from flag_gems.ops.linspace import linspace
from flag_gems.ops.log import log
from flag_gems.ops.log1p import log1p, log1p_out
from flag_gems.ops.log1p_ import log1p_
from flag_gems.ops.log2 import log2, log2_
from flag_gems.ops.log10 import log10, log10_, log10_out
from flag_gems.ops.log_normal_ import log_normal_
from flag_gems.ops.log_sigmoid import log_sigmoid
from flag_gems.ops.log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from flag_gems.ops.logaddexp import logaddexp, logaddexp_out
from flag_gems.ops.logaddexp2 import logaddexp2, logaddexp2_out
from flag_gems.ops.logcumsumexp import logcumsumexp, logcumsumexp_out
from flag_gems.ops.logical_and import logical_and, logical_and_
from flag_gems.ops.logical_not import logical_not, logical_not_
from flag_gems.ops.logical_or import logical_or, logical_or_
from flag_gems.ops.logical_xor import logical_xor
from flag_gems.ops.logical_xor_ import logical_xor_
from flag_gems.ops.logit import logit, logit_out
from flag_gems.ops.logit_ import logit_
from flag_gems.ops.logspace import logspace
from flag_gems.ops.logsumexp import logsumexp
from flag_gems.ops.lt import lt, lt_scalar
from flag_gems.ops.lt_ import lt_, lt_scalar_
from flag_gems.ops.margin_ranking_loss import margin_ranking_loss
from flag_gems.ops.masked_fill import masked_fill, masked_fill_
from flag_gems.ops.masked_scatter import masked_scatter, masked_scatter_
from flag_gems.ops.masked_scatter_backward import masked_scatter_backward
from flag_gems.ops.masked_select import masked_select
from flag_gems.ops.max import max, max_dim
from flag_gems.ops.max_pool2d_with_indices import (
    max_pool2d_backward,
    max_pool2d_with_indices,
)
from flag_gems.ops.max_pool3d_with_indices import (
    max_pool3d_backward,
    max_pool3d_with_indices,
)
from flag_gems.ops.max_unpool2d import max_unpool2d
from flag_gems.ops.maximum import maximum
from flag_gems.ops.mean import mean, mean_dim
from flag_gems.ops.median import median, median_dim, median_dim_values, median_out
from flag_gems.ops.min import min, min_dim
from flag_gems.ops.minimum import minimum
from flag_gems.ops.mish import mish, mish_
from flag_gems.ops.mm import mm, mm_out, router_gemm
from flag_gems.ops.mode import mode
from flag_gems.ops.mse_loss import mse_loss
from flag_gems.ops.mse_loss_backward import mse_loss_backward
from flag_gems.ops.mul import mul, mul_
from flag_gems.ops.multinomial import multinomial
from flag_gems.ops.multiply_ import multiply_
from flag_gems.ops.mv import mv
from flag_gems.ops.mvlgamma_ import mvlgamma_
from flag_gems.ops.nan_to_num import nan_to_num
from flag_gems.ops.nanmedian import (
    nanmedian,
    nanmedian_dim,
    nanmedian_dim_values,
    nanmedian_out,
)
from flag_gems.ops.narrow import narrow
from flag_gems.ops.narrow_copy import narrow_copy
from flag_gems.ops.ne import ne, ne_scalar
from flag_gems.ops.neg import neg, neg_
from flag_gems.ops.negative import negative
from flag_gems.ops.new_full import new_full
from flag_gems.ops.new_ones import new_ones
from flag_gems.ops.nextafter import nextafter, nextafter_
from flag_gems.ops.nll_loss_nd import nll_loss_nd_backward, nll_loss_nd_forward
from flag_gems.ops.nllloss import (
    nll_loss2d_backward,
    nll_loss2d_forward,
    nll_loss_backward,
    nll_loss_forward,
)
from flag_gems.ops.nonzero import nonzero
from flag_gems.ops.nonzero_numpy import nonzero_numpy
from flag_gems.ops.norm import norm, norm_scalar, norm_scalaropt_dim
from flag_gems.ops.normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from flag_gems.ops.not_equal import not_equal, not_equal_scalar
from flag_gems.ops.one_hot import one_hot
from flag_gems.ops.ones import ones
from flag_gems.ops.ones_like import ones_like
from flag_gems.ops.pad import constant_pad_nd, pad
from flag_gems.ops.pdist import pdist
from flag_gems.ops.per_token_group_quant_fp8 import (
    SUPPORTED_FP8_DTYPE,
    per_token_group_quant_fp8,
)
from flag_gems.ops.permute_copy import permute_copy
from flag_gems.ops.pixel_shuffle import pixel_shuffle
from flag_gems.ops.pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out
from flag_gems.ops.poisson import poisson
from flag_gems.ops.polar import polar
from flag_gems.ops.pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from flag_gems.ops.prelu import prelu
from flag_gems.ops.prod import prod, prod_dim
from flag_gems.ops.quantile import quantile
from flag_gems.ops.rad2deg import rad2deg, rad2deg_
from flag_gems.ops.rand import rand
from flag_gems.ops.rand_like import rand_like
from flag_gems.ops.randint import randint
from flag_gems.ops.randint_like import randint_like
from flag_gems.ops.randn import randn
from flag_gems.ops.randn_like import randn_like
from flag_gems.ops.randperm import randperm
from flag_gems.ops.range import range
from flag_gems.ops.reciprocal import reciprocal, reciprocal_
from flag_gems.ops.reflection_pad1d import reflection_pad1d, reflection_pad1d_out
from flag_gems.ops.reflection_pad1d_backward import reflection_pad1d_backward
from flag_gems.ops.reflection_pad2d import reflection_pad2d, reflection_pad2d_out
from flag_gems.ops.reflection_pad3d import reflection_pad3d, reflection_pad3d_out
from flag_gems.ops.reflection_pad3d_backward import reflection_pad3d_backward
from flag_gems.ops.relu import relu, relu_
from flag_gems.ops.relu6 import relu6
from flag_gems.ops.remainder import remainder, remainder_
from flag_gems.ops.renorm import renorm
from flag_gems.ops.renorm_ import renorm_
from flag_gems.ops.repeat import repeat
from flag_gems.ops.repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from flag_gems.ops.replication_pad1d import replication_pad1d, replication_pad1d_out
from flag_gems.ops.replication_pad3d import replication_pad3d
from flag_gems.ops.resize import resize, resize_
from flag_gems.ops.resize_as import resize_as, resize_as_
from flag_gems.ops.resolve_conj import resolve_conj
from flag_gems.ops.resolve_neg import resolve_neg
from flag_gems.ops.rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from flag_gems.ops.rnn_relu import rnn_relu
from flag_gems.ops.roll import roll
from flag_gems.ops.rot90 import rot90
from flag_gems.ops.round import round, round_, round_out
from flag_gems.ops.rrelu_with_noise_backward import rrelu_with_noise_backward
from flag_gems.ops.rrelu_with_noise_functional import rrelu_with_noise_functional
from flag_gems.ops.rsqrt import rsqrt, rsqrt_
from flag_gems.ops.rsub import rsub_scalar, rsub_tensor
from flag_gems.ops.scalar_tensor import scalar_tensor
from flag_gems.ops.scaled_grouped_mm import scaled_grouped_mm
from flag_gems.ops.scaled_mm import scaled_mm, scaled_mm_out
from flag_gems.ops.scaled_softmax import scaled_softmax_backward, scaled_softmax_forward
from flag_gems.ops.scatter import scatter, scatter_
from flag_gems.ops.scatter_add_ import scatter_add_
from flag_gems.ops.scatter_reduce import (
    scatter_reduce,
    scatter_reduce_,
    scatter_reduce_out,
)
from flag_gems.ops.searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from flag_gems.ops.segment_reduce import (
    _segment_reduce_backward,
    _segment_reduce_backward_out,
    segment_reduce,
    segment_reduce_out,
)
from flag_gems.ops.select_backward import select_backward
from flag_gems.ops.select_scatter import select_scatter
from flag_gems.ops.selu import selu
from flag_gems.ops.selu_ import selu_
from flag_gems.ops.sgn_ import sgn_
from flag_gems.ops.sigmoid import sigmoid, sigmoid_, sigmoid_backward
from flag_gems.ops.signbit import signbit, signbit_out
from flag_gems.ops.silu import silu, silu_, silu_backward
from flag_gems.ops.sin import sin, sin_
from flag_gems.ops.sinc import sinc, sinc_
from flag_gems.ops.sinh import sinh, sinh_
from flag_gems.ops.slice_backward import slice_backward
from flag_gems.ops.slice_scatter import slice_scatter
from flag_gems.ops.smooth_l1_loss import (
    smooth_l1_loss,
    smooth_l1_loss_backward,
    smooth_l1_loss_out,
)
from flag_gems.ops.soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from flag_gems.ops.soft_margin_loss_backward import soft_margin_loss_backward
from flag_gems.ops.softmax import (
    softmax,
    softmax_backward,
    softmax_backward_out,
    softmax_out,
)
from flag_gems.ops.softplus import softplus, softplus_backward
from flag_gems.ops.softshrink import softshrink, softshrink_out
from flag_gems.ops.sort import sort, sort_stable
from flag_gems.ops.special_airy_ai import special_airy_ai, special_airy_ai_out
from flag_gems.ops.special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from flag_gems.ops.special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from flag_gems.ops.special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from flag_gems.ops.special_digamma import special_digamma
from flag_gems.ops.special_erfinv import (
    special_erfinv,
    special_erfinv_,
    special_erfinv_out,
)
from flag_gems.ops.special_gammainc import special_gammainc
from flag_gems.ops.special_gammaln import special_gammaln, special_gammaln_out
from flag_gems.ops.special_hermite_polynomial_h import special_hermite_polynomial_h
from flag_gems.ops.special_i0e import special_i0e, special_i0e_out
from flag_gems.ops.special_i1 import special_i1, special_i1_out
from flag_gems.ops.special_log1p import special_log1p, special_log1p_out
from flag_gems.ops.special_log_softmax import special_log_softmax
from flag_gems.ops.special_logsumexp import special_logsumexp
from flag_gems.ops.special_modified_bessel_k0 import (
    special_modified_bessel_k0,
    special_modified_bessel_k0_out,
)
from flag_gems.ops.special_modified_bessel_k1 import (
    special_modified_bessel_k1,
    special_modified_bessel_k1_out,
)
from flag_gems.ops.special_scaled_modified_bessel_k1 import (
    special_scaled_modified_bessel_k1,
    special_scaled_modified_bessel_k1_out,
)
from flag_gems.ops.special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
)
from flag_gems.ops.special_shifted_chebyshev_polynomial_v import (
    special_shifted_chebyshev_polynomial_v,
)
from flag_gems.ops.special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from flag_gems.ops.special_sinc import special_sinc
from flag_gems.ops.special_xlog1py import special_xlog1py
from flag_gems.ops.split_with_sizes_copy import split_with_sizes_copy
from flag_gems.ops.sqrt import sqrt, sqrt_
from flag_gems.ops.square import square, square_, square_out
from flag_gems.ops.squeeze_copy import squeeze_copy
from flag_gems.ops.stack import stack
from flag_gems.ops.std import std
from flag_gems.ops.sub import sub, sub_
from flag_gems.ops.subtract_ import subtract_
from flag_gems.ops.sum import sum, sum_dim, sum_dim_out, sum_out
from flag_gems.ops.svd import svd
from flag_gems.ops.sym_stride import sym_stride
from flag_gems.ops.t_copy import t_copy, t_copy_out
from flag_gems.ops.tan import tan, tan_
from flag_gems.ops.tanh import tanh, tanh_, tanh_backward
from flag_gems.ops.tensor_split import tensor_split
from flag_gems.ops.threshold import threshold, threshold_backward
from flag_gems.ops.threshold_ import threshold_
from flag_gems.ops.tile import tile
from flag_gems.ops.to import to_copy
from flag_gems.ops.topk import topk
from flag_gems.ops.trace import trace
from flag_gems.ops.transpose import transpose
from flag_gems.ops.tril import tril, tril_, tril_out
from flag_gems.ops.triu import triu, triu_
from flag_gems.ops.trunc_ import trunc, trunc_
from flag_gems.ops.unbind import unbind
from flag_gems.ops.unbind_copy import unbind_copy
from flag_gems.ops.unfold_backward import unfold_backward
from flag_gems.ops.unfold_copy import unfold_copy
from flag_gems.ops.uniform import uniform_
from flag_gems.ops.unique import _unique2
from flag_gems.ops.unique_consecutive import unique_consecutive
from flag_gems.ops.unique_dim import unique_dim
from flag_gems.ops.unsqueeze import unsqueeze, unsqueeze_
from flag_gems.ops.upsample_bicubic2d import upsample_bicubic2d
from flag_gems.ops.upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from flag_gems.ops.upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from flag_gems.ops.upsample_linear1d import upsample_linear1d
from flag_gems.ops.upsample_linear1d_backward import upsample_linear1d_backward
from flag_gems.ops.upsample_nearest1d import upsample_nearest1d
from flag_gems.ops.upsample_nearest2d import upsample_nearest2d
from flag_gems.ops.upsample_nearest3d import upsample_nearest3d
from flag_gems.ops.upsample_trilinear3d import upsample_trilinear3d
from flag_gems.ops.var import var, var_correction, var_dim
from flag_gems.ops.var_mean import var_mean
from flag_gems.ops.vdot import vdot
from flag_gems.ops.vector_norm import vector_norm
from flag_gems.ops.view_copy import view_copy
from flag_gems.ops.vstack import vstack
from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
from flag_gems.ops.weightnorm import (
    weight_norm_interface,
    weight_norm_interface_backward,
)
from flag_gems.ops.where import (
    where_scalar_other,
    where_scalar_self,
    where_self,
    where_self_out,
)
from flag_gems.ops.xlogy import (
    xlogy,
    xlogy_out,
    xlogy_scalar_tensor,
    xlogy_scalar_tensor_out,
    xlogy_tensor_scalar,
    xlogy_tensor_scalar_out,
)
from flag_gems.ops.zero import zero, zero_out
from flag_gems.ops.zeros import zero_, zeros
from flag_gems.ops.zeros_like import zeros_like

__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "ScaleDotProductAttention",
    "__ilshift__",
    "__irshift__",
    "__lshift__",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_assert_async",
    "_batch_norm_no_update",
    "_cdist_backward",
    "_conj",
    "_conv_depthwise2d",
    "_embedding_bag_dense_backward",
    "_euclidean_dist",
    "_functional_sym_constrain_range",
    "_functional_sym_constrain_range_for_size",
    "_fused_adam",
    "_fused_adam_",
    "_index_put_impl_",
    "_is_all_true",
    "_jagged_to_padded_dense_forward",
    "_linalg_eigvals",
    "_masked_scale",
    "_nested_view_from_buffer_copy",
    "_pdist_backward",
    "_prelu_kernel",
    "_prelu_kernel_backward",
    "_resize_output",
    "_safe_softmax",
    "_scaled_dot_product_fused_attention_overrideable",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_sparse_semi_structured_mm",
    "_thnn_fused_lstm_cell",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unique2",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
    "_unsafe_view",
    "_upsample_bicubic2d_aa",
    "_upsample_bicubic2d_aa_backward",
    "_upsample_bilinear2d_aa",
    "_upsample_nearest_exact1d",
    "_upsample_nearest_exact2d_backward",
    "_upsample_nearest_exact3d",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "acosh",
    "acosh_",
    "adaptive_avg_pool2d",
    "adaptive_max_pool3d_backward",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcdiv_out",
    "addcmul",
    "addcmul_",
    "addcmul_out",
    "addmm",
    "addmm_",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv",
    "addmv_out",
    "addr",
    "affine_grid_generator",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "alpha_dropout",
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
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctan",
    "arctan_",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "as_strided_copy",
    "as_strided_copy_out",
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
    "avg_pool3d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_out",
    "batch_norm",
    "batch_norm_backward",
    "bernoulli",
    "bernoulli_",
    "binary_cross_entropy_with_logits",
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
    "broadcast_tensors",
    "broadcast_to",
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
    "channel_shuffle",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "col2im",
    "concat",
    "concatenate",
    "conj_physical",
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
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "ctc_loss",
    "cudnn_batch_norm_backward",
    "cudnn_convolution",
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
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "digamma_",
    "div_mode",
    "div_mode_",
    "dot",
    "dropout",
    "dropout_backward",
    "efficient_attention_backward",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
    "embedding_dense_backward",
    "empty",
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
    "expand",
    "expand_",
    "expm1",
    "expm1_",
    "expm1_out",
    "exponential_",
    "eye",
    "eye_m",
    "feature_dropout",
    "feature_dropout_",
    "fft",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "fix",
    "flash_attention_backward",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
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
    "fp8_matmul",
    "fp8_mqa_logits",
    "fp8_paged_mqa_logits",
    "frac",
    "frac_",
    "fractional_max_pool2d",
    "fractional_max_pool2d_backward",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "gather_block_quantized",
    "gcd",
    "gcd_",
    "gcd_out",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "geometric",
    "geometric_",
    "get_paged_mqa_logits_metadata",
    "get_scheduler_metadata",
    "glu",
    "glu_backward",
    "greater",
    "greater_equal_",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "grid_sample",
    "group_mm",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor_",
    "hadamard_transform",
    "hadamard_transform_12N",
    "hadamard_transform_20N",
    "hadamard_transform_28N",
    "hadamard_transform_40N",
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
    "igammac_",
    "im2col",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_reduce_",
    "index_select",
    "index_select_backward",
    "is_nonzero",
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
    "lcm",
    "lcm_",
    "ldl_factor",
    "ldl_factor_ex",
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
    "lift",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "lift_out",
    "linalg_cholesky",
    "linalg_slogdet",
    "linalg_svdvals",
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
    "log_normal_",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logaddexp",
    "logaddexp_out",
    "logaddexp2",
    "logaddexp2_out",
    "logcumsumexp",
    "logcumsumexp_out",
    "xlogy",
    "xlogy_out",
    "xlogy_tensor_scalar",
    "xlogy_tensor_scalar_out",
    "xlogy_scalar_tensor",
    "xlogy_scalar_tensor_out",
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
    "logsumexp",
    "lt",
    "lt_",
    "lt_scalar",
    "lt_scalar_",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_backward",
    "masked_scatter_",
    "masked_select",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "max_unpool2d",
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
    "mish",
    "mish_",
    "mm",
    "mm_out",
    "mode",
    "mse_loss",
    "mse_loss_backward",
    "mul",
    "mul_",
    "multinomial",
    "multiply_",
    "mv",
    "mvlgamma_",
    "nan_to_num",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "narrow",
    "narrow_copy",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "negative",
    "new_full",
    "new_ones",
    "nextafter",
    "nextafter_",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss_nd_backward",
    "nll_loss_nd_forward",
    "nonzero",
    "nonzero_numpy",
    "norm",
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
    "pad",
    "pdist",
    "per_token_group_quant_fp8",
    "permute_copy",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pixel_unshuffle_out",
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
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "range",
    "reciprocal",
    "reciprocal_",
    "reflection_pad1d",
    "reflection_pad1d_backward",
    "reflection_pad1d_out",
    "reflection_pad2d",
    "reflection_pad2d_out",
    "reflection_pad3d",
    "reflection_pad3d_backward",
    "reflection_pad3d_out",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad1d",
    "replication_pad1d_out",
    "replication_pad3d",
    "resize",
    "resize_",
    "resize_as",
    "resize_as_",
    "resolve_conj",
    "resolve_neg",
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "rnn_relu",
    "roll",
    "rot90",
    "round",
    "round_",
    "round_out",
    "router_gemm",
    "rrelu_with_noise_backward",
    "rrelu_with_noise_functional",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "scalar_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_cudnn_attention_backward",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_dot_product_flash_attention_backward",
    "scaled_grouped_mm",
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
    "segment_reduce",
    "segment_reduce_out",
    "select_backward",
    "select_scatter",
    "selu",
    "selu_",
    "sgn_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "signbit",
    "signbit_out",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "sinh",
    "sinh_",
    "slice_backward",
    "slice_scatter",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "smooth_l1_loss_out",
    "soft_margin_loss",
    "soft_margin_loss_backward",
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
    "special_airy_ai",
    "special_airy_ai_out",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_digamma",
    "special_erfinv",
    "special_erfinv_",
    "special_erfinv_out",
    "special_gammainc",
    "special_gammaln",
    "special_gammaln_out",
    "special_hermite_polynomial_h",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_log_softmax",
    "special_log1p",
    "special_log1p_out",
    "special_logsumexp",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_modified_bessel_k1",
    "special_modified_bessel_k1_out",
    "special_scaled_modified_bessel_k1",
    "special_scaled_modified_bessel_k1_out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_w",
    "special_sinc",
    "special_xlog1py",
    "split_with_sizes_copy",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "squeeze_copy",
    "stack",
    "std",
    "sub",
    "sub_",
    "subtract_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "svd",
    "sym_stride",
    "t_copy",
    "t_copy_out",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "tensor_split",
    "threshold",
    "threshold_",
    "threshold_backward",
    "tile",
    "to_copy",
    "topk",
    "trace",
    "transpose",
    "tril",
    "tril_",
    "tril_out",
    "triu",
    "triu_",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc",
    "trunc_",
    "unbind",
    "unbind_copy",
    "unfold_backward",
    "unfold_copy",
    "uniform_",
    "unique_consecutive",
    "unique_dim",
    "unsqueeze",
    "unsqueeze_",
    "upsample_bicubic2d",
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
    "w8a8_block_fp8_matmul",
    "weight_norm_interface",
    "weight_norm_interface_backward",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
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
