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

from .__ilshift__ import __ilshift__
from ._masked_scale import _masked_scale
from ._resize_output import _resize_output
from .abs import abs, abs_
from .acos import acos
from .acosh import acosh, acosh_
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_, addcdiv_out
from .addcmul import addcmul, addcmul_, addcmul_out
from .all import all, all_dim, all_dims
from .angle import angle
from .arange import arange, arange_start
from .arccos import arccos, arccos_
from .arcsin import arcsin, arcsin_, arcsin_out
from .arctan_ import arctan, arctan_
from .argmax import argmax
from .argmin import argmin
from .as_strided_copy import as_strided_copy, as_strided_copy_out
from .asin import asin, asin_
from .asinh import asinh, asinh_out
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
from .baddbmm import baddbmm, baddbmm_out
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
from .cat import cat
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_min, clamp_min_, clamp_tensor, clamp_tensor_
from .clamp_max import clamp_max
from .clip import clip, clip_
from .copy import copy, copy_
from .copysign import copysign, copysign_out
from .cos import cos, cos_
from .cosh import cosh, cosh_, cosh_out
from .count_nonzero import count_nonzero
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diagonal_copy import diagonal_copy
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .erfinv_ import erfinv, erfinv_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .expm1 import expm1, expm1_, expm1_out
from .exponential_ import exponential_
from .fill import fill_scalar, fill_scalar_, fill_scalar_out, fill_tensor, fill_tensor_
from .flash_api import mha_fwd, mha_varlan_fwd
from .flip import flip
from .floor import floor, floor_out
from .fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_
from .fmod_ import fmod_
from .frac_ import frac, frac_
from .full import full
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu, glu_backward
from .greater import greater, greater_out, greater_scalar, greater_scalar_out
from .gt import gt, gt_scalar, gt_scalar_, gt_tensor_
from .hstack import hstack
from .index import index
from .index_add import index_add, index_add_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_select import index_select
from .is_nonzero import is_nonzero
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .isneginf import isneginf, isneginf_out
from .kron import kron
from .le import le, le_scalar
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .less_equal import less_equal, less_equal_scalar
from .lgamma_ import lgamma, lgamma_
from .log import log
from .log1p import log1p, log1p_out
from .log2 import log2, log2_
from .log10 import log10, log10_, log10_out
from .log_normal_ import log_normal_
from .log_sigmoid import log_sigmoid
from .logaddexp import logaddexp, logaddexp_out
from .logaddexp2 import logaddexp2, logaddexp2_out
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not, logical_not_
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logical_xor_ import logical_xor_
from .lt import lt, lt_scalar
from .lt_ import lt_, lt_scalar_
from .masked_fill import masked_fill, masked_fill_
from .masked_select import masked_select
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max import max, max_dim
from .maximum import maximum
from .mean import mean, mean_dim
from .minimum import minimum
from .mish import mish, mish_
from .mm import mm, mm_out
from .mse_loss import mse_loss
from .mul import mul, mul_
from .nan_to_num import nan_to_num
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .nextafter_ import nextafter_
from .normal import (
    normal_,
    normal_distribution,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .not_equal import not_equal, not_equal_scalar
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .rad2deg import rad2deg, rad2deg_
from .randn import randn
from .randn_like import randn_like
from .reciprocal import reciprocal, reciprocal_
from .relu import relu, relu_
from .remainder import remainder, remainder_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .rrelu_with_noise_functional import rrelu_with_noise_functional
from .rsqrt import rsqrt, rsqrt_
from .rsub import rsub_scalar, rsub_tensor
from .select_scatter import select_scatter
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .signbit import signbit, signbit_out
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinc import sinc, sinc_
from .sinh import sinh, sinh_
from .slice_backward import slice_backward
from .softmax import softmax, softmax_out
from .softplus import softplus, softplus_backward
from .special_airy_ai import special_airy_ai, special_airy_ai_out
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
from .sqrt import sqrt, sqrt_
from .square import square, square_, square_out
from .stack import stack
from .sub import sub, sub_
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_backward
from .threshold_ import threshold_
from .tile import tile
from .to import to_copy
from .trunc_ import trunc, trunc_
from .unique import _unique2
from .upsample_bicubic2d import upsample_bicubic2d
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from .vdot import vdot
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
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "__ilshift__",
    "_index_put_impl_",
    "_masked_scale",
    "_resize_output",
    "_unique2",
    "_upsample_bicubic2d_aa",
    "_upsample_bicubic2d_aa_backward",
    "abs",
    "abs_",
    "acos",
    "acosh",
    "acosh_",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcdiv_out",
    "addcmul",
    "addcmul_",
    "addcmul_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "angle",
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
    "asinh",
    "asinh_out",
    "atan",
    "atan2",
    "atan2_out",
    "atan_",
    "atanh",
    "atanh_",
    "baddbmm",
    "baddbmm_out",
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
    "cat",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "clip",
    "clip_",
    "copy",
    "copy_",
    "copysign",
    "copysign_out",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosh_out",
    "count_nonzero",
    "cumsum",
    "cumsum_out",
    "diag_embed",
    "diagonal_backward",
    "diagonal_copy",
    "div_mode",
    "div_mode_",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_backward",
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
    "exponential_",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flip",
    "floor",
    "floor_divide",
    "floor_divide_",
    "floor_out",
    "fmod_",
    "fmod_scalar",
    "fmod_scalar_",
    "fmod_tensor",
    "fmod_tensor_",
    "frac",
    "frac_",
    "full",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "glu",
    "glu_backward",
    "greater",
    "greater_out",
    "greater_scalar",
    "greater_scalar_out",
    "gt",
    "gt_scalar",
    "gt_scalar_",
    "gt_tensor_",
    "hstack",
    "index",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_select",
    "is_nonzero",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isneginf_out",
    "kron",
    "le",
    "le_scalar",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "less_equal",
    "less_equal_scalar",
    "lgamma",
    "lgamma_",
    "log",
    "log10",
    "log10_",
    "log10_out",
    "log1p",
    "log1p_out",
    "log2",
    "log2_",
    "log_normal_",
    "log_sigmoid",
    "logaddexp",
    "logaddexp2",
    "logaddexp2_out",
    "logaddexp_out",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "lt",
    "lt_",
    "lt_scalar",
    "lt_scalar_",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "matmul_bf16",
    "matmul_int8",
    "max",
    "max_dim",
    "maximum",
    "mean",
    "mean_dim",
    "mha_fwd",
    "mha_varlan_fwd",
    "minimum",
    "mish",
    "mish_",
    "mm",
    "mm_out",
    "mse_loss",
    "mul",
    "mul_",
    "nan_to_num",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "nextafter_",
    "normal_",
    "normal_distribution",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "not_equal",
    "not_equal_scalar",
    "polar",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "rad2deg",
    "rad2deg_",
    "randn",
    "randn_like",
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
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "rrelu_with_noise_functional",
    "rsqrt",
    "rsqrt_",
    "rsub_scalar",
    "rsub_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "ScaleDotProductAttention",
    "select_scatter",
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
    "softmax",
    "softmax_out",
    "softplus",
    "softplus_backward",
    "special_airy_ai",
    "special_airy_ai_out",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_hermite_polynomial_h",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "square_out",
    "stack",
    "sub",
    "sub_",
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
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc",
    "trunc_",
    "upsample_bicubic2d",
    "vdot",
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
    "zero_",
    "zeros",
    "zeros_like",
]
