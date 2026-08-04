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

from .. import _install_typed_ptr_device_patch
from ._linalg_eigvals import _linalg_eigvals
from ._safe_softmax import _safe_softmax
from ._sparse_semi_structured_mm import _sparse_semi_structured_mm
from ._upsample_nearest_exact1d import _upsample_nearest_exact1d
from .abs import abs, abs_
from .add import add, add_
from .addmm import addmm, addmm_out
from .amax import amax, amax_out
from .amin import amin, amin_out
from .aminmax import aminmax, aminmax_out
from .angle import angle
from .arcsinh import arcsinh, arcsinh_out
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool3d import avg_pool3d, avg_pool3d_backward
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import (
    bitwise_left_shift,
    bitwise_left_shift_,
    bitwise_left_shift_out,
)
from .bitwise_right_shift import (
    bitwise_right_shift,
    bitwise_right_shift_,
    bitwise_right_shift_out,
)
from .cat import cat, cat_out
from .clamp import (
    clamp,
    clamp_,
    clamp_min,
    clamp_min_,
    clamp_min_out,
    clamp_tensor,
    clamp_tensor_,
)
from .concatenate import concatenate
from .conj_physical import conj_physical
from .conv2d import conv2d
from .cos import cos, cos_
from .count_nonzero import count_nonzero
from .ctc_loss import ctc_loss
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag_embed import diag_embed
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
from .dropout import dropout, dropout_backward
from .embedding import embedding, embedding_backward
from .empty import empty
from .eq import eq, eq_scalar, equal
from .exponential_ import exponential_
from .fft import fft, fft_c2c
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .fix import fix, fix_out
from .flash_attention_backward import (
    efficient_attention_backward,
    flash_attention_backward,
    scaled_dot_product_cudnn_attention_backward,
    scaled_dot_product_efficient_attention_backward,
    scaled_dot_product_flash_attention_backward,
)
from .gather import gather, gather_backward
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .group_gemm import group_mm
from .hypot import hypot, hypot_out
from .i0 import i0, i0_out
from .i0_ import i0_
from .index_add import index_add, index_add_
from .index_put import index_put, index_put_
from .index_reduce import index_reduce_
from .index_select import index_select
from .isin import isin
from .isnan import isnan
from .layernorm import layer_norm, layer_norm_backward
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linspace import linspace
from .log_softmax import log_softmax, log_softmax_backward
from .logaddexp import logaddexp, logaddexp_out
from .logical_and import logical_and
from .logical_or import logical_or, logical_or_
from .margin_ranking_loss import margin_ranking_loss
from .masked_select import masked_select
from .max_pool3d_with_indices import max_pool3d_backward, max_pool3d_with_indices
from .mean import mean, mean_dim
from .median import median, median_dim, median_dim_values, median_out
from .mul import mul, mul_
from .multinomial import multinomial
from .multiply_ import multiply_
from .mv import mv
from .neg import neg, neg_
from .nonzero import nonzero
from .one_hot import one_hot
from .pad import constant_pad_nd, pad
from .polar import polar
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu import prelu
from .quantile import quantile
from .randperm import randperm
from .reflection_pad2d import reflection_pad2d
from .reflection_pad3d_backward import reflection_pad3d_backward
from .renorm_ import renorm_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .scaled_grouped_mm import scaled_grouped_mm
from .scatter import scatter, scatter_
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .select_backward import select_backward
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .sinc import sinc, sinc_
from .soft_margin_loss import soft_margin_loss, soft_margin_loss_out
from .softmax import softmax, softmax_backward
from .sort import sort, sort_stable
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_gammainc import special_gammainc
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
    special_shifted_chebyshev_polynomial_u_,
    special_shifted_chebyshev_polynomial_u_out,
)
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .svd import svd
from .t_copy import t_copy, t_copy_out
from .tile import tile
from .to import to_copy
from .topk import topk
from .triu import triu
from .unique import _unique2
from .unique_consecutive import unique_consecutive
from .upsample_bicubic2d import upsample_bicubic2d
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest2d import upsample_nearest2d
from .vdot import vdot
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .zero import zero, zero_out

# Run after runtime initialization; importing tensor_wrapper in _sunrise/__init__.py
# would hit a circular import through flag_gems.utils.
_install_typed_ptr_device_patch()


__all__ = [
    "_linalg_eigvals",
    "_safe_softmax",
    "_sparse_semi_structured_mm",
    "_upsample_nearest_exact1d",
    "abs",
    "abs_",
    "add",
    "add_",
    "addmm",
    "addmm_out",
    "amin",
    "amin_out",
    "amax",
    "amax_out",
    "aminmax",
    "aminmax_out",
    "angle",
    "arcsinh",
    "arcsinh_out",
    "avg_pool3d",
    "avg_pool3d_backward",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_left_shift_out",
    "bitwise_right_shift",
    "bitwise_right_shift_",
    "bitwise_right_shift_out",
    "cat",
    "cat_out",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "clamp_min",
    "clamp_min_",
    "clamp_min_out",
    "conv2d",
    "cos",
    "cos_",
    "concatenate",
    "count_nonzero",
    "conj_physical",
    "ctc_loss",
    "cumsum",
    "cumsum_out",
    "normed_cumsum",
    "diag_embed",
    "div_mode",
    "div_mode_",
    "embedding",
    "embedding_backward",
    "empty",
    "floor_divide",
    "floor_divide_",
    "remainder",
    "remainder_",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "dropout",
    "dropout_backward",
    "efficient_attention_backward",
    "eq",
    "eq_scalar",
    "equal",
    "exponential_",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "fix",
    "fix_out",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "fft",
    "fft_c2c",
    "flash_attention_backward",
    "gather",
    "gather_backward",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "group_mm",
    "hypot",
    "hypot_out",
    "i0",
    "i0_out",
    "i0_",
    "index_add",
    "index_add_",
    "index_put",
    "index_put_",
    "index_reduce_",
    "index_select",
    "isin",
    "isnan",
    "layer_norm",
    "layer_norm_backward",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linspace",
    "log_softmax",
    "log_softmax_backward",
    "logaddexp",
    "logaddexp_out",
    "logical_and",
    "logical_or",
    "logical_or_",
    "margin_ranking_loss",
    "masked_select",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "mean",
    "mean_dim",
    "median",
    "median_dim",
    "median_dim_values",
    "median_out",
    "mul",
    "mul_",
    "multiply_",
    "multinomial",
    "mv",
    "neg",
    "neg_",
    "nonzero",
    "one_hot",
    "pad",
    "polar",
    "constant_pad_nd",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "quantile",
    "randperm",
    "reflection_pad2d",
    "reflection_pad3d_backward",
    "renorm_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "resolve_neg",
    "rms_norm",
    "rms_norm_forward",
    "rms_norm_backward",
    "scaled_grouped_mm",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_cudnn_attention_backward",
    "scaled_dot_product_efficient_attention_backward",
    "scaled_dot_product_flash_attention_backward",
    "scatter",
    "scatter_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "select_backward",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "sinc",
    "sinc_",
    "soft_margin_loss",
    "soft_margin_loss_out",
    "softmax",
    "softmax_backward",
    "sort",
    "sort_stable",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_gammainc",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_u_",
    "special_shifted_chebyshev_polynomial_u_out",
    "sub",
    "sub_",
    "svd",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "t_copy",
    "t_copy_out",
    "ScaleDotProductAttention",
    "tile",
    "to_copy",
    "topk",
    "triu",
    "_unique2",
    "unique_consecutive",
    "upsample_bicubic2d",
    "upsample_linear1d",
    "upsample_nearest2d",
    "vdot",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "zero",
    "zero_out",
]
