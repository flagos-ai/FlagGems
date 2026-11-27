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

from ._adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from ._adaptive_avg_pool3d_backward import _adaptive_avg_pool3d_backward
from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)
from ._conj import _conj
from ._conv_depthwise2d import _conv_depthwise2d
from ._convolution_double_backward import _convolution_double_backward
from ._convolution_mode import _convolution_mode
from ._embedding_bag_dense_backward import _embedding_bag_dense_backward
from ._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams,
)
from ._flash_attention_forward import _flash_attention_forward
from ._foreach_abs import _foreach_abs as _foreach_abs
from ._foreach_abs_ import _foreach_abs_ as _foreach_abs_
from ._foreach_acos_ import _foreach_acos_ as _foreach_acos_
from ._foreach_add__List import _foreach_add__List
from ._foreach_add__Scalar import _foreach_add__Scalar
from ._foreach_add__ScalarList import _foreach_add__ScalarList
from ._foreach_add_List import _foreach_add_List as _foreach_add_List
from ._foreach_add_Scalar import _foreach_add_Scalar as _foreach_add_Scalar
from ._foreach_add_ScalarList import _foreach_add_ScalarList as _foreach_add_ScalarList
from ._foreach_addcdiv__Scalar import _foreach_addcdiv__Scalar
from ._foreach_addcdiv__ScalarList import (
    _foreach_addcdiv__ScalarList as _foreach_addcdiv__ScalarList,
)
from ._foreach_addcdiv__Tensor import _foreach_addcdiv__Tensor
from ._foreach_addcmul__Scalar import _foreach_addcmul__Scalar
from ._foreach_addcmul__ScalarList import (
    _foreach_addcmul__ScalarList as _foreach_addcmul__ScalarList,
)
from ._foreach_addcmul__Tensor import (
    _foreach_addcmul__Tensor as _foreach_addcmul__Tensor,
)
from ._foreach_asin_ import _foreach_asin_ as _foreach_asin_
from ._foreach_atan import _foreach_atan as _foreach_atan
from ._foreach_atan_ import _foreach_atan_ as _foreach_atan_
from ._foreach_ceil import _foreach_ceil as _foreach_ceil
from ._foreach_clamp_max__List import (
    _foreach_clamp_max__List as _foreach_clamp_max__List,
)
from ._foreach_clamp_max__Scalar import _foreach_clamp_max__Scalar
from ._foreach_clamp_max__ScalarList import (
    _foreach_clamp_max__ScalarList as _foreach_clamp_max__ScalarList,
)
from ._foreach_clamp_max_List import _foreach_clamp_max_List as _foreach_clamp_max_List
from ._foreach_clamp_max_Scalar import (
    _foreach_clamp_max_Scalar as _foreach_clamp_max_Scalar,
)
from ._foreach_clamp_max_ScalarList import (
    _foreach_clamp_max_ScalarList as _foreach_clamp_max_ScalarList,
)
from ._foreach_clamp_min__List import (
    _foreach_clamp_min__List as _foreach_clamp_min__List,
)
from ._foreach_clamp_min__Scalar import _foreach_clamp_min__Scalar
from ._foreach_clamp_min__ScalarList import (
    _foreach_clamp_min__ScalarList as _foreach_clamp_min__ScalarList,
)
from ._foreach_clamp_min_List import _foreach_clamp_min_List as _foreach_clamp_min_List
from ._foreach_clamp_min_Scalar import (
    _foreach_clamp_min_Scalar as _foreach_clamp_min_Scalar,
)
from ._foreach_clamp_min_ScalarList import (
    _foreach_clamp_min_ScalarList as _foreach_clamp_min_ScalarList,
)
from ._foreach_copy import _foreach_copy as _foreach_copy
from ._foreach_copy_ import _foreach_copy_
from ._foreach_cos import _foreach_cos as _foreach_cos
from ._foreach_cos_ import _foreach_cos_ as _foreach_cos_
from ._foreach_div__List import _foreach_div__List as _foreach_div__List
from ._foreach_div__Scalar import _foreach_div__Scalar
from ._foreach_div__ScalarList import (
    _foreach_div__ScalarList as _foreach_div__ScalarList,
)
from ._foreach_div__Tensor import _foreach_div__Tensor as _foreach_div__Tensor
from ._foreach_div_Scalar import _foreach_div_Scalar as _foreach_div_Scalar
from ._foreach_div_ScalarList import _foreach_div_ScalarList as _foreach_div_ScalarList
from ._foreach_div_Tensor import _foreach_div_Tensor as _foreach_div_Tensor
from ._foreach_erf import _foreach_erf as _foreach_erf
from ._foreach_erf_ import _foreach_erf_ as _foreach_erf_
from ._foreach_erfc_ import _foreach_erfc_ as _foreach_erfc_
from ._foreach_exp import _foreach_exp as _foreach_exp
from ._foreach_exp_ import _foreach_exp_ as _foreach_exp_
from ._foreach_expm1 import _foreach_expm1 as _foreach_expm1
from ._foreach_expm1_ import _foreach_expm1_ as _foreach_expm1_
from ._foreach_floor import _foreach_floor as _foreach_floor
from ._foreach_floor_ import _foreach_floor_ as _foreach_floor_
from ._foreach_frac import _foreach_frac as _foreach_frac
from ._foreach_frac_ import _foreach_frac_ as _foreach_frac_
from ._foreach_log import _foreach_log as _foreach_log
from ._foreach_log1p import _foreach_log1p as _foreach_log1p
from ._foreach_log2 import _foreach_log2 as _foreach_log2
from ._foreach_log2_ import _foreach_log2_ as _foreach_log2_
from ._foreach_log10 import _foreach_log10 as _foreach_log10
from ._foreach_log10_ import _foreach_log10_ as _foreach_log10_
from ._foreach_max import _foreach_max as _foreach_max
from ._foreach_maximum__Scalar import _foreach_maximum__Scalar
from ._foreach_maximum__ScalarList import _foreach_maximum__ScalarList
from ._foreach_maximum_List import _foreach_maximum_List as _foreach_maximum_List
from ._foreach_minimum__Scalar import _foreach_minimum__Scalar
from ._foreach_minimum_List import _foreach_minimum_List as _foreach_minimum_List
from ._foreach_minimum_Scalar import _foreach_minimum_Scalar as _foreach_minimum_Scalar
from ._foreach_mul__List import _foreach_mul__List
from ._foreach_mul__Scalar import _foreach_mul__Scalar
from ._foreach_mul__ScalarList import _foreach_mul__ScalarList
from ._foreach_neg import _foreach_neg as _foreach_neg
from ._foreach_norm import _foreach_norm as _foreach_norm
from ._foreach_pow__Scalar import _foreach_pow__Scalar
from ._foreach_pow__ScalarList import _foreach_pow__ScalarList
from ._foreach_pow_List import _foreach_pow_List as _foreach_pow_List
from ._foreach_pow_Scalar import _foreach_pow_Scalar as _foreach_pow_Scalar
from ._foreach_pow_ScalarAndTensor import (
    _foreach_pow_ScalarAndTensor as _foreach_pow_ScalarAndTensor,
)
from ._foreach_pow_ScalarList import _foreach_pow_ScalarList as _foreach_pow_ScalarList
from ._foreach_reciprocal import _foreach_reciprocal as _foreach_reciprocal
from ._foreach_reciprocal_ import _foreach_reciprocal_ as _foreach_reciprocal_
from ._foreach_round_ import _foreach_round_
from ._foreach_rsqrt import _foreach_rsqrt as _foreach_rsqrt
from ._foreach_sigmoid import _foreach_sigmoid as _foreach_sigmoid
from ._foreach_sigmoid_ import _foreach_sigmoid_ as _foreach_sigmoid_
from ._foreach_sign_ import _foreach_sign_
from ._foreach_sin import _foreach_sin as _foreach_sin
from ._foreach_sin_ import _foreach_sin_ as _foreach_sin_
from ._foreach_sqrt import _foreach_sqrt as _foreach_sqrt
from ._foreach_sqrt_ import _foreach_sqrt_ as _foreach_sqrt_
from ._foreach_sub__List import _foreach_sub__List as _foreach_sub__List
from ._foreach_sub__ScalarList import (
    _foreach_sub__ScalarList as _foreach_sub__ScalarList,
)
from ._foreach_sub_List import _foreach_sub_List as _foreach_sub_List
from ._foreach_sub_Scalar import _foreach_sub_Scalar as _foreach_sub_Scalar
from ._foreach_tan import _foreach_tan as _foreach_tan
from ._foreach_tan_ import _foreach_tan_ as _foreach_tan_
from ._foreach_tanh import _foreach_tanh as _foreach_tanh
from ._foreach_tanh_ import _foreach_tanh_ as _foreach_tanh_
from ._foreach_trunc import _foreach_trunc as _foreach_trunc
from ._foreach_trunc_ import _foreach_trunc_ as _foreach_trunc_
from ._fused_rms_norm_backward import (
    _fused_rms_norm_backward as _fused_rms_norm_backward,
)

# Hygon internal implementation for attention
from ._masked_scale import _masked_scale
from ._native_batch_norm_legit_no_training import _native_batch_norm_legit_no_training
from ._scaled_dot_product_attention_math_for_mps import (
    _scaled_dot_product_attention_math_for_mps,
)
from ._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention
from ._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from .adaptive_avg_pool2d_backward import adaptive_avg_pool2d_backward
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .addmm_ import addmm_
from .addmv_ import addmv_
from .addr import addr
from .any import any, any_dim, any_dims
from .argsort import argsort
from .as_strided_scatter import as_strided_scatter
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool1d import avg_pool1d
from .avg_pool3d_backward import avg_pool3d_backward
from .baddbmm import baddbmm, baddbmm_out
from .baddbmm_ import baddbmm_
from .beam_search_score import beam_search_score
from .binary_cross_entropy_backward import binary_cross_entropy_backward
from .block_diag import block_diag
from .bmm import bmm, bmm_out
from .bmm_w8a8 import bmm_w8a8
from .bmm_w8a8_fp8 import bmm_w8a8_fp8
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .clamp import clamp, clamp_, clamp_min, clamp_min_, clamp_tensor, clamp_tensor_
from .conj_physical import conj_physical
from .conv2d import conv2d
from .conv_depthwise2d import conv_depthwise2d
from .conv_tbc_backward import conv_tbc_backward
from .convolution_overrideable import convolution_overrideable
from .convolution_overrideable_out import convolution_overrideable_out
from .cudnn_convolution import cudnn_convolution
from .dequantize import dequantize
from .diagonal_scatter import diagonal_scatter
from .diff import diff
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide_out,
    trunc_divide,
    trunc_divide_,
)
from .divide import divide
from .embedding_bag import _embedding_bag, _embedding_bag_forward_only
from .embedding_bag_backward import _embedding_bag_backward
from .embedding_bag_dense_backward import embedding_bag_dense_backward
from .expand_copy import expand_copy
from .exponential import exponential
from .exponential_ import exponential_
from .feature_dropout import feature_dropout as feature_dropout
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .flash_attention_backward import flash_attention_backward
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
from .fmod_scalar import fmod_scalar
from .fused_moving_avg_obs_fq_helper import fused_moving_avg_obs_fq_helper
from .gcd_ import gcd_
from .gelu import gelu, gelu_, gelu_backward
from .greater_ import greater_
from .grid_sampler_3d import grid_sampler_3d
from .hadamard_transform import hadamard_transform
from .hardsigmoid_backward import hardsigmoid_backward
from .hash_tensor import hash_tensor
from .histc import histc
from .igamma import igamma
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_reduce_out import index_reduce_out
from .index_select_backward import index_select_backward
from .inner import inner
from .int_mm import int_mm, int_mm_out
from .isin import isin
from .jagged_to_padded_dense_forward import jagged_to_padded_dense_forward
from .layernorm import layer_norm, layer_norm_backward, native_layer_norm
from .lcm import lcm, lcm_
from .ldexp import ldexp
from .ldexp_ import ldexp_
from .ldl_factor_ex import ldl_factor_ex
from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
from .lift_fresh import lift_fresh
from .linalg_ldl_factor import ldl_factor, linalg_ldl_factor
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_multi_dot import linalg_multi_dot as linalg_multi_dot
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linear import linear
from .log_normal_ import log_normal_
from .lstm import lstm
from .masked_scale import masked_scale
from .masked_scatter_backward import masked_scatter_backward
from .matmul_bf16 import matmul_bf16
from .matmul_bias_activation import matmul_bias_activation
from .matmul_int8 import matmul_int8
from .max_pool3d_with_indices import (
    max_pool3d_backward,
    max_pool3d_with_indices,
    pool3d_output_size,
)
from .max_unpool2d import max_unpool2d
from .median import median_dim, median_dim_values
from .mixed_dtypes_linear import mixed_dtypes_linear
from .mm import mm
from .mm_w8a8_int8 import mm_w8a8_int8, mm_w8a8_int8_out
from .mse_loss_backward import mse_loss_backward
from .mul import mul, mul_
from .multiply_ import multiply_
from .mv import mv
from .mvlgamma import mvlgamma
from .nan_to_num import nan_to_num
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .nll_loss2d import nll_loss2d
from .nll_loss_backward import heur_block_n, nll_loss_backward
from .nll_loss_nd import nll_loss_forward, nll_loss_nd_backward
from .nonzero_numpy import nonzero_numpy
from .norm_scalaropt_dim import norm_scalaropt_dim
from .nuclear_norm import nuclear_norm
from .or_tensor import or_tensor
from .ormqr import ormqr
from .pad_sequence import pad_sequence
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .randperm import randperm
from .reflection_pad1d_backward import reflection_pad1d_backward
from .reflection_pad3d_backward import reflection_pad3d_backward
from .renorm import renorm
from .renorm_ import renorm_ as renorm_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .replication_pad2d import replication_pad2d
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .replication_pad3d_backward import replication_pad3d_backward
from .rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8
from .rrelu_with_noise import rrelu_with_noise, rrelu_with_noise_
from .scalar_tensor import scalar_tensor
from .scatter import scatter, scatter_
from .scatter_add import scatter_add
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from .silu import silu, silu_, silu_backward
from .softplus_backward import softplus_backward as softplus_backward
from .softshrink_backward import softshrink_backward
from .sort import sort, sort_stable
from .special_bessel_j0 import special_bessel_j0
from .special_chebyshev_polynomial_t import special_chebyshev_polynomial_t
from .special_chebyshev_polynomial_t_out import special_chebyshev_polynomial_t_out
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import (
    special_chebyshev_polynomial_v as special_chebyshev_polynomial_v,
)
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_hermite_polynomial_h import (
    special_hermite_polynomial_h as special_hermite_polynomial_h,
)
from .special_i1e import special_i1e
from .special_i1e_out import special_i1e_out
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_multigammaln import special_multigammaln
from .special_ndtri import special_ndtri
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_t import (
    special_shifted_chebyshev_polynomial_t,
)
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
)
from .special_shifted_chebyshev_polynomial_v import (
    special_shifted_chebyshev_polynomial_v,
)
from .split_with_sizes_copy import split_with_sizes_copy
from .thnn_fused_gru_cell_backward import (
    _thnn_fused_gru_cell_backward,
    _thnn_fused_gru_cell_backward_out,
)
from .tile import tile
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .triangular_solve import triangular_solve, triangular_solve_out
from .true_divide import true_divide as true_divide
from .true_divide import true_divide_tensor as true_divide_tensor
from .true_divide_ import true_divide_ as true_divide_
from .true_divide_ import true_divide_tensor_ as true_divide_tensor_
from .unique import _unique2
from .unique_dim import unique_dim
from .upsample_lanczos2d_aa_backward import upsample_lanczos2d_aa_backward
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest_exact2d_backward import upsample_nearest_exact2d_backward
from .vdot import vdot
from .weight_norm import (
    weight_norm,
    weight_norm_except_dim,
    weight_norm_except_dim_backward,
    weight_norm_interface,
    weight_norm_interface_backward,
)

__all__ = [
    "_adaptive_avg_pool2d_backward",
    "_adaptive_avg_pool3d_backward",
    "_amp_foreach_non_finite_check_and_unscale_",
    "_conj",
    "_conv_depthwise2d",
    "_convolution_double_backward",
    "_convolution_mode",
    "_embedding_bag",
    "_embedding_bag_backward",
    "_embedding_bag_dense_backward",
    "_embedding_bag_forward_only",
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_flash_attention_forward",
    "_foreach_abs",
    "_foreach_abs_",
    "_foreach_acos_",
    "_foreach_add__List",
    "_foreach_add__Scalar",
    "_foreach_add__ScalarList",
    "_foreach_add_List",
    "_foreach_add_Scalar",
    "_foreach_add_ScalarList",
    "_foreach_addcdiv__Scalar",
    "_foreach_addcdiv__ScalarList",
    "_foreach_addcdiv__Tensor",
    "_foreach_addcmul__Scalar",
    "_foreach_addcmul__ScalarList",
    "_foreach_addcmul__Tensor",
    "_foreach_asin_",
    "_foreach_atan",
    "_foreach_atan_",
    "_foreach_ceil",
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
    "_foreach_div__List",
    "_foreach_div__Scalar",
    "_foreach_div__ScalarList",
    "_foreach_div__Tensor",
    "_foreach_div_Scalar",
    "_foreach_div_ScalarList",
    "_foreach_div_Tensor",
    "_foreach_erf",
    "_foreach_erf_",
    "_foreach_erfc_",
    "_foreach_exp",
    "_foreach_exp_",
    "_foreach_expm1",
    "_foreach_expm1_",
    "_foreach_floor",
    "_foreach_floor_",
    "_foreach_frac",
    "_foreach_frac_",
    "_foreach_log",
    "_foreach_log10",
    "_foreach_log10_",
    "_foreach_log1p",
    "_foreach_log2",
    "_foreach_log2_",
    "_foreach_max",
    "_foreach_maximum__Scalar",
    "_foreach_maximum__ScalarList",
    "_foreach_maximum_List",
    "_foreach_minimum__Scalar",
    "_foreach_minimum_List",
    "_foreach_minimum_Scalar",
    "_foreach_mul__List",
    "_foreach_mul__Scalar",
    "_foreach_mul__ScalarList",
    "_foreach_neg",
    "_foreach_norm",
    "_foreach_pow__Scalar",
    "_foreach_pow__ScalarList",
    "_foreach_pow_List",
    "_foreach_pow_Scalar",
    "_foreach_pow_ScalarAndTensor",
    "_foreach_pow_ScalarList",
    "_foreach_reciprocal",
    "_foreach_reciprocal_",
    "_foreach_round_",
    "_foreach_rsqrt",
    "_foreach_sigmoid",
    "_foreach_sigmoid_",
    "_foreach_sign_",
    "_foreach_sin",
    "_foreach_sin_",
    "_foreach_sqrt",
    "_foreach_sqrt_",
    "_foreach_sub__List",
    "_foreach_sub__ScalarList",
    "_foreach_sub_List",
    "_foreach_sub_Scalar",
    "_foreach_tan",
    "_foreach_tan_",
    "_foreach_tanh",
    "_foreach_tanh_",
    "_foreach_trunc",
    "_foreach_trunc_",
    "_fused_rms_norm_backward",
    "_masked_scale",
    "_native_batch_norm_legit_no_training",
    "_scaled_dot_product_attention_math_for_mps",
    "_scaled_dot_product_flash_attention",
    "_thnn_fused_gru_cell_backward",
    "_thnn_fused_gru_cell_backward_out",
    "_thnn_fused_lstm_cell",
    "_unique2",
    "adaptive_avg_pool2d_backward",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d_backward",
    "addmm",
    "addmm_",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv_",
    "addr",
    "any",
    "any_dim",
    "any_dims",
    "argsort",
    "as_strided_scatter",
    "avg_pool1d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_",
    "baddbmm_out",
    "beam_search_score",
    "binary_cross_entropy_backward",
    "block_diag",
    "bmm",
    "bmm_out",
    "bmm_w8a8",
    "bmm_w8a8_fp8",
    "cholesky_inverse",
    "cholesky_solve",
    "cholesky_solve_out",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "conj_physical",
    "conv2d",
    "conv_depthwise2d",
    "conv_tbc_backward",
    "convolution_overrideable",
    "convolution_overrideable_out",
    "cudnn_convolution",
    "dequantize",
    "diagonal_scatter",
    "diff",
    "div_mode",
    "div_mode_",
    "divide",
    "embedding_bag_dense_backward",
    "expand_copy",
    "exponential",
    "exponential_",
    "feature_dropout",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "fill_tensor_out",
    "flash_attention_backward",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "float_power_scalar_tensor",
    "float_power_scalar_tensor_out",
    "float_power_tensor_scalar",
    "float_power_tensor_scalar_",
    "float_power_tensor_scalar_out",
    "float_power_tensor_tensor",
    "float_power_tensor_tensor_",
    "float_power_tensor_tensor_out",
    "floor_divide",
    "floor_divide_",
    "fmod_scalar",
    "fused_moving_avg_obs_fq_helper",
    "gcd_",
    "gelu",
    "gelu_",
    "gelu_backward",
    "greater_",
    "grid_sampler_3d",
    "hadamard_transform",
    "hardsigmoid_backward",
    "hash_tensor",
    "heur_block_n",
    "histc",
    "igamma",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_reduce_out",
    "index_select_backward",
    "inner",
    "int_mm",
    "int_mm_out",
    "isin",
    "jagged_to_padded_dense_forward",
    "layer_norm",
    "layer_norm_backward",
    "lcm",
    "lcm_",
    "ldexp",
    "ldexp_",
    "ldl_factor",
    "ldl_factor_ex",
    "lerp_scalar",
    "lerp_scalar_",
    "lerp_tensor",
    "lerp_tensor_",
    "lift_fresh",
    "linalg_ldl_factor",
    "linalg_lstsq",
    "linalg_matrix_norm",
    "linalg_matrix_norm_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_multi_dot",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linear",
    "log_normal_",
    "lstm",
    "masked_scale",
    "masked_scatter_backward",
    "matmul_bf16",
    "matmul_bias_activation",
    "matmul_int8",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "max_unpool2d",
    "median_dim",
    "median_dim_values",
    "mixed_dtypes_linear",
    "mm",
    "mm_w8a8_int8",
    "mm_w8a8_int8_out",
    "mse_loss_backward",
    "mul",
    "mul_",
    "multiply_",
    "mv",
    "mvlgamma",
    "nan_to_num",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "native_layer_norm",
    "nll_loss2d",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss_nd_backward",
    "nonzero_numpy",
    "norm_scalaropt_dim",
    "nuclear_norm",
    "or_tensor",
    "ormqr",
    "pad_sequence",
    "per_token_group_quant_fp8",
    "pool3d_output_size",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "randperm",
    "reflection_pad1d_backward",
    "reflection_pad3d_backward",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "replication_pad2d",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "replication_pad3d_backward",
    "rms_norm_w8a16_fp8",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "scalar_tensor",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "ScaleDotProductAttention",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "searchsorted",
    "searchsorted_out",
    "searchsorted_scalar",
    "searchsorted_scalar_out",
    "silu",
    "silu_",
    "silu_backward",
    "softplus_backward",
    "softshrink_backward",
    "sort",
    "sort_stable",
    "special_bessel_j0",
    "special_chebyshev_polynomial_t",
    "special_chebyshev_polynomial_t_out",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_hermite_polynomial_h",
    "special_i1e",
    "special_i1e_out",
    "special_legendre_polynomial_p",
    "special_multigammaln",
    "special_ndtri",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_v",
    "split_with_sizes_copy",
    "SUPPORTED_FP8_DTYPE",
    "tile",
    "topk_w8a16_fp8",
    "triangular_solve",
    "triangular_solve_out",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "true_divide_tensor",
    "true_divide_tensor_",
    "trunc_divide",
    "trunc_divide_",
    "unique_dim",
    "upsample_lanczos2d_aa_backward",
    "upsample_linear1d",
    "upsample_nearest2d",
    "upsample_nearest_exact2d_backward",
    "vdot",
    "weight_norm",
    "weight_norm_except_dim",
    "weight_norm_except_dim_backward",
    "weight_norm_interface",
    "weight_norm_interface_backward",
]
