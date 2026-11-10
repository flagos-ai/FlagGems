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
from ._flash_attention_forward import _flash_attention_forward

# Hygon internal implementation for attention
from ._masked_scale import _masked_scale
from ._scaled_dot_product_attention_math_for_mps import (
    _scaled_dot_product_attention_math_for_mps,
)
from ._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention
from ._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from .adaptive_avg_pool2d_backward import adaptive_avg_pool2d_backward
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
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
from .baddbmm_ import baddbmm_
from .beam_search_score import beam_search_score
from .binary_cross_entropy_backward import binary_cross_entropy_backward
from .block_diag import block_diag
from .bmm_w8a8 import bmm_w8a8
from .cholesky_inverse import cholesky_inverse
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
    true_divide,
    true_divide_,
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
from .feature_dropout import feature_dropout
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
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
from .layernorm import layer_norm, native_layer_norm
from .lcm import lcm, lcm_
from .ldexp_ import ldexp_
from .lift_fresh import lift_fresh
from .linalg_ldl_factor import ldl_factor, linalg_ldl_factor
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
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
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .nll_loss_backward import heur_block_n, nll_loss_backward
from .nonzero_numpy import nonzero_numpy
from .norm_scalaropt_dim import norm_scalaropt_dim
from .nuclear_norm import nuclear_norm
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
from .renorm import renorm, renorm_
from .repeat import repeat
from .replication_pad2d import replication_pad2d
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .replication_pad3d_backward import replication_pad3d_backward
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
from .softplus_backward import softplus_backward
from .softshrink_backward import softshrink_backward
from .sort import sort, sort_stable
from .special_bessel_j0 import special_bessel_j0
from .special_chebyshev_polynomial_t import special_chebyshev_polynomial_t
from .special_chebyshev_polynomial_t_out import special_chebyshev_polynomial_t_out
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_hermite_polynomial_h import special_hermite_polynomial_h
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
from .thnn_fused_gru_cell_backward import thnn_fused_gru_cell_backward
from .tile import tile
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .triangular_solve import triangular_solve, triangular_solve_out
from .unique import _unique2
from .unique_dim import unique_dim
from .unsqueeze import unsqueeze, unsqueeze_
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
    "_flash_attention_forward",
    "_masked_scale",
    "_scaled_dot_product_attention_math_for_mps",
    "_scaled_dot_product_flash_attention",
    "_thnn_fused_lstm_cell",
    "_unique2",
    "adaptive_avg_pool2d_backward",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d_backward",
    "addmm_",
    "addmv_",
    "addr",
    "any",
    "any_dim",
    "any_dims",
    "argsort",
    "as_strided_scatter",
    "avg_pool1d",
    "avg_pool3d_backward",
    "baddbmm_",
    "beam_search_score",
    "binary_cross_entropy_backward",
    "block_diag",
    "bmm_w8a8",
    "cholesky_inverse",
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
    "lcm",
    "lcm_",
    "ldexp_",
    "ldl_factor",
    "lift_fresh",
    "linalg_ldl_factor",
    "linalg_lstsq",
    "linalg_matrix_norm",
    "linalg_matrix_norm_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
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
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "native_layer_norm",
    "nll_loss_backward",
    "nonzero_numpy",
    "norm_scalaropt_dim",
    "nuclear_norm",
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
    "replication_pad2d",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "replication_pad3d_backward",
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
    "thnn_fused_gru_cell_backward",
    "tile",
    "topk_w8a16_fp8",
    "triangular_solve",
    "triangular_solve_out",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc_divide",
    "trunc_divide_",
    "unique_dim",
    "unsqueeze",
    "unsqueeze_",
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
