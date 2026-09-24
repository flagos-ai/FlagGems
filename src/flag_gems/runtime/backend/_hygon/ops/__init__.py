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

from ._flash_attention_forward import _flash_attention_forward

# Hygon internal implementation for attention
from ._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention
from .adaptive_avg_pool2d_backward import adaptive_avg_pool2d_backward
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addmm_ import addmm_
from .addmv_ import addmv_
from .addr import addr
from .amp_foreach_non_finite_check_and_unscale_ import (
    amp_foreach_non_finite_check_and_unscale_,
)
from .any import any, any_dim, any_dims
from .as_strided_scatter import as_strided_scatter
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool3d_backward import avg_pool3d_backward
from .baddbmm_ import baddbmm_
from .beam_search_score import beam_search_score
from .binary_cross_entropy_backward import binary_cross_entropy_backward
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .cholesky_inverse import cholesky_inverse
from .conj_physical import conj_physical
from .conv_depthwise2d import conv_depthwise2d
from .cudnn_convolution import cudnn_convolution
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
from .embedding_bag_dense_backward import embedding_bag_dense_backward
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
from .fused_moving_avg_obs_fq_helper import fused_moving_avg_obs_fq_helper
from .gcd_ import gcd_
from .gelu import gelu, gelu_, gelu_backward
from .hadamard_transform import hadamard_transform
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_select_backward import index_select_backward
from .int_mm import int_mm, int_mm_out
from .isin import isin
from .jagged_to_padded_dense_forward import jagged_to_padded_dense_forward
from .lcm import lcm, lcm_
from .lift_fresh import lift_fresh
from .linalg_ldl_factor import ldl_factor, linalg_ldl_factor
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .log_normal_ import log_normal_
from .masked_scale import masked_scale
from .masked_scatter_backward import masked_scatter_backward
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max_pool3d_with_indices import (
    max_pool3d_backward,
    max_pool3d_with_indices,
    pool3d_output_size,
)
from .max_unpool2d import max_unpool2d
from .median import median_dim, median_dim_values
from .mm import mm
from .mm_w8a8_int8 import mm_w8a8_int8, mm_w8a8_int8_out
from .mse_loss_backward import mse_loss_backward
from .mul import mul, mul_
from .mv import mv
from .mvlgamma import mvlgamma
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .nll_loss_backward import heur_block_n, nll_loss_backward
from .nonzero_numpy import nonzero_numpy
from .norm_scalaropt_dim import norm_scalaropt_dim
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
from .sort import sort, sort_stable
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_hermite_polynomial_h import special_hermite_polynomial_h
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_multigammaln import special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
)
from .special_shifted_chebyshev_polynomial_v import (
    special_shifted_chebyshev_polynomial_v,
)
from .split_with_sizes_copy import split_with_sizes_copy
from .thnn_fused_lstm_cell import thnn_fused_lstm_cell
from .tile import tile
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .unique import _unique2
from .unique_dim import unique_dim
from .unsqueeze import unsqueeze, unsqueeze_
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest_exact2d_backward import upsample_nearest_exact2d_backward
from .weight_norm import (
    weight_norm,
    weight_norm_except_dim,
    weight_norm_except_dim_backward,
    weight_norm_interface,
    weight_norm_interface_backward,
)

__all__ = [
    "_flash_attention_forward",
    "_scaled_dot_product_flash_attention",
    "_unique2",
    "adaptive_avg_pool2d_backward",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d_backward",
    "addmm_",
    "addmv_",
    "addr",
    "amp_foreach_non_finite_check_and_unscale_",
    "any",
    "any_dim",
    "any_dims",
    "as_strided_scatter",
    "avg_pool3d_backward",
    "baddbmm_",
    "beam_search_score",
    "binary_cross_entropy_backward",
    "broadcast_tensors",
    "broadcast_to",
    "cholesky_inverse",
    "conj_physical",
    "conv_depthwise2d",
    "cudnn_convolution",
    "diagonal_scatter",
    "diff",
    "div_mode",
    "div_mode_",
    "embedding_bag_dense_backward",
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
    "floor_divide",
    "floor_divide_",
    "fused_moving_avg_obs_fq_helper",
    "gcd_",
    "gelu",
    "gelu_",
    "gelu_backward",
    "hadamard_transform",
    "heur_block_n",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_select_backward",
    "int_mm",
    "int_mm_out",
    "isin",
    "jagged_to_padded_dense_forward",
    "lcm",
    "lcm_",
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
    "log_normal_",
    "masked_scale",
    "masked_scatter_backward",
    "matmul_bf16",
    "matmul_int8",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "max_unpool2d",
    "median_dim",
    "median_dim_values",
    "mm",
    "mm_w8a8_int8",
    "mm_w8a8_int8_out",
    "mse_loss_backward",
    "mul",
    "mul_",
    "mv",
    "mvlgamma",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "nll_loss_backward",
    "nonzero_numpy",
    "norm_scalaropt_dim",
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
    "sort",
    "sort_stable",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_hermite_polynomial_h",
    "special_legendre_polynomial_p",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_v",
    "split_with_sizes_copy",
    "SUPPORTED_FP8_DTYPE",
    "thnn_fused_lstm_cell",
    "tile",
    "topk_w8a16_fp8",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc_divide",
    "trunc_divide_",
    "unique_dim",
    "unsqueeze",
    "unsqueeze_",
    "upsample_nearest2d",
    "upsample_nearest_exact2d_backward",
    "weight_norm",
    "weight_norm_except_dim",
    "weight_norm_except_dim_backward",
    "weight_norm_interface",
    "weight_norm_interface_backward",
]
