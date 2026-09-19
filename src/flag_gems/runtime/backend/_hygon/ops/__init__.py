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
from ._conj import _conj
from ._thnn_fused_lstm_cell import _thnn_fused_lstm_cell
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addr import addr
from .any import any, any_dim, any_dims
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool3d_backward import avg_pool3d_backward
from .block_diag import block_diag
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .conj_physical import conj_physical
from .cudnn_convolution import cudnn_convolution
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
from .exponential import exponential
from .exponential_ import exponential_
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
from .gelu import gelu, gelu_, gelu_backward
from .hadamard_transform import hadamard_transform
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_select_backward import index_select_backward
from .isin import isin
from .lcm import lcm, lcm_
from .linalg_ldl_factor import ldl_factor
from .linalg_matrix_norm import linalg_matrix_norm
from .log_normal_ import log_normal_
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
from .mul import mul, mul_
from .nansum import nansum, nansum_out
from .nll_loss_backward import heur_block_n, nll_loss_backward
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .randperm import randperm
from .reflection_pad3d_backward import reflection_pad3d_backward
from .renorm import renorm, renorm_
from .repeat import repeat
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .scatter import scatter, scatter_
from .searchsorted import (
    searchsorted,
    searchsorted_out,
    searchsorted_scalar,
    searchsorted_scalar_out,
)
from .silu import silu, silu_, silu_backward
from .softplus_backward import softplus_backward
from .sort import sort, sort_stable
from .special_bessel_j0 import special_bessel_j0
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_shifted_chebyshev_polynomial_u import (
    special_shifted_chebyshev_polynomial_u,
)
from .split_with_sizes_copy import split_with_sizes_copy
from .tile import tile
from .unique import _unique2
from .unique_dim import unique_dim
from .unsqueeze import unsqueeze, unsqueeze_
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest2d import upsample_nearest2d
from .vdot import vdot
from .weight_norm import (
    weight_norm,
    weight_norm_except_dim,
    weight_norm_except_dim_backward,
    weight_norm_interface,
    weight_norm_interface_backward,
)

__all__ = [
    "_amp_foreach_non_finite_check_and_unscale_",
    "_conj",
    "_thnn_fused_lstm_cell",
    "_unique2",
    "adaptive_max_pool3d_backward",
    "addr",
    "any",
    "any_dim",
    "any_dims",
    "avg_pool3d_backward",
    "block_diag",
    "broadcast_tensors",
    "broadcast_to",
    "conj_physical",
    "cudnn_convolution",
    "diff",
    "div_mode",
    "div_mode_",
    "exponential",
    "exponential_",
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
    "isin",
    "ldl_factor",
    "lcm",
    "lcm_",
    "linalg_matrix_norm",
    "log_normal_",
    "masked_scatter_backward",
    "matmul_bf16",
    "matmul_int8",
    "max_pool3d_backward",
    "max_pool3d_with_indices",
    "max_unpool2d",
    "median_dim",
    "median_dim_values",
    "mul",
    "mul_",
    "mm",
    "nansum",
    "nansum_out",
    "nll_loss_backward",
    "per_token_group_quant_fp8",
    "pool3d_output_size",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "randperm",
    "reflection_pad3d_backward",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
    "repeat",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "ScaleDotProductAttention",
    "scatter",
    "scatter_",
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
    "special_bessel_j0",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_shifted_chebyshev_polynomial_u",
    "split_with_sizes_copy",
    "SUPPORTED_FP8_DTYPE",
    "tile",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc_divide",
    "trunc_divide_",
    "unique_dim",
    "upsample_linear1d",
    "upsample_nearest2d",
    "unsqueeze",
    "unsqueeze_",
    "vdot",
    "weight_norm",
    "weight_norm_except_dim",
    "weight_norm_except_dim_backward",
    "weight_norm_interface",
    "weight_norm_interface_backward",
]
