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

from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
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
from .exponential_ import exponential_
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .gelu import gelu, gelu_
from .hadamard_transform import hadamard_transform
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_select_backward import index_select_backward
from .isin import isin
from .lcm import lcm, lcm_
from .log_normal_ import log_normal_
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
from .silu import silu, silu_, silu_backward
from .softplus_backward import softplus_backward
from .sort import sort, sort_stable
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .split_with_sizes_copy import split_with_sizes_copy
from .tile import tile
from .unique import _unique2
from .upsample_nearest2d import upsample_nearest2d
from .weight_norm import (
    backward,
    forward,
    weight_norm,
    weight_norm_except_dim,
    weight_norm_except_dim_backward,
    weight_norm_interface,
    weight_norm_interface_backward,
)

__all__ = [
    "_unique2",
    "adaptive_max_pool3d_backward",
    "avg_pool3d_backward",
    "backward",
    "broadcast_tensors",
    "broadcast_to",
    "conj_physical",
    "ScaleDotProductAttention",
    "SUPPORTED_FP8_DTYPE",
    "any",
    "any_dim",
    "any_dims",
    "cudnn_convolution",
    "diff",
    "div_mode",
    "div_mode_",
    "exponential_",
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
    "forward",
    "gelu",
    "gelu_",
    "hadamard_transform",
    "heur_block_n",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_select_backward",
    "isin",
    "lcm",
    "lcm_",
    "log_normal_",
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
    "silu",
    "silu_",
    "silu_backward",
    "softplus_backward",
    "sort",
    "sort_stable",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "split_with_sizes_copy",
    "tile",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc_divide",
    "trunc_divide_",
    "upsample_nearest2d",
    "weight_norm",
    "weight_norm_except_dim",
    "weight_norm_except_dim_backward",
    "weight_norm_interface",
    "weight_norm_interface_backward",
]
