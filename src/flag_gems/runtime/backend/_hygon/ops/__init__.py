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

from .any import any, any_dim, any_dims
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
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
from .isin import isin
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .mm import mm
from .mul import mul, mul_
from .nansum import nansum, nansum_out
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .randperm import randperm
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .silu import silu, silu_, silu_backward
from .sort import sort, sort_stable
from .unique import _unique2
from .upsample_nearest2d import upsample_nearest2d

__all__ = [
    "_unique2",
    "ScaleDotProductAttention",
    "SUPPORTED_FP8_DTYPE",
    "any",
    "any_dim",
    "any_dims",
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
    "gelu",
    "gelu_",
    "hadamard_transform",
    "index_add",
    "index_add_",
    "isin",
    "matmul_bf16",
    "matmul_int8",
    "mul",
    "mul_",
    "mm",
    "nansum",
    "nansum_out",
    "per_token_group_quant_fp8",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "randperm",
    "remainder",
    "remainder_",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "silu",
    "silu_",
    "silu_backward",
    "sort",
    "sort_stable",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "trunc_divide",
    "trunc_divide_",
    "upsample_nearest2d",
]
