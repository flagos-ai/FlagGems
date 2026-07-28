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

from .add import add, add_
from .all import all, all_dim, all_dims
from .arange import arange, arange_start
from .argmax import argmax
from .argmin import argmin
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .baddbmm import baddbmm, baddbmm_out
from .cat import cat
from .count_nonzero import count_nonzero
from .cumsum import cumsum, cumsum_out, normed_cumsum
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
from .embedding import embedding, embedding_backward
from .exponential_ import exponential_
from .fill import fill_scalar, fill_scalar_, fill_scalar_out, fill_tensor, fill_tensor_
from .flash_api import mha_fwd, mha_varlan_fwd
from .hstack import hstack
from .index import index
from .index_add import index_add, index_add_
from .isin import isin
from .kron import kron
from .masked_select import masked_select
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max import max, max_dim
from .mean import mean, mean_dim
from .mm import mm, mm_out
from .mse_loss import mse_loss
from .mul import mul, mul_
from .normal import (
    normal_,
    normal_distribution,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .randn import randn
from .randn_like import randn_like
from .repeat import repeat
from .rms_norm import rms_norm
from .rsqrt import rsqrt, rsqrt_
from .select_scatter import select_scatter
from .silu_and_mul import silu_and_mul, silu_and_mul_out
from .stack import stack
from .sub import sub, sub_
from .tile import tile
from .unique import _unique2
from .upsample_bicubic2d import upsample_bicubic2d
from .vdot import vdot
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "_unique2",
    "add",
    "add_",
    "all",
    "all_dim",
    "all_dims",
    "arange",
    "arange_start",
    "argmax",
    "argmin",
    "baddbmm",
    "baddbmm_out",
    "cat",
    "count_nonzero",
    "cumsum",
    "cumsum_out",
    "div_mode",
    "div_mode_",
    "embedding",
    "embedding_backward",
    "exponential_",
    "fill_scalar",
    "fill_scalar_",
    "fill_scalar_out",
    "fill_tensor",
    "fill_tensor_",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "floor_divide",
    "floor_divide_",
    "hstack",
    "index",
    "index_add",
    "index_add_",
    "isin",
    "kron",
    "masked_select",
    "matmul_bf16",
    "matmul_int8",
    "max",
    "max_dim",
    "mean",
    "mean_dim",
    "mha_fwd",
    "mha_varlan_fwd",
    "mm",
    "mm_out",
    "mse_loss",
    "mul",
    "mul_",
    "normal_",
    "normal_distribution",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "randn",
    "randn_like",
    "remainder",
    "remainder_",
    "repeat",
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "rsqrt",
    "rsqrt_",
    "ScaleDotProductAttention",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "select_scatter",
    "silu_and_mul",
    "silu_and_mul_out",
    "stack",
    "sub",
    "sub_",
    "tile",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "upsample_bicubic2d",
    "vdot",
    "zero_",
    "zeros",
    "zeros_like",
]
