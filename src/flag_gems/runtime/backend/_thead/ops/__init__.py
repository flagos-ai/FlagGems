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


from ._conv_depthwise2d import _conv_depthwise2d
from ._flash_attention_forward import _flash_attention_forward
from ._resize_output_ import _resize_output_
from ._scaled_dot_product_fused_attention_overrideable import (
    _scaled_dot_product_fused_attention_overrideable,
)
from ._unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from ._upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addmm_ import addmm_
from .addmv_ import addmv_
from .as_strided_scatter import as_strided_scatter
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .cholesky_inverse import cholesky_inverse
from .conv_transpose1d import conv_transpose1d, conv_transpose1d_output_size
from .cudnn_batch_norm_backward import cudnn_batch_norm_backward, make_3d_for_bn
from .cudnn_convolution import cudnn_convolution
from .diagonal_scatter import diagonal_scatter
from .embedding_dense_backward import embedding_dense_backward
from .erfc import erfc
from .gcd_ import gcd, gcd_
from .grid_sampler_3d_backward import grid_sampler_3d_backward
from .index_copy_ import index_copy, index_copy_
from .index_select_backward import index_select_backward
from .lcm import lcm, lcm_
from .linalg_cholesky import linalg_cholesky
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_svdvals import linalg_svdvals
from .linear import linear
from .linear_backward import linear_backward
from .log_normal_ import log_normal_, log_normal_heur_block, log_normal_heur_num_warps
from .log_sigmoid_backward import log_sigmoid_backward, log_sigmoid_backward_out
from .matmul_bias_activation import matmul_bias_activation
from .max_pool3d_with_indices_backward import max_pool3d_with_indices_backward
from .mm import mm, mm_out
from .mm_w8a8_int8 import mm_w8a8_int8, mm_w8a8_int8_out
from .mv import mv
from .mvlgamma import mvlgamma
from .nll_loss_backward import nll_loss_backward
from .nonzero_numpy import nonzero_numpy
from .reflection_pad3d_backward import reflection_pad3d_backward
from .renorm import renorm, renorm_
from .repeat import repeat
from .replication_pad2d import replication_pad2d
from .replication_pad3d_backward import replication_pad3d_backward
from .rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8
from .scatter_reduce_ import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .softplus_backward import softplus_backward
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_erfc import special_erfc
from .special_erfinv import special_erfinv, special_erfinv_, special_erfinv_out
from .special_gammainc import special_gammainc
from .special_gammaln import special_gammaln
from .special_gammaln_out import special_gammaln_out
from .special_hermite_polynomial_h import (
    special_hermite_polynomial_h,
    special_hermite_polynomial_h_tensor_tensor,
)
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_multigammaln import special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .tile import tile
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .unbind_copy import unbind_copy

__all__ = [
    "_conv_depthwise2d",
    "_flash_attention_forward",
    "_resize_output_",
    "_scaled_dot_product_fused_attention_overrideable",
    "_unsafe_masked_index_put_accumulate",
    "_upsample_nearest_exact2d_backward",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d_backward",
    "addmm_",
    "addmv_",
    "as_strided_scatter",
    "broadcast_tensors",
    "broadcast_to",
    "cholesky_inverse",
    "conv_transpose1d",
    "conv_transpose1d_output_size",
    "cudnn_batch_norm_backward",
    "cudnn_convolution",
    "diagonal_scatter",
    "embedding_dense_backward",
    "erfc",
    "gcd",
    "gcd_",
    "grid_sampler_3d_backward",
    "index_copy",
    "index_copy_",
    "index_select_backward",
    "lcm",
    "lcm_",
    "linalg_cholesky",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_svdvals",
    "linear",
    "linear_backward",
    "log_normal_",
    "log_normal_heur_block",
    "log_normal_heur_num_warps",
    "log_sigmoid_backward",
    "log_sigmoid_backward_out",
    "make_3d_for_bn",
    "matmul_bias_activation",
    "max_pool3d_with_indices_backward",
    "mm",
    "mm_out",
    "mm_w8a8_int8",
    "mm_w8a8_int8_out",
    "mv",
    "mvlgamma",
    "nll_loss_backward",
    "nonzero_numpy",
    "reflection_pad3d_backward",
    "renorm",
    "renorm_",
    "repeat",
    "replication_pad2d",
    "replication_pad3d_backward",
    "rms_norm_w8a16_fp8",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "softplus_backward",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_erfc",
    "special_erfinv",
    "special_erfinv_",
    "special_erfinv_out",
    "special_gammainc",
    "special_gammaln",
    "special_gammaln_out",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_h_tensor_tensor",
    "special_legendre_polynomial_p",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_w",
    "tile",
    "topk_w8a16_fp8",
    "unbind_copy",
]
