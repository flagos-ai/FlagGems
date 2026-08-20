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

from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from .adaptive_max_pool3d_backward import run
from .addmm import addmm, addmm_out
from .addmm_ import addmm_
from .arccosh_ import arccosh_
from .avg_pool3d import avg_pool3d_backward
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d
from .diagonal_scatter import diagonal_scatter
from .div import div_mode, div_mode_
from .gcd_ import gcd_
from .hadamard_transform import hadamard_transform
from .histc import histc
from .index_select_backward import index_select_backward
from .linalg_cholesky import linalg_cholesky
from .linear import linear
from .log_normal_ import log_normal_
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .mm import mm, mm_out
from .narrow_copy import narrow_copy
from .nonzero_numpy import nonzero_numpy
from .permute_copy import permute_copy
from .renorm_ import renorm_
from .repeat import repeat
from .resolve_neg import resolve_neg
from .scatter_add import scatter_add_
from .softplus import softplus_backward
from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_gammainc import special_gammainc
from .special_hermite_polynomial_h import (
    special_hermite_polynomial_h,
    special_hermite_polynomial_h_tensor_tensor,
)
from .special_modified_bessel_k1 import (
    special_modified_bessel_k1,
    special_modified_bessel_k1_out,
)
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .tile import tile
from .var import var, var_correction, var_dim

__all__ = [
    "_conv_depthwise2d",
    "_native_batch_norm_legit_functional",
    "addmm",
    "addmm_",
    "addmm_out",
    "arccosh_",
    "avg_pool3d_backward",
    "broadcast_tensors",
    "broadcast_to",
    "cholesky_solve",
    "cholesky_solve_out",
    "conv_transpose1d",
    "diagonal_scatter",
    "div_mode",
    "div_mode_",
    "gcd_",
    "hadamard_transform",
    "histc",
    "index_select_backward",
    "linalg_cholesky",
    "linear",
    "log_normal_",
    "matmul_bf16",
    "matmul_int8",
    "mm",
    "mm_out",
    "narrow_copy",
    "nonzero_numpy",
    "permute_copy",
    "renorm_",
    "repeat",
    "resolve_neg",
    "run",
    "scatter_add_",
    "softplus_backward",
    "sparse_sampled_addmm",
    "sparse_sampled_addmm_out",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_gammainc",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_h_tensor_tensor",
    "special_modified_bessel_k1",
    "special_modified_bessel_k1_out",
    "special_shifted_chebyshev_polynomial_w",
    "tile",
    "var",
    "var_correction",
    "var_dim",
]
