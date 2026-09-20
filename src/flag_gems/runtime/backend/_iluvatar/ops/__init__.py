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

import importlib

from ..utils.pointwise_dynamic import ModuleGenerator
from ._native_batch_norm_legit_functional import _native_batch_norm_legit_functional
from .acos_ import acos_
from .adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from .adaptive_max_pool2d_backward import adaptive_max_pool2d_backward
from .adaptive_max_pool3d_backward import run
from .addmm import addmm, addmm_out
from .addmm_ import addmm_
from .arccos_ import arccos_
from .arccosh_ import arccosh_
from .as_strided_scatter import as_strided_scatter
from .avg_pool3d import avg_pool3d_backward
from .batch_norm_no_update import run as _batch_norm_no_update
from .broadcast_tensors import broadcast_tensors
from .broadcast_to import broadcast_to
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .cholesky_solve_helper import run as _cholesky_solve_helper
from .concatenate import run as concatenate
from .constant_pad_nd import constant_pad_nd
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d
from .diagonal_scatter import diagonal_scatter
from .div import div_mode, div_mode_
from .fractional_max_pool2d_backward import fractional_max_pool2d_backward
from .gcd_ import gcd_
from .gru import gru, gru_data
from .hadamard_transform import hadamard_transform
from .histc import histc
from .igamma_ import igamma_
from .index_copy_ import index_copy_
from .index_select_backward import index_select_backward
from .kthvalue import kthvalue
from .lift_out import lift_out
from .linalg_cholesky import linalg_cholesky
from .linalg_householder_product import run as linalg_householder_product
from .linalg_ldl_factor_ex import ldl_factor_ex
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_norm import linalg_norm
from .linalg_qr import linalg_qr, linalg_qr_out
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linalg_svdvals import linalg_svdvals
from .linear import linear
from .log10_ import log10_
from .log_normal import log_normal
from .log_normal_ import log_normal_
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max_pool3d_with_indices_backward import max_pool3d_with_indices_backward
from .median import median
from .mm import mm, mm_out
from .mvlgamma import run as mvlgamma
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .narrow_copy import narrow_copy
from .nonzero_numpy import nonzero_numpy
from .pad_sequence import pad_sequence
from .permute_copy import permute_copy
from .randperm import randperm
from .renorm_ import renorm_
from .repeat import repeat
from .repeat_interleave import repeat_interleave_self_int
from .resolve_neg import resolve_neg
from .scatter_add import scatter_add_
from .softplus import softplus_backward
from .sort import sort, sort_stable
from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out
from .special_bessel_j0 import run as special_bessel_j0
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_gammainc import special_gammainc
from .special_hermite_polynomial_h import special_hermite_polynomial_h
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_modified_bessel_k0 import special_modified_bessel_k0
from .special_modified_bessel_k0_out import special_modified_bessel_k0_out
from .special_modified_bessel_k1 import (
    special_modified_bessel_k1,
    special_modified_bessel_k1_out,
)
from .special_multigammaln import run as special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from .tile import tile
from .unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from .var import var, var_correction, var_dim

_pointwise_dynamic = importlib.import_module("flag_gems.utils.pointwise_dynamic")
_pointwise_dynamic.ModuleGenerator = ModuleGenerator

__all__ = [
    "_adaptive_avg_pool2d_backward",
    "_batch_norm_no_update",
    "_cholesky_solve_helper",
    "_conv_depthwise2d",
    "_native_batch_norm_legit_functional",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unsafe_masked_index_put_accumulate",
    "acos_",
    "adaptive_max_pool2d_backward",
    "addmm",
    "addmm_",
    "addmm_out",
    "arccos_",
    "arccosh_",
    "as_strided_scatter",
    "avg_pool3d_backward",
    "broadcast_tensors",
    "broadcast_to",
    "cholesky_solve",
    "cholesky_solve_out",
    "concatenate",
    "constant_pad_nd",
    "conv_transpose1d",
    "diagonal_scatter",
    "div_mode",
    "div_mode_",
    "fractional_max_pool2d_backward",
    "gcd_",
    "gru",
    "gru_data",
    "hadamard_transform",
    "histc",
    "igamma_",
    "index_copy_",
    "index_select_backward",
    "kthvalue",
    "ldl_factor_ex",
    "lift_out",
    "linalg_cholesky",
    "linalg_householder_product",
    "linalg_lstsq",
    "linalg_matrix_norm",
    "linalg_matrix_norm_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_norm",
    "linalg_qr",
    "linalg_qr_out",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linalg_svdvals",
    "linear",
    "log10_",
    "log_normal",
    "log_normal_",
    "matmul_bf16",
    "matmul_int8",
    "max_pool3d_with_indices_backward",
    "median",
    "mm",
    "mm_out",
    "mvlgamma",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "narrow_copy",
    "nonzero_numpy",
    "pad_sequence",
    "permute_copy",
    "randperm",
    "renorm_",
    "repeat",
    "repeat_interleave_self_int",
    "resolve_neg",
    "run",
    "scatter_add_",
    "softplus_backward",
    "sort",
    "sort_stable",
    "sparse_sampled_addmm",
    "sparse_sampled_addmm_out",
    "special_bessel_j0",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_gammainc",
    "special_hermite_polynomial_h",
    "special_legendre_polynomial_p",
    "special_modified_bessel_k0",
    "special_modified_bessel_k0_out",
    "special_modified_bessel_k1",
    "special_modified_bessel_k1_out",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_w",
    "tile",
    "var",
    "var_correction",
    "var_dim",
]
