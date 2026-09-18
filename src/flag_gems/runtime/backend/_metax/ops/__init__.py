from ._make_dep_token import _make_dep_token
from ._nested_view_from_buffer_copy import _nested_view_from_buffer_copy
from ._thnn_fused_lstm_cell_backward_impl import _thnn_fused_lstm_cell_backward_impl
from .adaptive_avg_pool2d_backward import _adaptive_avg_pool2d_backward
from .adaptive_max_pool3d_backward import adaptive_max_pool3d_backward
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .addmv_ import addmv_
from .alpha_dropout import alpha_dropout
from .amax import amax
from .arange import arange, arange_start
from .arccosh_ import arccosh_
from .as_strided_scatter import as_strided_scatter
from .avg_pool3d import avg_pool3d_backward
from .baddbmm import baddbmm, baddbmm_out
from .bmm import bmm
from .broadcast_to import broadcast_to
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .conv_depthwise2d import _conv_depthwise2d
from .conv_transpose1d import conv_transpose1d, conv_transpose1d_output_size
from .cudnn_convolution import cudnn_convolution
from .erfinv import erfinv
from .erfinv_ import erfinv_
from .exponential_ import exponential_
from .float_power_ import float_power_tensor_scalar_, float_power_tensor_tensor_
from .full import full
from .full_like import full_like
from .gcd_ import gcd_
from .greater_equal import greater_equal_
from .groupnorm import group_norm
from .gru import gru, gru_data
from .gt_scalar_ import gt_scalar_
from .gt_tensor_ import gt_tensor_
from .hadamard_transform import hadamard_transform
from .index import index
from .index_put import index_put, index_put_
from .index_reduce import index_reduce, index_reduce_, index_reduce_out
from .index_select import index_select
from .isin import isin
from .kthvalue import kthvalue
from .layernorm import layer_norm, layer_norm_backward
from .lcm import lcm
from .lcm_ import lcm_
from .lgamma_ import lgamma, lgamma_
from .linalg_cholesky import linalg_cholesky
from .linalg_ldl_factor import ldl_factor
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_exp import linalg_matrix_exp, linalg_matrix_exp_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_qr import linalg_qr, linalg_qr_out
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linalg_svdvals import linalg_svdvals
from .log_normal_ import log_normal_
from .log_sigmoid_forward import log_sigmoid_forward
from .log_softmax import log_softmax, log_softmax_backward
from .logcumsumexp import logcumsumexp
from .logcumsumexp_out import logcumsumexp_out
from .logical_not_ import logical_not_
from .logical_or import logical_or, logical_or_
from .lt_ import lt_, lt_scalar_
from .masked_fill import masked_fill, masked_fill_
from .masked_scatter import masked_scatter, masked_scatter_, masked_scatter_impl
from .masked_scatter_backward import masked_scatter_backward
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max_pool3d_with_indices_backward import max_pool3d_with_indices_backward
from .min import min, min_dim
from .mm import mm, mm_out
from .mv import mv
from .mvlgamma import mvlgamma
from .mvlgamma_ import mvlgamma_
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .new_ones import new_ones
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .ones import ones
from .ones_like import ones_like
from .ormqr import ormqr
from .outer import outer
from .polar import polar
from .prod import prod, prod_dim
from .renorm import renorm, renorm_
from .repeat import repeat
from .repeat_interleave import repeat_interleave_self_tensor
from .replication_pad3d_backward import replication_pad3d_backward
from .resolve_conj import resolve_conj
from .rms_norm_w8a16_fp8 import rms_norm_w8a16_fp8
from .rsqrt import rsqrt, rsqrt_
from .scalar_tensor import scalar_tensor
from .segment_reduce import segment_reduce, segment_reduce_out
from .sigmoid import sigmoid
from .silu import silu
from .special_bessel_j0 import special_bessel_j0, special_bessel_j0_out
from .special_bessel_y0 import special_bessel_y0
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_gammainc import special_gammainc
from .special_gammaln import special_gammaln
from .special_gammaln_out import special_gammaln_out
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_multigammaln import special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .tanh import tanh
from .to_copy import to_copy
from .unique import _unique2
from .unsafe_masked_index_put_accumulate import _unsafe_masked_index_put_accumulate
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest_exact2d_backward import _upsample_nearest_exact2d_backward
from .weight_int8pack_mm import weight_int8pack_mm
from .zero import zero, zero_, zero_out
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "_adaptive_avg_pool2d_backward",
    "_conv_depthwise2d",
    "_make_dep_token",
    "_nested_view_from_buffer_copy",
    "_thnn_fused_lstm_cell_backward_impl",
    "_unique2",
    "_unsafe_masked_index_put_accumulate",
    "_upsample_nearest_exact2d_backward",
    "adaptive_max_pool3d_backward",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "addmv_",
    "alpha_dropout",
    "amax",
    "arange",
    "arange_start",
    "arccosh_",
    "as_strided_scatter",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_out",
    "bmm",
    "broadcast_to",
    "cholesky_inverse",
    "cholesky_solve",
    "cholesky_solve_out",
    "conv_transpose1d",
    "conv_transpose1d_output_size",
    "cudnn_convolution",
    "erfinv",
    "erfinv_",
    "exponential_",
    "float_power_tensor_scalar_",
    "float_power_tensor_tensor_",
    "full",
    "full_like",
    "gcd_",
    "greater_equal_",
    "group_norm",
    "gru",
    "gru_data",
    "gt_scalar_",
    "gt_tensor_",
    "hadamard_transform",
    "index",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_reduce_",
    "index_reduce_out",
    "index_select",
    "isin",
    "kthvalue",
    "layer_norm",
    "layer_norm_backward",
    "lcm",
    "lcm_",
    "ldl_factor",
    "lgamma",
    "lgamma_",
    "linalg_cholesky",
    "linalg_lstsq",
    "linalg_matrix_exp",
    "linalg_matrix_exp_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_qr",
    "linalg_qr_out",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linalg_svdvals",
    "log_normal_",
    "log_sigmoid_forward",
    "log_softmax",
    "log_softmax_backward",
    "logcumsumexp",
    "logcumsumexp_out",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "lt_",
    "lt_scalar_",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_scatter_backward",
    "masked_scatter_impl",
    "matmul_bf16",
    "matmul_int8",
    "max_pool3d_with_indices_backward",
    "min",
    "min_dim",
    "mm",
    "mm_out",
    "mv",
    "mvlgamma",
    "mvlgamma_",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "new_ones",
    "nonzero",
    "nonzero_numpy",
    "ones",
    "ones_like",
    "ormqr",
    "outer",
    "polar",
    "prod",
    "prod_dim",
    "renorm",
    "renorm_",
    "repeat",
    "repeat_interleave_self_tensor",
    "replication_pad3d_backward",
    "resolve_conj",
    "rms_norm_w8a16_fp8",
    "rsqrt",
    "rsqrt_",
    "scalar_tensor",
    "segment_reduce",
    "segment_reduce_out",
    "sigmoid",
    "silu",
    "special_bessel_j0",
    "special_bessel_j0_out",
    "special_bessel_y0",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_gammainc",
    "special_gammaln",
    "special_gammaln_out",
    "special_legendre_polynomial_p",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_w",
    "tanh",
    "to_copy",
    "upsample_linear1d",
    "upsample_nearest2d",
    "weight_int8pack_mm",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]
