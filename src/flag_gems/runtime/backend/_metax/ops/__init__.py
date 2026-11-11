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
from .bmm import bmm, bmm_out
from .bmm_w8a8 import bmm_w8a8
from .cholesky_inverse import cholesky_inverse
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .conv_depthwise2d import _conv_depthwise2d
from .conv_tbc import conv_tbc
from .conv_tbc_backward import conv_tbc_backward
from .conv_transpose1d import conv_transpose1d, conv_transpose1d_output_size
from .convolution_double_backward import _convolution_double_backward
from .cross_attention import cross_attention
from .dequantize import dequantize
from .erfinv import erfinv
from .erfinv_ import erfinv_
from .exponential_ import exponential_
from .fake_quantize_per_tensor_affine_cachemask_backward import (
    fake_quantize_per_tensor_affine_cachemask_backward,
)
from .float_power_ import float_power_tensor_scalar_, float_power_tensor_tensor_
from .foreach_abs import foreach_abs
from .foreach_abs_ import foreach_abs_
from .foreach_acos_ import foreach_acos_
from .foreach_add_list import foreach_add_list
from .foreach_add_list_ import foreach_add_list_
from .foreach_add_scalar import foreach_add_scalar
from .foreach_add_scalar_ import foreach_add_scalar_
from .foreach_add_scalar_list import foreach_add_scalar_list
from .foreach_add_scalar_list_ import foreach_add_scalar_list_
from .foreach_addcdiv_scalar_ import foreach_addcdiv_scalar_
from .foreach_addcdiv_scalar_list_ import foreach_addcdiv_scalar_list_
from .foreach_addcdiv_tensor_ import foreach_addcdiv_tensor_
from .foreach_addcmul_scalar_ import foreach_addcmul_scalar_
from .foreach_addcmul_scalar_list_ import foreach_addcmul_scalar_list_
from .foreach_asin_ import foreach_asin_
from .foreach_atan import foreach_atan
from .foreach_atan_ import foreach_atan_
from .foreach_ceil import foreach_ceil
from .foreach_clamp_max_list import foreach_clamp_max_list
from .foreach_clamp_max_list_ import foreach_clamp_max_list_
from .foreach_clamp_max_scalar import foreach_clamp_max_scalar
from .foreach_clamp_max_scalar_ import foreach_clamp_max_scalar_
from .foreach_clamp_max_scalar_list import foreach_clamp_max_scalar_list
from .foreach_clamp_max_scalar_list_ import foreach_clamp_max_scalar_list_
from .foreach_clamp_min_list import foreach_clamp_min_list
from .foreach_clamp_min_list_ import foreach_clamp_min_list_
from .foreach_clamp_min_scalar import foreach_clamp_min_scalar
from .foreach_clamp_min_scalar_ import foreach_clamp_min_scalar_
from .foreach_clamp_min_scalar_list import foreach_clamp_min_scalar_list
from .foreach_clamp_min_scalar_list_ import foreach_clamp_min_scalar_list_
from .foreach_copy_ import foreach_copy_
from .foreach_cos import foreach_cos
from .foreach_cos_ import foreach_cos_
from .foreach_cosh import foreach_cosh
from .foreach_cosh_ import foreach_cosh_
from .foreach_div_list_ import foreach_div_list_
from .foreach_div_scalar import foreach_div_scalar
from .foreach_div_scalar_ import foreach_div_scalar_
from .foreach_div_scalar_list import foreach_div_scalar_list
from .foreach_div_scalar_list_ import foreach_div_scalar_list_
from .foreach_div_tensor import foreach_div_tensor
from .foreach_div_tensor_ import foreach_div_tensor_
from .foreach_erf import foreach_erf
from .foreach_erf_ import foreach_erf_
from .foreach_exp import foreach_exp
from .foreach_exp_ import foreach_exp_
from .foreach_expm1 import foreach_expm1
from .foreach_expm1_ import foreach_expm1_
from .foreach_floor import foreach_floor
from .foreach_floor_ import foreach_floor_
from .foreach_frac import foreach_frac
from .foreach_frac_ import foreach_frac_
from .foreach_log import foreach_log
from .foreach_log1p import foreach_log1p
from .foreach_log2 import foreach_log2
from .foreach_log2_ import foreach_log2_
from .foreach_log10 import foreach_log10
from .foreach_log10_ import foreach_log10_
from .foreach_max import foreach_max
from .foreach_maximum_list import foreach_maximum_list
from .foreach_maximum_list_ import foreach_maximum_list_
from .foreach_maximum_scalar import foreach_maximum_scalar
from .foreach_maximum_scalar_ import foreach_maximum_scalar_
from .foreach_maximum_scalar_list import foreach_maximum_scalar_list
from .foreach_maximum_scalar_list_ import foreach_maximum_scalar_list_
from .foreach_minimum_list import foreach_minimum_list
from .foreach_minimum_list_ import foreach_minimum_list_
from .foreach_minimum_scalar import foreach_minimum_scalar
from .foreach_minimum_scalar_ import foreach_minimum_scalar_
from .foreach_minimum_scalar_list import foreach_minimum_scalar_list
from .foreach_minimum_scalar_list_ import foreach_minimum_scalar_list_
from .foreach_mul_list import foreach_mul_list
from .foreach_mul_list_ import foreach_mul_list_
from .foreach_mul_scalar import foreach_mul_scalar
from .foreach_mul_scalar_ import foreach_mul_scalar_
from .foreach_mul_scalar_list import foreach_mul_scalar_list
from .foreach_mul_scalar_list_ import foreach_mul_scalar_list_
from .foreach_mul_tensor import foreach_mul_tensor
from .foreach_mul_tensor_ import foreach_mul_tensor_
from .foreach_neg import foreach_neg
from .foreach_neg_ import foreach_neg_
from .foreach_norm_scalar import foreach_norm_scalar
from .foreach_pow_scalar_ import foreach_pow_scalar_
from .foreach_pow_scalar_and_tensor import foreach_pow_scalar_and_tensor
from .foreach_pow_scalar_list_ import foreach_pow_scalar_list_
from .foreach_reciprocal import foreach_reciprocal
from .foreach_reciprocal_ import foreach_reciprocal_
from .foreach_round_ import foreach_round_
from .foreach_rsqrt import foreach_rsqrt
from .foreach_rsqrt_ import foreach_rsqrt_
from .foreach_sigmoid import foreach_sigmoid
from .foreach_sigmoid_ import foreach_sigmoid_
from .foreach_sin import foreach_sin
from .foreach_sin_ import foreach_sin_
from .foreach_sinh import foreach_sinh
from .foreach_sinh_ import foreach_sinh_
from .foreach_sqrt import foreach_sqrt
from .foreach_sqrt_ import foreach_sqrt_
from .foreach_sub_list import foreach_sub_list
from .foreach_sub_list_ import foreach_sub_list_
from .foreach_sub_scalar import foreach_sub_scalar
from .foreach_sub_scalar_ import foreach_sub_scalar_
from .foreach_sub_scalar_list import foreach_sub_scalar_list
from .foreach_sub_scalar_list_ import foreach_sub_scalar_list_
from .foreach_tan import foreach_tan
from .foreach_tan_ import foreach_tan_
from .foreach_tanh import foreach_tanh
from .foreach_tanh_ import foreach_tanh_
from .foreach_trunc import foreach_trunc
from .foreach_trunc_ import foreach_trunc_
from .fractional_max_pool2d_backward import fractional_max_pool2d_backward
from .full import full
from .full_like import full_like
from .gcd_ import gcd_
from .greater_equal import greater_equal_
from .grid_sampler_3d import grid_sampler_3d
from .grouped_mm_w8a8 import grouped_mm_w8a8
from .groupnorm import group_norm
from .gru import gru, gru_data
from .gt_scalar_ import gt_scalar_
from .gt_tensor_ import gt_tensor_
from .hadamard_transform import hadamard_transform
from .histc import histc
from .index import index
from .index_put import index_put, index_put_
from .index_reduce import index_reduce, index_reduce_, index_reduce_out
from .index_select import index_select
from .inner import inner
from .inverse import inverse
from .isin import isin
from .kthvalue import kthvalue
from .layernorm import layer_norm, layer_norm_backward
from .lcm import lcm
from .lcm_ import lcm_
from .ldexp_ import ldexp_
from .lgamma_ import lgamma, lgamma_
from .lift_fresh import lift_fresh
from .linalg_cholesky import linalg_cholesky
from .linalg_ldl_factor import ldl_factor
from .linalg_ldl_factor_ex import ldl_factor_ex
from .linalg_lstsq import linalg_lstsq
from .linalg_matrix_exp import linalg_matrix_exp, linalg_matrix_exp_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_qr import linalg_qr, linalg_qr_out
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linalg_svdvals import linalg_svdvals
from .linear import linear
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
from .masked_select_backward import masked_select_backward
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max_pool3d_with_indices_backward import max_pool3d_with_indices_backward
from .min import min, min_dim
from .mm import mm, mm_out

try:
    from .mm_w8a8_int8 import mm_w8a8_int8, mm_w8a8_int8_out
except ImportError:
    # mm_w8a8_int8 is the only module in this package that requires the MetaX
    # triton backend (``triton.backends.metax`` / ``triton.experimental.tle``)
    # at import time.  The package is also reached from non-MetaX backends: the
    # generic ``_nested_view_from_buffer_copy`` forwards to its MetaX
    # implementation, and importing that submodule executes this package first.
    # Skip the MetaX-only module when its backend is absent; the other modules
    # import normally and MetaX behaviour is unchanged.
    pass
from ._make_dep_token import _make_dep_token
from .mode import mode
from .mul import mul, mul_
from .mv import mv
from .mvlgamma import mvlgamma
from .mvlgamma_ import mvlgamma_
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .new_ones import new_ones
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .nuclear_norm import nuclear_norm
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
from .rnn_tanh import rnn_tanh
from .router_gemm import router_gemm
from .rsqrt import rsqrt, rsqrt_
from .scalar_tensor import scalar_tensor
from .scaled_dot_product_attention_math_for_mps import (
    _scaled_dot_product_attention_math_for_mps,
)
from .segment_reduce import segment_reduce, segment_reduce_out
from .sigmoid import sigmoid
from .silu import silu
from .special_bessel_j0 import special_bessel_j0, special_bessel_j0_out
from .special_bessel_y0 import special_bessel_y0
from .special_chebyshev_polynomial_u import special_chebyshev_polynomial_u
from .special_chebyshev_polynomial_v import special_chebyshev_polynomial_v
from .special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w,
    special_chebyshev_polynomial_w_out,
)
from .special_gammainc import special_gammainc
from .special_gammaln import special_gammaln
from .special_gammaln_out import special_gammaln_out
from .special_hermite_polynomial_he import special_hermite_polynomial_he
from .special_legendre_polynomial_p import special_legendre_polynomial_p
from .special_multigammaln import special_multigammaln
from .special_round import special_round
from .special_round_out import special_round_out
from .special_shifted_chebyshev_polynomial_w import (
    special_shifted_chebyshev_polynomial_w,
)
from .standard_gamma_grad import standard_gamma_grad
from .tanh import tanh
from .to_copy import to_copy
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .triangular_solve import triangular_solve, triangular_solve_out
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
    "_convolution_double_backward",
    "_make_dep_token",
    "_nested_view_from_buffer_copy",
    "_scaled_dot_product_attention_math_for_mps",
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
    "bmm_out",
    "bmm_w8a8",
    "cholesky_inverse",
    "cholesky_solve",
    "cholesky_solve_out",
    "conv_tbc",
    "conv_tbc_backward",
    "conv_transpose1d",
    "conv_transpose1d_output_size",
    "cross_attention",
    "dequantize",
    "erfinv",
    "erfinv_",
    "exponential_",
    "fake_quantize_per_tensor_affine_cachemask_backward",
    "float_power_tensor_scalar_",
    "float_power_tensor_tensor_",
    "foreach_abs",
    "foreach_abs_",
    "foreach_acos_",
    "foreach_add_list",
    "foreach_add_list_",
    "foreach_add_scalar",
    "foreach_add_scalar_",
    "foreach_add_scalar_list",
    "foreach_add_scalar_list_",
    "foreach_addcdiv_scalar_",
    "foreach_addcdiv_scalar_list_",
    "foreach_addcdiv_tensor_",
    "foreach_addcmul_scalar_",
    "foreach_addcmul_scalar_list_",
    "foreach_asin_",
    "foreach_atan",
    "foreach_atan_",
    "foreach_ceil",
    "foreach_clamp_max_list",
    "foreach_clamp_max_list_",
    "foreach_clamp_max_scalar",
    "foreach_clamp_max_scalar_",
    "foreach_clamp_max_scalar_list",
    "foreach_clamp_max_scalar_list_",
    "foreach_clamp_min_list",
    "foreach_clamp_min_list_",
    "foreach_clamp_min_scalar",
    "foreach_clamp_min_scalar_",
    "foreach_clamp_min_scalar_list",
    "foreach_clamp_min_scalar_list_",
    "foreach_copy_",
    "foreach_cos",
    "foreach_cos_",
    "foreach_cosh",
    "foreach_cosh_",
    "foreach_div_list_",
    "foreach_div_scalar",
    "foreach_div_scalar_",
    "foreach_div_scalar_list",
    "foreach_div_scalar_list_",
    "foreach_div_tensor",
    "foreach_div_tensor_",
    "foreach_erf",
    "foreach_erf_",
    "foreach_exp",
    "foreach_exp_",
    "foreach_expm1",
    "foreach_expm1_",
    "foreach_floor",
    "foreach_floor_",
    "foreach_frac",
    "foreach_frac_",
    "foreach_log",
    "foreach_log10",
    "foreach_log10_",
    "foreach_log1p",
    "foreach_log2",
    "foreach_log2_",
    "foreach_max",
    "foreach_maximum_list",
    "foreach_maximum_list_",
    "foreach_maximum_scalar",
    "foreach_maximum_scalar_",
    "foreach_maximum_scalar_list",
    "foreach_maximum_scalar_list_",
    "foreach_minimum_list",
    "foreach_minimum_list_",
    "foreach_minimum_scalar",
    "foreach_minimum_scalar_",
    "foreach_minimum_scalar_list",
    "foreach_minimum_scalar_list_",
    "foreach_mul_list",
    "foreach_mul_list_",
    "foreach_mul_scalar",
    "foreach_mul_scalar_",
    "foreach_mul_scalar_list",
    "foreach_mul_scalar_list_",
    "foreach_mul_tensor",
    "foreach_mul_tensor_",
    "foreach_neg",
    "foreach_neg_",
    "foreach_norm_scalar",
    "foreach_pow_scalar_",
    "foreach_pow_scalar_and_tensor",
    "foreach_pow_scalar_list_",
    "foreach_reciprocal",
    "foreach_reciprocal_",
    "foreach_round_",
    "foreach_rsqrt",
    "foreach_rsqrt_",
    "foreach_sigmoid",
    "foreach_sigmoid_",
    "foreach_sin",
    "foreach_sin_",
    "foreach_sinh",
    "foreach_sinh_",
    "foreach_sqrt",
    "foreach_sqrt_",
    "foreach_sub_list",
    "foreach_sub_list_",
    "foreach_sub_scalar",
    "foreach_sub_scalar_",
    "foreach_sub_scalar_list",
    "foreach_sub_scalar_list_",
    "foreach_tan",
    "foreach_tan_",
    "foreach_tanh",
    "foreach_tanh_",
    "foreach_trunc",
    "foreach_trunc_",
    "fractional_max_pool2d_backward",
    "full",
    "full_like",
    "gcd_",
    "greater_equal_",
    "grid_sampler_3d",
    "group_norm",
    "grouped_mm_w8a8",
    "gru",
    "gru_data",
    "gt_scalar_",
    "gt_tensor_",
    "hadamard_transform",
    "histc",
    "index",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_reduce_",
    "index_reduce_out",
    "index_select",
    "inner",
    "inverse",
    "isin",
    "kthvalue",
    "layer_norm",
    "layer_norm_backward",
    "lcm",
    "lcm_",
    "ldexp_",
    "ldl_factor",
    "ldl_factor_ex",
    "lgamma",
    "lgamma_",
    "lift_fresh",
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
    "linear",
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
    "masked_select_backward",
    "matmul_bf16",
    "matmul_int8",
    "max_pool3d_with_indices_backward",
    "min",
    "min_dim",
    "mm",
    "mm_out",
    "mm_w8a8_int8",
    "mm_w8a8_int8_out",
    "mode",
    "mul",
    "mul_",
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
    "nuclear_norm",
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
    "rnn_tanh",
    "router_gemm",
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
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_chebyshev_polynomial_w_out",
    "special_gammainc",
    "special_gammaln",
    "special_gammaln_out",
    "special_hermite_polynomial_he",
    "special_legendre_polynomial_p",
    "special_multigammaln",
    "special_round",
    "special_round_out",
    "special_shifted_chebyshev_polynomial_w",
    "standard_gamma_grad",
    "tanh",
    "to_copy",
    "topk_w8a16_fp8",
    "triangular_solve",
    "triangular_solve_out",
    "upsample_linear1d",
    "upsample_nearest2d",
    "weight_int8pack_mm",
    "zero",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]
