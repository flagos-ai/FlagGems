from ._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size,
)
from ._safe_softmax import _safe_softmax
from .abs import abs, abs_
from .absolute import absolute
from .acos import acos
from .add import add, add_
from .addcdiv import addcdiv, addcdiv_out
from .addcmul import addcmul, addcmul_out
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .alias_copy import alias_copy, alias_copy_out
from .all import all, all_dim, all_dims
from .amax import amax
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arcsinh import arcsinh, arcsinh_out
from .arcsinh_ import arcsinh_
from .arctanh_ import arctanh_
from .argmax import argmax
from .argsort import argsort
from .asinh_ import asinh_
from .assert_async import _assert_async
from .atan import atan, atan_
from .attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .avg_pool2d import avg_pool2d, avg_pool2d_backward
from .avg_pool3d import avg_pool3d, avg_pool3d_backward
from .bernoulli_ import bernoulli_
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_left_shift import bitwise_left_shift
from .bitwise_not import bitwise_not, bitwise_not_
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .bitwise_right_shift import bitwise_right_shift
from .bmm import bmm, bmm_out
from .cat import cat, cat_out
from .cauchy import cauchy, cauchy_
from .ceil import ceil, ceil_, ceil_out
from .celu import celu, celu_
from .clamp import clamp, clamp_, clamp_min, clamp_min_, clamp_tensor, clamp_tensor_
from .concatenate import concatenate
from .conj_physical import conj_physical
from .contiguous import contiguous
from .copy import copy, copy_
from .cos import cos, cos_
from .count_nonzero import count_nonzero
from .cummin import cummin
from .cumsum import cumsum, cumsum_out, normed_cumsum
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .digamma_ import digamma_
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    true_divide,
    true_divide_,
    true_divide_out,
)
from .dropout import dropout, dropout_backward
from .elu import elu, elu_, elu_backward
from .embedding import embedding, embedding_backward
from .embedding_dense_backward import embedding_dense_backward
from .eq import eq, eq_scalar, equal
from .erf import erf, erf_
from .exp import exp, exp_, exp_out
from .exp2 import exp2, exp2_
from .exponential_ import exponential_
from .feature_dropout import feature_dropout, feature_dropout_
from .fill import (
    fill_scalar,
    fill_scalar_,
    fill_scalar_out,
    fill_tensor,
    fill_tensor_,
    fill_tensor_out,
)
from .flip import flip
from .floor_ import floor_
from .fmin import fmin, fmin_out
from .full import full
from .full_like import full_like
from .gather import gather, gather_backward
from .gcd import gcd, gcd_out
from .ge import ge, ge_scalar
from .gelu import gelu, gelu_, gelu_backward
from .glu import glu, glu_backward
from .grid_sample import grid_sample
from .groupnorm import group_norm, group_norm_backward
from .gt import gt, gt_scalar
from .hardsigmoid import hardsigmoid, hardsigmoid_out
from .hardswish_ import hardswish_
from .histc import histc
from .hstack import hstack
from .hypot import hypot, hypot_out
from .i0 import i0, i0_out
from .i0_ import i0_
from .index import index
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_put import _index_put_impl_, index_put, index_put_
from .index_reduce import index_reduce_
from .index_select import index_select
from .isclose import allclose, isclose
from .isfinite import isfinite
from .isin import isin
from .isinf import isinf
from .isnan import isnan
from .kron import kron
from .layernorm import layer_norm, layer_norm_backward
from .le import le, le_scalar
from .leaky_relu import leaky_relu, leaky_relu_, leaky_relu_out
from .lift_fresh_copy import lift_fresh_copy, lift_fresh_copy_out
from .linspace import linspace
from .log import log
from .log1p_ import log1p_
from .log_sigmoid import log_sigmoid
from .log_softmax import (
    log_softmax,
    log_softmax_backward,
    log_softmax_backward_out,
    log_softmax_out,
)
from .logical_and import logical_and, logical_and_
from .logical_not import logical_not
from .logical_or import logical_or, logical_or_
from .logical_xor import logical_xor
from .logit import logit, logit_out
from .logit_ import logit_
from .logspace import logspace
from .logsumexp import logsumexp
from .lt import lt, lt_scalar
from .margin_ranking_loss import margin_ranking_loss
from .masked_fill import masked_fill, masked_fill_
from .masked_select import masked_select
from .max import max, max_dim
from .max_pool2d_with_indices import max_pool2d_backward, max_pool2d_with_indices
from .maximum import maximum
from .mean import mean, mean_dim
from .min import min, min_dim
from .minimum import minimum
from .mm import mm, mm_out, router_gemm
from .mul import mul, mul_
from .multinomial import multinomial
from .mv import mv
from .nan_to_num import nan_to_num
from .ne import ne, ne_scalar
from .neg import neg, neg_
from .nonzero import nonzero
from .nonzero_numpy import nonzero_numpy
from .normal import (
    normal_,
    normal_float_tensor,
    normal_tensor_float,
    normal_tensor_tensor,
)
from .ones import ones
from .ones_like import ones_like
from .pad import constant_pad_nd, pad
from .per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE, per_token_group_quant_fp8
from .poisson import poisson
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .prelu import prelu
from .prod import prod, prod_dim
from .quantile import quantile
from .rand import rand
from .rand_like import rand_like
from .randint import randint
from .randint_like import randint_like
from .randn import randn
from .randn_like import randn_like
from .randperm import randperm
from .reciprocal import reciprocal, reciprocal_
from .relu import relu, relu_
from .relu6 import relu6
from .remainder import remainder, remainder_
from .repeat import repeat
from .repeat_interleave import (
    repeat_interleave_self_int,
    repeat_interleave_self_tensor,
    repeat_interleave_tensor,
)
from .resolve_conj import resolve_conj
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward
from .roll import roll
from .round import round, round_, round_out
from .rrelu_with_noise_backward import rrelu_with_noise_backward
from .rsqrt import rsqrt, rsqrt_
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .select_scatter import select_scatter
from .selu import selu
from .selu_ import selu_
from .sgn_ import sgn_
from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
from .silu import silu, silu_, silu_backward
from .sin import sin, sin_
from .sinh_ import sinh_
from .slice_backward import slice_backward
from .slice_scatter import slice_scatter
from .soft_margin_loss import soft_margin_loss
from .softmax import softmax, softmax_backward, softmax_backward_out, softmax_out
from .softplus import softplus
from .softshrink import softshrink, softshrink_out
from .sort import sort, sort_stable
from .special_i0e import special_i0e, special_i0e_out
from .special_i1 import special_i1, special_i1_out
from .sqrt import sqrt, sqrt_
from .stack import stack
from .sub import sub, sub_
from .sum import sum, sum_dim, sum_dim_out, sum_out
from .svd import svd
from .tan import tan, tan_
from .tanh import tanh, tanh_, tanh_backward
from .threshold import threshold, threshold_backward
from .tile import tile
from .to import to_copy
from .topk import topk
from .triu import triu, triu_
from .uniform import uniform_
from .unique import _unique2
from .unique_consecutive import unique_consecutive
from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward
from .upsample_linear1d import upsample_linear1d
from .upsample_nearest2d import upsample_nearest2d
from .var import var, var_correction, var_dim
from .var_mean import var_mean
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack import vstack
from .weightnorm import weight_norm_interface, weight_norm_interface_backward
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .zero import zero, zero_out
from .zeros import zero_, zeros
from .zeros_like import zeros_like

__all__ = [
    "_functional_sym_constrain_range_for_size",
    "_index_put_impl_",
    "_safe_softmax",
    "_unique2",
    "_upsample_bicubic2d_aa_backward",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_out",
    "addcmul",
    "addcmul_out",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "alias_copy",
    "alias_copy_out",
    "all",
    "all_dim",
    "all_dims",
    "allclose",
    "amax",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arcsinh",
    "arcsinh_",
    "arcsinh_out",
    "arctanh_",
    "argmax",
    "argsort",
    "asinh_",
    "_assert_async",
    "atan",
    "atan_",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "bernoulli_",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_right_shift",
    "bmm",
    "bmm_out",
    "cat",
    "cat_out",
    "cauchy",
    "cauchy_",
    "ceil",
    "ceil_",
    "ceil_out",
    "celu",
    "celu_",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_tensor",
    "clamp_tensor_",
    "concatenate",
    "conj_physical",
    "contiguous",
    "copy",
    "copy_",
    "cos",
    "cos_",
    "count_nonzero",
    "constant_pad_nd",
    "cummin",
    "cumsum",
    "cumsum_out",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "digamma_",
    "div_mode",
    "div_mode_",
    "dropout",
    "dropout_backward",
    "elu",
    "elu_",
    "elu_backward",
    "erf",
    "erf_",
    "embedding",
    "embedding_backward",
    "embedding_dense_backward",
    "eq",
    "eq_scalar",
    "equal",
    "exp",
    "exp_",
    "exp_out",
    "exp2",
    "exp2_",
    "exponential_",
    "feature_dropout",
    "feature_dropout_",
    "fill_scalar",
    "fill_tensor",
    "fill_scalar_",
    "fill_tensor_",
    "fill_scalar_out",
    "fill_tensor_out",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
    "flip",
    "floor_",
    "floor_divide",
    "floor_divide_",
    "fmin",
    "fmin_out",
    "full",
    "full_like",
    "gather",
    "gather_backward",
    "gcd",
    "gcd_out",
    "grid_sample",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_",
    "gelu_backward",
    "get_specific_ops",  # FIXME
    "get_unused_ops",  # FIXME
    "glu",
    "glu_backward",
    "group_norm",
    "group_norm_backward",
    "gt",
    "gt_scalar",
    "hardsigmoid",
    "hardsigmoid_out",
    "hardswish_",
    "histc",
    "hstack",
    "hypot",
    "hypot_out",
    "i0",
    "i0_",
    "i0_out",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_reduce_",
    "index_select",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "kron",
    "layer_norm",
    "layer_norm_backward",
    "le",
    "le_scalar",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_out",
    "lift_fresh_copy",
    "lift_fresh_copy_out",
    "linspace",
    "log",
    "log1p_",
    "log_sigmoid",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_backward_out",
    "log_softmax_out",
    "logical_or",
    "logical_or_",
    "logical_and",
    "logical_and_",
    "logical_xor",
    "logical_not",
    "logit",
    "logit_out",
    "logit_",
    "logspace",
    "logsumexp",
    "lt",
    "lt_scalar",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "max",
    "max_dim",
    "max_pool2d_backward",
    "max_pool2d_with_indices",
    "maximum",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "minimum",
    "mm",
    "mm_out",
    "router_gemm",
    "mul",
    "mul_",
    "multinomial",
    "mv",
    "nan_to_num",
    "ne",
    "ne_scalar",
    "neg",
    "neg_",
    "nonzero",
    "nonzero_numpy",
    "normal_",
    "normal_float_tensor",
    "normal_tensor_float",
    "normal_tensor_tensor",
    "normed_cumsum",
    "ones",
    "ones_like",
    "pad",
    "per_token_group_quant_fp8",
    "poisson",
    "prod",
    "prod_dim",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "prelu",
    "quantile",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "relu6",
    "remainder",
    "remainder_",
    "repeat",
    "repeat_interleave_self_int",
    "repeat_interleave_self_tensor",
    "repeat_interleave_tensor",
    "resolve_neg",
    "resolve_conj",
    "rms_norm",
    "rms_norm_backward",
    "rms_norm_forward",
    "roll",
    "round",
    "round_",
    "round_out",
    "rrelu_with_noise_backward",
    "rsqrt",
    "rsqrt_",
    "ScaleDotProductAttention",
    "SUPPORTED_FP8_DTYPE",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "select_scatter",
    "selu",
    "selu_",
    "sgn_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinh_",
    "slice_backward",
    "slice_scatter",
    "soft_margin_loss",
    "softmax",
    "softmax_backward",
    "softmax_backward_out",
    "softmax_out",
    "softplus",
    "softshrink",
    "softshrink_out",
    "sort",
    "sort_stable",
    "special_i0e",
    "special_i0e_out",
    "special_i1",
    "special_i1_out",
    "sqrt",
    "sqrt_",
    "stack",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "sum_out",
    "svd",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "to_copy",
    "topk",
    "tile",
    "triu",
    "triu_",
    "true_divide",
    "true_divide_",
    "true_divide_out",
    "uniform_",
    "unique_consecutive",
    "upsample_linear1d",
    "upsample_nearest2d",
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    "vector_norm",
    "view_copy",
    "vstack",
    "weight_norm_interface",
    "weight_norm_interface_backward",
    "where_self",
    "where_self_out",
    "where_scalar_other",
    "where_scalar_self",
    "zero",
    "threshold",
    "threshold_backward",
    "zero_",
    "zero_out",
    "zeros",
    "zeros_like",
]
