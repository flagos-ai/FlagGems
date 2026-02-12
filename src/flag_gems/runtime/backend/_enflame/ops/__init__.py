from ..utils.config_utils import arch_version

__all__ = []

if arch_version == 300:
    from .gcu300.mean import mean_dim
    from .gcu300.mean import mean
    from .gcu300.zeros import zeros
    from .gcu300.scatter import scatter_, scatter
    from .gcu300.sort import sort, sort_stable
    from .gcu300.cat import cat
    from .gcu300.addmm import addmm
    from .gcu300.bmm import bmm
    from .gcu300.mm import mm
    from .gcu300.mv import mv
    from .gcu300.arange import arange, arange_start
    from .gcu300.embedding import embedding
    from .gcu300.multinomial import multinomial
    from .gcu300.repeat_interleave import repeat_interleave_self_tensor, repeat_interleave_tensor, repeat_interleave_self_int
    from .gcu300.argmax import argmax
    from .gcu300.argmin import argmin
    from .gcu300.exponential_ import exponential_
    from .gcu300.gather import gather
    from .gcu300.gt import gt, gt_scalar
    from .gcu300.index_select import index_select
    from .gcu300.index import index
    from .gcu300.isin import isin
    from .gcu300.max import max, max_dim
    from .gcu300.min import min, min_dim
    from .gcu300.sum import sum, sum_out, sum_dim_out, sum_dim
    from .gcu300.full import full
    from .gcu300.ones import ones
    from .gcu300.abs import abs, abs_
    from .gcu300.add import add, add_
    from .gcu300.angle import angle
    from .gcu300.bitwise_and import bitwise_and_scalar, bitwise_and_scalar_, bitwise_and_scalar_tensor, bitwise_and_tensor, bitwise_and_tensor_
    from .gcu300.bitwise_not import bitwise_not, bitwise_not_
    from .gcu300.bitwise_or import bitwise_or_scalar, bitwise_or_scalar_, bitwise_or_scalar_tensor, bitwise_or_tensor, bitwise_or_tensor_
    from .gcu300.bitwise_xor import bitwise_xor_scalar, bitwise_xor_scalar_, bitwise_xor_scalar_tensor, bitwise_xor_tensor, bitwise_xor_tensor_
    from .gcu300.clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
    from .gcu300.copy import copy
    from .gcu300.copy import copy_
    from .gcu300.cos import cos
    from .gcu300.count_nonzero import count_nonzero
    from .gcu300.diag_embed import diag_embed
    from .gcu300.div import true_divide , true_divide_, trunc_divide_, trunc_divide, floor_divide, floor_divide_, remainder, remainder_
    from .gcu300.elu import elu
    from .gcu300.eq import eq_scalar, eq
    from .gcu300.erf import erf, erf_
    from .gcu300.exp import exp, exp_
    from .gcu300.fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
    from .gcu300.flip import flip
    from .gcu300.ge import ge, ge_scalar
    from .gcu300.gelu import gelu_backward, gelu_, gelu
    from .gcu300.glu import glu
    from .gcu300.isclose import isclose, allclose
    from .gcu300.isfinite import isfinite
    from .gcu300.isinf import isinf
    from .gcu300.isnan import isnan
    from .gcu300.le import le_scalar, le
    from .gcu300.lerp import lerp_tensor_, lerp_tensor, lerp_scalar, lerp_scalar_
    from .gcu300.log_sigmoid import log_sigmoid
    from .gcu300.log import log
    from .gcu300.logical_and import logical_and
    from .gcu300.logical_not import logical_not
    from .gcu300.logical_or import logical_or
    from .gcu300.logical_xor import logical_xor
    from .gcu300.lt import lt_scalar, lt
    from .gcu300.maximum import maximum
    from .gcu300.minimum import minimum
    from .gcu300.mul import mul, mul_
    from .gcu300.nan_to_num import nan_to_num
    from .gcu300.ne import ne_scalar, ne
    from .gcu300.neg import neg, neg_
    from .gcu300.normal import normal_tensor_tensor, normal_tensor_float, normal_float_tensor
    from .gcu300.polar import polar
    from .gcu300.pow import pow_tensor_tensor, pow_tensor_tensor_, pow_tensor_scalar, pow_tensor_scalar_, pow_scalar
    from .gcu300.reciprocal import reciprocal, reciprocal_
    from .gcu300.relu import relu, relu_
    from .gcu300.repeat import repeat
    from .gcu300.rsqrt import rsqrt, rsqrt_
    from .gcu300.sigmoid import sigmoid_backward, sigmoid_, sigmoid
    from .gcu300.silu import silu_backward, silu, silu_
    from .gcu300.sin import sin, sin_
    from .gcu300.sub import sub, sub_
    from .gcu300.tanh import tanh_backward, tanh, tanh_
    from .gcu300.threshold import threshold_backward, threshold
    from .gcu300.tile import tile
    from .gcu300.upsample_nearest2d import upsample_nearest2d
    from .gcu300.where import where_self_out, where_self, where_scalar_self, where_scalar_other
    from .gcu300.contiguous import contiguous
    from .gcu300.masked_fill import masked_fill, masked_fill_
    from .gcu300.masked_select import masked_select
    from .gcu300.bitwise_left_shift import bitwise_left_shift, bitwise_left_shift_
    from .gcu300.bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_
    from .gcu300.outer import outer
    from .gcu300.diagonal import diagonal_backward
    from .gcu300.topk import topk
    from .gcu300.eye import eye
    from .gcu300.eye_m import eye_m
    from .gcu300.pad import pad
    from .gcu300.log_softmax import log_softmax
    from .gcu300.count_nonzero import count_nonzero
    from .gcu300.linspace import linspace
    from .gcu300.var_mean import var_mean
    from .gcu300.slice_scatter import slice_scatter
    from .gcu300.select_scatter import select_scatter
    from .gcu300.ones_like import ones_like
    from .gcu300.prod import prod, prod_dim
    from .gcu300.zeros_like import zeros_like
    from .gcu300.rand import rand
    from .gcu300.rand_like import rand_like
    from .gcu300.randn import randn
    from .gcu300.randn_like import randn_like
    from .gcu300.randperm import randperm
    from .gcu300.cumsum import normed_cumsum, cumsum, cumsum_out
    from .gcu300.nonzero import nonzero
    from .gcu300.uniform import uniform_
    from .gcu300.cummin import cummin
    from .gcu300.unique import simple_unique_flat, _unique2, sorted_indices_unique_flat
    from .gcu300.dropout import dropout
    from .gcu300.cummax import cummax
    from .gcu300.index_put import index_put, index_put_
    from .gcu300.vstack import vstack
    from .gcu300.all import all, all_dim, all_dims
    from .gcu300.amax import amax
    from .gcu300.groupnorm import group_norm, group_norm_backward
    from .gcu300.layernorm import layer_norm, layer_norm_backward
    from .gcu300.to import to_copy
    from .gcu300.any import any, any_dim, any_dims
    __all__ = [
        "mean_dim",
        "mean",
        "zeros",
        "scatter",
        "scatter_",
        "sort",
        "sort_stable",
        "cat",
        "addmm",
        "bmm",
        "mm",
        "mv",
        "arange",
        "embedding",
        "multinomial",
        "repeat_interleave_self_tensor",
        "repeat_interleave_tensor",
        "repeat_interleave_self_int",
        "argmax",
        "argmin",
        "exponential_",
        "gather",
        "gt",
        "gt_scalar",
        "index_select",
        "index",
        "isin",
        "max",
        "max_dim",
        "min",
        "min_dim",
        "sum",
        "sum_out",
        "sum_dim_out",
        "sum_dim",
        "full",
        "abs",
        "abs_",
        "add",
        "add_",
        "angle",
        "bitwise_and_scalar",
        "bitwise_and_scalar_",
        "bitwise_and_scalar_tensor",
        "bitwise_and_tensor",
        "bitwise_and_tensor_",
        "bitwise_not",
        "bitwise_not",
        "bitwise_or_scalar",
        "bitwise_or_scalar_",
        "bitwise_or_scalar_tensor",
        "bitwise_or_tensor",
        "bitwise_or_tensor_",
        "bitwise_xor_scalar",
        "bitwise_xor_scalar_",
        "bitwise_xor_scalar_tensor",
        "bitwise_xor_tensor",
        "bitwise_xor_tensor_",
        "clamp",
        "clamp_",
        "clamp_tensor",
        "clamp_tensor_",
        "copy",
        "copy_",
        "cos",
        "cos_",
        "count_nonzero",
        "diag_embed",
        "true_divide",
        "true_divide_",
        "trunc_divide_",
        "trunc_divide",
        "floor_divide",
        "floor_divide_",
        "remainder",
        "remainder_",
        "elu",
        "eq_scalar",
        "eq",
        "erf",
        "erf_",
        "exp",
        "exp_",
        "fill_scalar",
        "fill_scalar_",
        "fill_tensor",
        "fill_tensor_",
        "flip",
        "ge",
        "ge_scalar",
        "gelu_backward",
        "gelu_",
        "gelu",
        "glu",
        "isclose",
        "allclose",
        "isfinite",
        "isinf",
        "isnan",
        "le_scalar",
        "le",
        "lerp_tensor_",
        "lerp_tensor",
        "lerp_scalar",
        "lerp_scalar_",
        "log_sigmoid",
        "log",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "lt_scalar",
        "lt",
        "maximum",
        "minimum",
        "mul",
        "mul_",
        "nan_to_num",
        "ne_scalar",
        "ne",
        "neg",
        "neg_",
        "normal_tensor_tensor",
        "normal_tensor_float",
        "normal_float_tensor",
        "polar",
        "pow_tensor_tensor",
        "pow_tensor_tensor_",
        "pow_tensor_scalar",
        "pow_tensor_scalar_",
        "pow_scalar",
        "reciprocal",
        "reciprocal_",
        "relu",
        "relu_",
        "repeat",
        "rsqrt",
        "rsqrt_",
        "sigmoid_backward",
        "sigmoid_",
        "sigmoid",
        "silu_backward",
        "silu",
        "silu_",
        "sin",
        "sin_",
        "sub",
        "sub_",
        "tanh_backward",
        "tanh",
        "tanh_",
        "threshold_backward",
        "threshold",
        "tile",
        "upsample_nearest2d",
        "where_self_out",
        "where_self",
        "where_scalar_self",
        "where_scalar_other",
        "contiguous",
        "masked_fill",
        "masked_fill_",
        "masked_select",
        "bitwise_left_shift",
        "bitwise_left_shift_",
        "bitwise_right_shift",
        "bitwise_right_shift_",
        "outer",
        "diagonal_backward",
        "topk",
        "eye",
        "eye_m",
        "pad",
        "log_softmax",
        "count_nonzero",
        "linspace",
        "var_mean",
        "slice_scatter",
        "select_scatter",
        "ones_like",
        "prod",
        "prod_dim",
        "zeros_like",
        "rand",
        "randn",
        "rand_like",
        "randn_like",
        "randperm",
        "normed_cumsum",
        "cumsum",
        "cumsum_out",
        "nonzero",
        "uniform_",
        "cummin",
        "simple_unique_flat",
        "_unique2",
        "sorted_indices_unique_flat",
        "dropout",
        "cummax",
        "index_put",
        "index_put_",
        "vstack",
        "all",
        "all_dim",
        "all_dims",
        "amax",
        "group_norm",
        "group_norm_backward",
        "layer_norm",
        "layer_norm_backward",
        "to_copy",
        "any",
        "any_dim",
        "any_dims",
        "amax",
        ]
elif arch_version >= 400:
    from .gcu400.mean import mean_dim
    from .gcu400.zeros import zeros
    from .gcu400.scatter import scatter_
    from .gcu400.sort import sort
    from .gcu400.cat import cat
    from .gcu400.mm import mm
    from .gcu400.div import true_divide , true_divide_, trunc_divide_, trunc_divide, floor_divide, floor_divide_, remainder, remainder_
    from .gcu400.add import add, add_
    from .gcu400.bitwise_and import bitwise_and_scalar, bitwise_and_scalar_, bitwise_and_scalar_tensor, bitwise_and_tensor, bitwise_and_tensor_
    from .gcu400.bitwise_or import bitwise_or_scalar, bitwise_or_scalar_, bitwise_or_scalar_tensor, bitwise_or_tensor, bitwise_or_tensor_
    from .gcu400.clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
    from .gcu400.eq import eq_scalar, eq
    from .gcu400.ge import ge, ge_scalar
    from .gcu400.gt import gt, gt_scalar
    from .gcu400.le import le_scalar, le
    from .gcu400.lt import lt_scalar, lt
    from .gcu400.mul import mul, mul_
    from .gcu400.ne import ne_scalar, ne
    from .gcu400.pow import pow_tensor_tensor, pow_tensor_tensor_, pow_tensor_scalar, pow_tensor_scalar_, pow_scalar
    from .gcu400.maximum import maximum
    from .gcu400.minimum import minimum
    from .gcu400.sub import sub, sub_
    from .gcu400.where import where_self_out, where_self, where_scalar_self, where_scalar_other
    from .gcu400.isclose import isclose, allclose
    from .gcu400.logical_and import logical_and
    from .gcu400.logical_or import logical_or
    from .gcu400.logical_xor import logical_xor
    from .gcu400.threshold import threshold_backward, threshold
    from .gcu400.lerp import lerp_tensor_, lerp_tensor, lerp_scalar, lerp_scalar_
    from .gcu400.masked_fill import masked_fill, masked_fill_
    from .gcu400.masked_select import masked_select
    from .gcu400.fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
    from .gcu400.pad import pad
    from .gcu400.eye import eye
    from .gcu400.eye_m import eye_m
    from .gcu400.cumsum import normed_cumsum, cumsum, cumsum_out
    from .gcu400.multinomial import multinomial
    from .gcu400.isfinite import isfinite
    from .gcu400.isin import isin
    from .gcu400.bitwise_xor import bitwise_xor_scalar, bitwise_xor_scalar_, bitwise_xor_scalar_tensor, bitwise_xor_tensor, bitwise_xor_tensor_
    from .gcu400.bitwise_left_shift import bitwise_left_shift, bitwise_left_shift_
    from .gcu400.bitwise_right_shift import bitwise_right_shift, bitwise_right_shift_
    from .gcu400.log_softmax import log_softmax
    from .gcu400.argmax import argmax
    from .gcu400.unique import sorted_quick_unique_flat, sorted_indices_unique_flat, simple_unique_flat, _unique2
    from .gcu400.upsample_nearest2d import upsample_nearest2d
    from .gcu400.max import max, max_dim
    from .gcu400.rms_norm import rms_norm
    from .gcu400.cummin import cummin
    from .gcu400.index_select import index_select
    from .gcu400.vector_norm import vector_norm
    from .gcu400.cummax import cummax
    from .gcu400.copy import copy, copy_
    from .gcu400.contiguous import contiguous
    from .gcu400.index_add import index_add
    from .gcu400.bmm import bmm
    from .gcu400.diag_embed import diag_embed
    from .gcu400.diagonal import diagonal_backward
    from .gcu400.flip import flip
    from .gcu400.abs import abs, abs_
    from .gcu400.angle import angle
    from .gcu400.bitwise_not import bitwise_not, bitwise_not_
    from .gcu400.cos import cos, cos_
    from .gcu400.diag_embed import diag_embed
    from .gcu400.elu import elu
    from .gcu400.erf import erf, erf_
    from .gcu400.exp import exp, exp_
    from .gcu400.full import full
    from .gcu400.gelu import gelu, gelu_, gelu_backward
    from .gcu400.isinf import isinf
    from .gcu400.isnan import isnan
    from .gcu400.log import log
    from .gcu400.log_sigmoid import log_sigmoid
    from .gcu400.logical_not import logical_not
    from .gcu400.mse_loss import mse_loss
    from .gcu400.nan_to_num import nan_to_num
    from .gcu400.neg import neg, neg_
    from .gcu400.normal import normal_float_tensor, normal_tensor_float, normal_tensor_tensor
    from .gcu400.reciprocal import reciprocal, reciprocal_
    from .gcu400.relu import relu, relu_
    from .gcu400.repeat_interleave import repeat_interleave_self_int, repeat_interleave_self_tensor, repeat_interleave_tensor
    from .gcu400.rsqrt import rsqrt, rsqrt_
    from .gcu400.sigmoid import sigmoid, sigmoid_, sigmoid_backward
    from .gcu400.silu import silu, silu_, silu_backward
    from .gcu400.sin import sin, sin_
    from .gcu400.tanh import tanh, tanh_, tanh_backward
    from .gcu400.to import to_dtype
    from .gcu400.full_like import full_like
    from .gcu400.resolve_neg import resolve_neg
    from .gcu400.linspace import linspace
    from .gcu400.arange import arange, arange_start
    __all__ = [
        "mean_dim",
        "zeros",
        "scatter_",
        "sort",
        "cat",
        "mm",
        "true_divide",
        "true_divide_",
        "trunc_divide_",
        "trunc_divide",
        "floor_divide",
        "floor_divide_",
        "remainder",
        "remainder_",
        "add",
        "add_",
        "bitwise_and_scalar",
        "bitwise_and_scalar_",
        "bitwise_and_scalar_tensor",
        "bitwise_and_tensor",
        "bitwise_and_tensor_",
        "bitwise_or_scalar",
        "bitwise_or_scalar_",
        "bitwise_or_scalar_tensor",
        "bitwise_or_tensor",
        "bitwise_or_tensor_",
        "clamp",
        "clamp_",
        "clamp_tensor",
        "clamp_tensor_",
        "eq_scalar",
        "eq",
        "ge",
        "ge_scalar",
        "gt",
        "gt_scalar",
        "le_scalar",
        "le",
        "lt_scalar",
        "lt",
        "mul",
        "mul_",
        "ne_scalar",
        "ne",
        "pow_tensor_tensor",
        "pow_tensor_tensor_",
        "pow_tensor_scalar",
        "pow_tensor_scalar_",
        "pow_scalar",
        "maximum",
        "minimum",
        "sub",
        "sub_",
        "where_self_out",
        "where_self",
        "where_scalar_self",
        "where_scalar_other",
        "isclose",
        "allclose",
        "logical_and",
        "logical_or",
        "logical_xor",
        "threshold_backward",
        "threshold",
        "polar",
        "lerp_tensor_",
        "lerp_tensor",
        "lerp_scalar",
        "lerp_scalar_",
        "masked_fill",
        "masked_fill_",
        "masked_select",
        "fill_scalar",
        "fill_scalar_",
        "fill_tensor",
        "fill_tensor_",
        "pad",
        "eye",
        "normed_cumsum",
        "cumsum",
        "cumsum_out",
        "multinomial",
        "isfinite",
        "bitwise_xor_scalar",
        "bitwise_xor_scalar_",
        "bitwise_xor_scalar_tensor",
        "bitwise_xor_tensor",
        "bitwise_xor_tensor_",
        "bitwise_left_shift",
        "bitwise_left_shift_",
        "bitwise_right_shift",
        "bitwise_right_shift_",
        "log_softmax",
        "argmax",
        "sorted_quick_unique_flat",
        "sorted_indices_unique_flat",
        "simple_unique_flat",
        "_unique2",
        "upsample_nearest2d",
        "max",
        "max_dim",
        "rms_norm",
        "cummin",
        "index_select",
        "vector_norm",
        "cummax",
        "copy", 
        "copy_",
        "contiguous",
        "eye_m",
        "index_add",
        "bmm",
        "diag_embed",
        "diagonal_backward",
        "flip",
        "abs",
        "abs_",
        "angle",
        "bitwise_not",
        "bitwise_not_",
        "cos",
        "cos_",
        "diag_embed",
        "elu",
        "erf",
        "erf_",
        "exp",
        "exp_",
        "full",
        "gelu",
        "gelu_",
        "gelu_backward",
        "isinf",
        "isnan",
        "log",
        "log_sigmoid",
        "logical_not",
        "mse_loss",
        "nan_to_num",
        "neg",
        "neg_",
        "normal_float_tensor",
        "normal_tensor_float",
        "normal_tensor_tensor",
        "reciprocal",
        "reciprocal_",
        "relu",
        "relu_",
        "repeat_interleave_self_int",
        "repeat_interleave_self_tensor",
        "repeat_interleave_tensor",
        "rsqrt",
        "rsqrt_",
        "sigmoid",
        "sigmoid_",
        "sigmoid_backward",
        "silu",
        "silu_",
        "silu_backward",
        "sin",
        "sin_",
        "tanh",
        "tanh_",
        "tanh_backward",
        "to_dtype",
        "full_like",
        "resolve_neg",
        "linspace",
        "arange",
        "arange_start",
    ]
