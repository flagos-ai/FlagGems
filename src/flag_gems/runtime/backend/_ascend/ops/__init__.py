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

from ._dyn_quant_matmul_4bit import _dyn_quant_matmul_4bit
from .adaptive_avg_pool2d import adaptive_avg_pool2d
from .adaptive_max_pool3d import adaptive_max_pool3d
from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out
from .all import all, all_dim, all_dims
from .amax import amax
from .angle import angle
from .any import any, any_dim, any_dims
from .arange import arange, arange_start
from .arccos import arccos
from .argmax import argmax
from .argmin import argmin
from .argsort import argsort
from .asin_ import asin_
from .atan2_ import atan2_
from .attention import (
    ScaleDotProductAttention,
    _flash_attention_forward,
    flash_attention_forward,
    flash_attn_varlen_func,
    flash_attn_varlen_opt_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from .baddbmm import baddbmm
from .binary_cross_entropy_with_logits import binary_cross_entropy_with_logits
from .block_diag import block_diag
from .bmm import bmm
from .bmm_w8a8_int8 import bmm_w8a8_int8
from .bmm_w8a8_int8 import bmm_w8a8_int8 as bmm_w8a8
from .cat import cat, cat_out
from .cholesky_solve import cholesky_solve, cholesky_solve_out
from .clamp_max import clamp_max
from .clamp_max_ import clamp_max_
from .concat import concat
from .concatenate import concatenate
from .cosh_ import cosh_
from .count_nonzero import count_nonzero
from .cross import cross
from .cudnn_convolution import cudnn_convolution
from .cummax import cummax
from .cummin import cummin
from .cumsum import cumsum, normed_cumsum
from .deg2rad_out import deg2rad_out
from .diag import diag
from .diag_embed import diag_embed
from .diagonal import diagonal_backward
from .diff import diff
from .dist import dist
from .dot import dot
from .dropout_ import dropout_
from .embedding import embedding
from .embedding_bag import _embedding_bag, _embedding_bag_forward_only
from .embedding_bag_backward import _embedding_bag_backward
from .embedding_spmdm import embedding_spmdm
from .erfinv import erfinv
from .exponential import exponential
from .exponential_ import exponential_
from .fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
from .flash_api import mha_fwd, mha_varlan_fwd, mha_varlan_fwd_opt
from .flip import flip
from .frac import frac
from .full import full
from .full_like import full_like
from .fused_adam_ import fused_adam_
from .gather import gather, gather_backward
from .geometric import geometric, geometric_
from .greater_equal_ import greater_equal_
from .grouped_matmul import grouped_matmul
from .groupnorm import group_norm, group_norm_backward
from .gru import gru, gru_data
from .hadamard_transform import hadamard_transform
from .hinge_embedding_loss import hinge_embedding_loss
from .hstack import hstack
from .igammac import igammac, igammac_out
from .index import index
from .index_add import index_add, index_add_
from .index_copy_ import index_copy, index_copy_
from .index_fill import index_fill, index_fill_
from .index_reduce import index_reduce, index_reduce_, index_reduce_out
from .index_select import index_select
from .isin import isin
from .isposinf import isposinf
from .l1_loss import l1_loss
from .layernorm import layer_norm, native_layer_norm
from .linalg_cross import linalg_cross, linalg_cross_out
from .linalg_det import linalg_det, linalg_det_out
from .linalg_lstsq import linalg_lstsq
from .linalg_lu import linalg_lu, linalg_lu_out
from .linalg_lu_factor import linalg_lu_factor, linalg_lu_factor_out
from .linalg_lu_factor_ex import linalg_lu_factor_ex, linalg_lu_factor_ex_out
from .linalg_matrix_exp import linalg_matrix_exp, linalg_matrix_exp_out
from .linalg_matrix_norm import linalg_matrix_norm, linalg_matrix_norm_out
from .linalg_matrix_power import linalg_matrix_power, linalg_matrix_power_out
from .linalg_matrix_rank import (
    linalg_matrix_rank,
    linalg_matrix_rank_out,
    linalg_matrix_rank_tol,
    linalg_matrix_rank_tol_out,
)
from .linalg_norm import linalg_norm
from .linalg_qr import linalg_qr, linalg_qr_out
from .linalg_slogdet import linalg_slogdet
from .linalg_solve_triangular import (
    linalg_solve_triangular,
    linalg_solve_triangular_out,
)
from .linear import linear
from .linspace import linspace
from .log_ import log_
from .log_normal import log_normal
from .log_sigmoid_backward import log_sigmoid_backward, log_sigmoid_backward_out
from .log_softmax import log_softmax, log_softmax_backward, log_softmax_out
from .logical_not_ import logical_not_
from .masked_fill import masked_fill, masked_fill_
from .masked_scatter import masked_scatter, masked_scatter_
from .masked_scatter_backward import masked_scatter_backward
from .masked_select import masked_select
from .matmul_bf16 import matmul_bf16
from .matmul_int8 import matmul_int8
from .max import max, max_dim
from .mean import mean, mean_dim
from .min import min, min_dim
from .mixed_dtypes_linear import mixed_dtypes_linear
from .mm import mm, mm_out
from .mm_w8a8_int8 import mm_w8a8_int8, mm_w8a8_int8_out
from .mode import mode
from .mul import mul
from .multinomial import multinomial
from .nanmedian import nanmedian, nanmedian_dim, nanmedian_dim_values, nanmedian_out
from .nansum import nansum, nansum_out
from .new_ones import new_ones
from .nonzero_static import nonzero_static, nonzero_static_out
from .ones import ones
from .ones_like import ones_like
from .outer import outer
from .pad_sequence import pad_sequence
from .pairwise_distance import pairwise_distance
from .polar import polar
from .polygamma import polygamma_
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .randperm import randperm
from .relu6_ import relu6_
from .repeat_interleave import repeat_interleave_self_int
from .replication_pad2d_backward import (
    replication_pad2d_backward,
    replication_pad2d_backward_grad_input,
)
from .resolve_neg import resolve_neg
from .rms_norm import rms_norm
from .rms_norm_w8a16_int8 import rms_norm_w8a16_int8
from .rnn_tanh import rnn_tanh, rnn_tanh_data
from .rrelu_with_noise import rrelu_with_noise, rrelu_with_noise_
from .scatter import scatter, scatter_
from .scatter_add_ import scatter_add_
from .scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from .segment_reduce import _segment_reduce_backward, _segment_reduce_backward_out
from .select_backward import select_backward
from .select_scatter import select_scatter
from .silu import silu, silu_
from .slice_scatter import slice_scatter
from .softmax import softmax, softmax_backward, softmax_backward_out, softmax_out
from .sort import sort
from .sparse_sampled_addmm import sparse_sampled_addmm, sparse_sampled_addmm_out
from .special_erfinv import special_erfinv
from .stack import stack
from .subtract import subtract
from .swiglu import swiglu
from .thnn_fused_lstm_cell import thnn_fused_lstm_cell
from .threshold import threshold, threshold_backward
from .threshold_ import threshold_
from .topk import topk
from .topk_w8a16_fp8 import topk_w8a16_fp8
from .triangular_solve import triangular_solve, triangular_solve_out
from .triu import triu
from .uniform import uniform
from .unique import _unique2
from .unique_dim import unique_dim
from .unsafe_index import unsafe_index
from .unsafe_index_put import unsafe_index_put
from .unsafe_masked_index import unsafe_masked_index
from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa
from .upsample_linear1d_backward import upsample_linear1d_backward
from .upsample_nearest2d import upsample_nearest2d
from .upsample_nearest3d_backward_grad_input import (
    upsample_nearest3d_backward_grad_input,
)
from .var_mean import var_mean
from .vector_norm import vector_norm
from .view_copy import view_copy
from .vstack import vstack
from .where import where_scalar_other, where_scalar_self, where_self, where_self_out
from .xlogy_ import xlogy_
from .zero import zero
from .zeros import zeros
from .zeros_like import zeros_like

__all__ = [
    "_dyn_quant_matmul_4bit",
    "_embedding_bag",
    "_embedding_bag_backward",
    "_embedding_bag_forward_only",
    "_flash_attention_forward",
    "_segment_reduce_backward",
    "_segment_reduce_backward_out",
    "_unique2",
    "_upsample_bicubic2d_aa",
    "adaptive_avg_pool2d",
    "adaptive_max_pool3d",
    "addmm",
    "addmm_dtype",
    "addmm_dtype_out",
    "addmm_out",
    "all",
    "all_dim",
    "all_dims",
    "amax",
    "angle",
    "any",
    "any_dim",
    "any_dims",
    "arange",
    "arange_start",
    "arccos",
    "argmax",
    "argmin",
    "argsort",
    "asin_",
    "atan2_",
    "baddbmm",
    "binary_cross_entropy_with_logits",
    "block_diag",
    "bmm",
    "bmm_w8a8",
    "bmm_w8a8_int8",
    "cat",
    "cat_out",
    "cholesky_solve",
    "cholesky_solve_out",
    "clamp_max",
    "clamp_max_",
    "concat",
    "concatenate",
    "cosh_",
    "count_nonzero",
    "cross",
    "cudnn_convolution",
    "cummax",
    "cummin",
    "cumsum",
    "deg2rad_out",
    "diag",
    "diag_embed",
    "diagonal_backward",
    "diff",
    "dist",
    "dot",
    "dropout_",
    "embedding",
    "embedding_spmdm",
    "erfinv",
    "exponential",
    "exponential_",
    "fill_scalar",
    "fill_scalar_",
    "fill_tensor",
    "fill_tensor_",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_varlen_opt_func",
    "flip",
    "frac",
    "full",
    "full_like",
    "fused_adam_",
    "gather",
    "gather_backward",
    "geometric",
    "geometric_",
    "greater_equal_",
    "group_norm",
    "group_norm_backward",
    "grouped_matmul",
    "gru",
    "gru_data",
    "hadamard_transform",
    "hinge_embedding_loss",
    "hstack",
    "igammac",
    "igammac_out",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_reduce",
    "index_reduce_",
    "index_reduce_out",
    "index_select",
    "isin",
    "isposinf",
    "l1_loss",
    "layer_norm",
    "linalg_cross",
    "linalg_cross_out",
    "linalg_det",
    "linalg_det_out",
    "linalg_lstsq",
    "linalg_lu",
    "linalg_lu_factor",
    "linalg_lu_factor_ex",
    "linalg_lu_factor_ex_out",
    "linalg_lu_factor_out",
    "linalg_lu_out",
    "linalg_matrix_exp",
    "linalg_matrix_exp_out",
    "linalg_matrix_norm",
    "linalg_matrix_norm_out",
    "linalg_matrix_power",
    "linalg_matrix_power_out",
    "linalg_matrix_rank",
    "linalg_matrix_rank_out",
    "linalg_matrix_rank_tol",
    "linalg_matrix_rank_tol_out",
    "linalg_norm",
    "linalg_qr",
    "linalg_qr_out",
    "linalg_slogdet",
    "linalg_solve_triangular",
    "linalg_solve_triangular_out",
    "linear",
    "linspace",
    "log_",
    "log_normal",
    "log_sigmoid_backward",
    "log_sigmoid_backward_out",
    "log_softmax",
    "log_softmax_backward",
    "log_softmax_out",
    "logical_not_",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_scatter_backward",
    "masked_select",
    "matmul_bf16",
    "matmul_int8",
    "max",
    "max_dim",
    "mean",
    "mean_dim",
    "mha_fwd",
    "mha_varlan_fwd",
    "mha_varlan_fwd_opt",
    "min",
    "min_dim",
    "mixed_dtypes_linear",
    "mm",
    "mm_out",
    "mm_w8a8_int8",
    "mm_w8a8_int8_out",
    "mode",
    "mul",
    "multinomial",
    "nanmedian",
    "nanmedian_dim",
    "nanmedian_dim_values",
    "nanmedian_out",
    "nansum",
    "nansum_out",
    "native_layer_norm",
    "new_ones",
    "nonzero_static",
    "nonzero_static_out",
    "normed_cumsum",
    "ones",
    "ones_like",
    "outer",
    "pad_sequence",
    "pairwise_distance",
    "polar",
    "polygamma_",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "randperm",
    "relu6_",
    "repeat_interleave_self_int",
    "replication_pad2d_backward",
    "replication_pad2d_backward_grad_input",
    "resolve_neg",
    "rms_norm",
    "rms_norm_w8a16_int8",
    "rnn_tanh",
    "rnn_tanh_data",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "ScaleDotProductAttention",
    "scatter",
    "scatter_",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "scatter_reduce_out",
    "select_backward",
    "select_scatter",
    "silu",
    "silu_",
    "slice_scatter",
    "softmax",
    "softmax_backward",
    "softmax_backward_out",
    "softmax_out",
    "sort",
    "sparse_sampled_addmm",
    "sparse_sampled_addmm_out",
    "special_erfinv",
    "stack",
    "subtract",
    "swiglu",
    "thnn_fused_lstm_cell",
    "threshold",
    "threshold_",
    "threshold_backward",
    "topk",
    "topk_w8a16_fp8",
    "triangular_solve",
    "triangular_solve_out",
    "triu",
    "uniform",
    "unique_dim",
    "unsafe_index",
    "unsafe_index_put",
    "unsafe_masked_index",
    "upsample_linear1d_backward",
    "upsample_nearest2d",
    "upsample_nearest3d_backward_grad_input",
    "var_mean",
    "vector_norm",
    "view_copy",
    "vstack",
    "where_scalar_other",
    "where_scalar_self",
    "where_self",
    "where_self_out",
    "xlogy_",
    "zero",
    "zeros",
    "zeros_like",
]
