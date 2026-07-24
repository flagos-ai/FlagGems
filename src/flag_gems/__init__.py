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

# ruff: noqa: F405
import warnings

import torch
from packaging import version

from flag_gems import testing  # noqa: F401
from flag_gems import runtime
from flag_gems.config import aten_patch_list, resolve_user_setting
from flag_gems.experimental_ops import *  # noqa: F403
from flag_gems.fused import *  # noqa: F403
from flag_gems.logging_utils import setup_flaggems_logging, teardown_flaggems_logging
from flag_gems.modules import *  # noqa: F403
from flag_gems.ops import *  # noqa: F403
from flag_gems.ops import range as range_op
from flag_gems.patches import *  # noqa: F403
from flag_gems.patches import patch_empty_vllm  # noqa: F401
from flag_gems.runtime import flagtune
from flag_gems.runtime.backend import SpecOpRegistrar
from flag_gems.runtime.op_registrar import GeneralOpRegistrar

try:
    from flag_gems._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version as _meta_version

        __version__ = _meta_version("flag-gems")
    except Exception:
        __version__ = "unknown"
device = runtime.device.name
vendor_name = runtime.device.vendor_name
backend_info = runtime.device
aten_lib = torch.library.Library("aten", "IMPL")

# Register all ops in the current backend with SpecOpRegistrar to support architecture-specialized implementations
SpecOpRegistrar(registry=globals(), vendor=vendor_name).apply()

registrar = GeneralOpRegistrar
current_work_registrar = None
AUTOGRAD_DISPATCH_KEY = torch._C.DispatchKey.Autograd.name


def torch_ge(v):
    return version.parse(torch.__version__) >= version.parse(v)


_FULL_CONFIG = (
    ("__and__.Scalar", bitwise_and_scalar),
    ("__and__.Tensor", bitwise_and_tensor),
    ("__ilshift__.Tensor", __ilshift__),
    ("__ior__.Scalar", bitwise_or_scalar_),
    ("__ior__.Tensor", bitwise_or_tensor_),
    ("__irshift__.Tensor", __irshift__),
    ("__lshift__", __lshift__),
    ("__ixor__.Scalar", xor_scalar_),
    ("__ixor__.Tensor", xor_),
    ("__or__.Scalar", bitwise_or_scalar),
    ("__or__.Tensor", bitwise_or_tensor),
    ("__xor__.Scalar", xor_scalar),
    ("__xor__.Tensor", xor),
    ("_adaptive_avg_pool2d", adaptive_avg_pool2d),
    (
        "_amp_foreach_non_finite_check_and_unscale_",
        _amp_foreach_non_finite_check_and_unscale_,
    ),
    ("_assert_async", _assert_async),
    ("_batch_norm_no_update", _batch_norm_no_update),
    ("_cdist_backward", _cdist_backward),
    ("_conj", _conj),
    ("_conv_depthwise2d", _conv_depthwise2d),
    ("_efficient_attention_backward", efficient_attention_backward),
    ("_embedding_bag_dense_backward", _embedding_bag_dense_backward),
    ("_euclidean_dist", _euclidean_dist),
    ("_flash_attention_backward", flash_attention_backward),
    ("_flash_attention_forward", flash_attention_forward),
    (
        "_functional_sym_constrain_range",
        _functional_sym_constrain_range,
    ),
    (
        "_functional_sym_constrain_range_for_size",
        _functional_sym_constrain_range_for_size,
    ),
    ("_fused_adam", _fused_adam),
    ("_fused_adam_", _fused_adam_),
    ("_grouped_mm", group_mm),
    ("_index_put_impl_", _index_put_impl_),
    ("_is_all_true", _is_all_true),
    ("_jagged_to_padded_dense_forward", _jagged_to_padded_dense_forward),
    ("_linalg_eigvals", _linalg_eigvals),
    ("linalg_svdvals", linalg_svdvals),
    ("_log_softmax", log_softmax),
    ("_log_softmax.out", log_softmax_out),
    ("_log_softmax_backward_data", log_softmax_backward),
    ("_log_softmax_backward_data.out", log_softmax_backward_out),
    ("_masked_scale", _masked_scale),
    ("_nested_view_from_buffer_copy", _nested_view_from_buffer_copy),
    ("_pdist_backward", _pdist_backward),
    ("_prelu_kernel", _prelu_kernel),
    ("_prelu_kernel_backward", _prelu_kernel_backward),
    ("_resize_output", _resize_output),
    ("_safe_softmax", _safe_softmax),
    (
        "_scaled_dot_product_cudnn_attention_backward",
        scaled_dot_product_cudnn_attention_backward,
    ),
    (
        "_scaled_dot_product_efficient_attention_backward",
        scaled_dot_product_efficient_attention_backward,
    ),
    (
        "_scaled_dot_product_flash_attention_backward",
        scaled_dot_product_flash_attention_backward,
    ),
    (
        "_scaled_dot_product_fused_attention_overrideable",
        _scaled_dot_product_fused_attention_overrideable,
    ),
    ("_scaled_grouped_mm", scaled_grouped_mm, lambda: torch_ge("2.8")),
    ("_scaled_mm", scaled_mm, lambda: torch_ge("2.5")),
    ("_scaled_mm.out", scaled_mm_out, lambda: torch_ge("2.5")),
    ("_segment_reduce_backward", _segment_reduce_backward),
    ("_segment_reduce_backward.out", _segment_reduce_backward_out),
    ("_softmax", softmax),
    ("_softmax.out", softmax_out),
    ("_softmax_backward_data", softmax_backward),
    ("_softmax_backward_data.out", softmax_backward_out),
    ("_sparse_semi_structured_mm", _sparse_semi_structured_mm),
    ("_thnn_fused_lstm_cell", _thnn_fused_lstm_cell),
    ("_thnn_fused_lstm_cell_backward_impl", _thnn_fused_lstm_cell_backward_impl),
    (
        "_to_copy",
        to_copy,
        lambda: version.parse(torch.__version__) >= version.parse("2.4"),
    ),
    ("_unique2", _unique2),
    ("_unsafe_masked_index", _unsafe_masked_index),
    ("_unsafe_masked_index_put_accumulate", _unsafe_masked_index_put_accumulate),
    ("_unsafe_view", _unsafe_view),
    ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa),
    ("_upsample_bicubic2d_aa_backward", _upsample_bicubic2d_aa_backward),
    ("_upsample_bilinear2d_aa", _upsample_bilinear2d_aa),
    ("_upsample_nearest_exact1d", _upsample_nearest_exact1d),
    ("_upsample_nearest_exact2d_backward", _upsample_nearest_exact2d_backward),
    ("_upsample_nearest_exact3d", _upsample_nearest_exact3d),
    ("_weight_norm_interface", weight_norm_interface),
    ("_weight_norm_interface_backward", weight_norm_interface_backward),
    ("abs", abs),
    ("abs_", abs_),
    ("absolute", absolute),
    ("acos", acos),
    ("acosh", acosh),
    ("acosh_", acosh_),
    ("adaptive_max_pool3d_backward", adaptive_max_pool3d_backward),
    ("add.Tensor", add),
    ("add_.Tensor", add_),
    ("add_rms_norm", add_rms_norm),
    ("addcdiv", addcdiv),
    ("addcdiv.out", addcdiv_out),
    ("addcdiv_", addcdiv_),
    ("addcmul", addcmul),
    ("addcmul.out", addcmul_out),
    ("addcmul_", addcmul_),
    ("addmm", addmm),
    ("addmm.dtype", addmm_dtype),
    ("addmm.dtype_out", addmm_dtype_out),
    ("addmm.out", addmm_out),
    ("addmm_", addmm_),
    ("addmv", addmv),
    ("addmv.out", addmv_out),
    ("addr", addr),
    ("affine_grid_generator", affine_grid_generator),
    ("alias_copy", alias_copy),
    ("all", all),
    ("all.dim", all_dim),
    ("all.dims", all_dims),
    ("allclose", allclose),
    ("alpha_dropout", alpha_dropout),
    ("amax", amax),
    ("amin", amin),
    ("amin_", amin_),
    ("aminmax", aminmax),
    ("angle", angle),
    ("any", any),
    ("any.dim", any_dim),
    ("any.dims", any_dims),
    ("arange", arange),
    ("arange.start", arange_start),
    ("arange.start_step", arange_start),
    ("arccos", arccos),
    ("arccos_", arccos_),
    ("arcsin", arcsin),
    ("arcsin.out", arcsin),
    ("arcsin_", arcsin_),
    ("arcsinh", arcsinh),
    ("arcsinh.out", arcsinh_out),
    ("arcsinh_", arcsinh_),
    ("arctan", arctan),
    ("arctan_", arctan_),
    ("arctanh_", arctanh_),
    ("argmax", argmax),
    ("argmin", argmin),
    ("argsort", argsort),
    ("as_strided_copy", as_strided_copy),
    ("as_strided_copy.out", as_strided_copy_out),
    ("asin", asin),
    ("asin_", asin_),
    ("asinh", asinh),
    ("asinh.out", asinh_out),
    ("asinh_", asinh_),
    ("atan", atan),
    ("atan2", atan2),
    ("atan2.out", atan2_out),
    ("atan_", atan_),
    ("atanh", atanh),
    ("avg_pool2d", avg_pool2d),
    ("avg_pool2d_backward", avg_pool2d_backward),
    ("avg_pool3d", avg_pool3d),
    ("avg_pool3d_backward", avg_pool3d_backward),
    ("baddbmm", baddbmm),
    ("baddbmm.out", baddbmm_out),
    ("beam_search_score", beam_search_score),
    ("beam_search_score_", beam_search_score_),
    ("bernoulli", bernoulli),
    ("bernoulli_.float", bernoulli_),
    ("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits),
    ("bincount", bincount),
    ("bitwise_and.Scalar", bitwise_and_scalar),
    ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor),
    ("bitwise_and.Tensor", bitwise_and_tensor),
    ("bitwise_and_.Scalar", bitwise_and_scalar_),
    ("bitwise_and_.Tensor", bitwise_and_tensor_),
    ("bitwise_left_shift", bitwise_left_shift),
    ("bitwise_left_shift_", bitwise_left_shift_),
    ("bitwise_not", bitwise_not),
    ("bitwise_not_", bitwise_not_),
    ("bitwise_or.Scalar", bitwise_or_scalar),
    ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor),
    ("bitwise_or.Tensor", bitwise_or_tensor),
    ("bitwise_or_.Scalar", bitwise_or_scalar_),
    ("bitwise_or_.Tensor", bitwise_or_tensor_),
    ("bitwise_right_shift", bitwise_right_shift),
    ("bitwise_right_shift_", bitwise_right_shift_),
    ("bitwise_xor.Scalar", bitwise_xor_scalar),
    ("bitwise_xor.Scalar_Tensor", bitwise_xor_scalar_tensor),
    ("bitwise_xor.Tensor", bitwise_xor_tensor),
    ("bitwise_xor_.Scalar", bitwise_xor_scalar_),
    ("bitwise_xor_.Tensor", bitwise_xor_tensor_),
    ("bmm", bmm),
    ("bmm.out", bmm_out),
    ("broadcast_tensors", broadcast_tensors),
    ("broadcast_to", broadcast_to),
    ("bucketize.Tensor", bucketize),
    ("cat", cat),
    ("cat.out", cat_out),
    ("cauchy", cauchy),
    ("cauchy_", cauchy_),
    ("ceil", ceil),
    ("ceil.out", ceil_out),
    ("ceil_", ceil_),
    ("celu", celu),
    ("celu_", celu_),
    ("channel_shuffle", channel_shuffle),
    ("clamp", clamp),
    ("clamp.Tensor", clamp_tensor),
    ("clamp_", clamp_),
    ("clamp_.Tensor", clamp_tensor_),
    ("clamp_max", clamp_max),
    ("clamp_max_", clamp_max_),
    ("clamp_min", clamp_min),
    ("clamp_min_", clamp_min_),
    ("clip", clip),
    ("clip_", clip_),
    ("col2im", col2im),
    ("concat", concat),
    ("concatenate", concatenate),
    ("conj_physical", conj_physical),
    ("constant_pad_nd", constant_pad_nd),
    # ("contiguous", contiguous),
    ("conv1d", conv1d),
    ("conv1d.padding", conv1d),
    ("conv2d", conv2d),
    ("conv2d.padding", conv2d),
    ("conv3d", conv3d),
    ("conv3d.padding", conv3d),
    ("conv_transpose1d", conv_transpose1d),
    ("conv_transpose2d", conv_transpose2d),
    (
        "copy_",
        copy_,
        lambda: version.parse(torch.__version__) >= version.parse("2.4"),
    ),
    ("copysign", copysign),
    ("copysign.out", copysign_out),
    ("copysign_.Tensor", copysign_),
    ("cos", cos),
    ("cos_", cos_),
    ("cosh", cosh),
    ("cosh.out", cosh_out),
    ("cosh_", cosh_),
    ("count_nonzero", count_nonzero),
    ("ctc_loss.IntList", ctc_loss, None, (AUTOGRAD_DISPATCH_KEY,)),
    ("ctc_loss.Tensor", ctc_loss, None, (AUTOGRAD_DISPATCH_KEY,)),
    ("cudnn_batch_norm_backward", cudnn_batch_norm_backward),
    ("cudnn_convolution", cudnn_convolution),
    ("cummax", cummax),
    ("cummin", cummin),
    ("cumprod", cumprod),
    ("cumprod_", cumprod_),
    ("cumsum", cumsum),
    ("cumsum.out", cumsum_out),
    ("deg2rad", deg2rad),
    ("deg2rad.out", deg2rad_out),
    ("deg2rad_", deg2rad_),
    ("dequantize", dequantize),
    ("dequantize.self", dequantize),
    ("diag", diag),
    ("diag_embed", diag_embed),
    ("diagonal_backward", diagonal_backward),
    ("diagonal_copy", diagonal_copy),
    ("diagonal_scatter", diagonal_scatter),
    ("diff", diff),
    ("digamma", digamma),
    ("digamma_", digamma_),
    ("div.Scalar", true_divide),
    ("div.Scalar_mode", div_mode),
    ("div.Tensor", true_divide),
    ("div.Tensor_mode", div_mode),
    ("div.out", true_divide_out),
    ("div_.Scalar", true_divide_),
    ("div_.Scalar_mode", div_mode_),
    ("div_.Tensor", true_divide_),
    ("div_.Tensor_mode", div_mode_),
    ("divide.Scalar", true_divide),
    ("divide.Scalar_mode", div_mode),
    ("divide.Tensor", true_divide),
    ("divide.Tensor_mode", div_mode),
    ("divide_.Scalar", true_divide_),
    ("divide_.Scalar_mode", div_mode_),
    ("divide_.Tensor", true_divide_),
    ("divide_.Tensor_mode", div_mode_),
    ("dot", dot),
    ("elu", elu),
    ("elu_", elu_),
    ("elu_backward", elu_backward),
    ("embedding", embedding),
    ("embedding_backward", embedding_backward),
    ("embedding_dense_backward", embedding_dense_backward),
    ("empty", empty),
    ("eq.Scalar", eq_scalar),
    ("eq.Tensor", eq),
    ("equal", equal),
    ("erf", erf),
    ("erf_", erf_),
    ("erfinv", erfinv),
    ("erfinv_", erfinv_),
    ("exp", exp),
    ("exp.out", exp_out),
    ("exp2", exp2),
    ("exp2_", exp2_),
    ("exp_", exp_),
    ("expand", expand),
    ("expand_", expand_),
    ("expm1", expm1),
    ("expm1.out", expm1_out),
    ("expm1_", expm1_),
    ("exponential_", exponential_),
    ("eye", eye),
    ("eye.m", eye_m),
    ("feature_dropout", feature_dropout),
    ("feature_dropout_", feature_dropout_),
    ("fill.Scalar", fill_scalar),
    ("fill.Scalar_out", fill_scalar_out),
    ("fill.Tensor", fill_tensor),
    ("fill.Tensor_out", fill_tensor_out),
    ("fill_.Scalar", fill_scalar_),
    ("fill_.Tensor", fill_tensor_),
    ("fix", fix),
    ("flip", flip),
    ("floor", floor),
    ("floor.out", floor_out),
    ("floor_", floor_),
    ("floor_divide", floor_divide),
    ("floor_divide.Scalar", floor_divide),
    ("floor_divide_.Scalar", floor_divide_),
    ("floor_divide_.Tensor", floor_divide_),
    ("fmax", fmax),
    ("fmax.out", fmax_out),
    ("fmin", fmin),
    ("fmin.out", fmin_out),
    ("fmod.Scalar", fmod_scalar),
    ("fmod.Tensor", fmod_tensor),
    ("fmod_", fmod_),
    ("fmod_.Scalar", fmod_scalar_),
    ("fmod_.Tensor", fmod_tensor_),
    ("frac", frac),
    ("frac_", frac_),
    ("fractional_max_pool2d", fractional_max_pool2d),
    ("fractional_max_pool2d_backward", fractional_max_pool2d_backward),
    ("full", full),
    ("full_like", full_like),
    ("gather", gather),
    ("gather_backward", gather_backward),
    ("gather_block_quantized", gather_block_quantized),
    ("gcd", gcd),
    ("gcd.out", gcd_out),
    ("gcd_", gcd_),
    ("ge.Scalar", ge_scalar),
    ("ge.Tensor", ge),
    ("gelu", gelu),
    ("gelu_", gelu_),
    ("gelu_backward", gelu_backward),
    ("geometric.float", geometric),
    ("geometric_.float", geometric_),
    ("glu", glu),
    ("glu_backward", glu_backward),
    ("greater.Scalar", greater_scalar),
    ("greater.Scalar_out", greater_scalar_out),
    ("greater.Tensor", greater),
    ("greater.out", greater_out),
    ("greater_equal.Scalar", ge_scalar),
    ("greater_equal.Tensor", ge),
    ("greater_equal_.Tensor", greater_equal_),
    ("grid_sample", grid_sample),
    ("gt.Scalar", gt_scalar),
    ("gt.Tensor", gt),
    ("gt_.Scalar", gt_scalar_),
    ("gt_.Tensor", gt_tensor_),
    ("hardsigmoid", hardsigmoid),
    ("hardsigmoid.out", hardsigmoid_out),
    ("hardswish_", hardswish_),
    ("histc", histc),
    ("hstack", hstack),
    ("hypot", hypot),
    ("i0", i0),
    ("i0.out", i0_out),
    ("i0_", i0_),
    ("igammac_", igammac_),
    ("im2col", im2col),
    ("index.Tensor", index),
    ("index_add", index_add),
    ("index_add_", index_add_),
    ("index_copy", index_copy),
    ("index_copy_", index_copy_),
    ("index_put", index_put),
    ("index_put_", index_put_),
    ("index_reduce_", index_reduce_),
    ("index_select", index_select),
    ("index_select_backward", index_select_backward),
    ("is_nonzero", is_nonzero),
    ("isclose", isclose),
    ("isfinite", isfinite),
    ("isin.Scalar_Tensor", isin),
    ("isin.Tensor_Scalar", isin),
    ("isin.Tensor_Tensor", isin),
    ("isinf", isinf),
    ("isnan", isnan),
    ("isneginf", isneginf),
    ("isneginf.out", isneginf_out),
    ("kron", kron),
    ("kthvalue", kthvalue),
    ("lcm", lcm),
    ("lcm_", lcm_),
    ("le.Scalar", le_scalar),
    ("le.Tensor", le),
    ("leaky_relu", leaky_relu),
    ("leaky_relu.out", leaky_relu_out),
    ("leaky_relu_", leaky_relu_),
    ("lerp.Scalar", lerp_scalar),
    ("lerp.Tensor", lerp_tensor),
    ("lerp_.Scalar", lerp_scalar_),
    ("lerp_.Tensor", lerp_tensor_),
    ("less.Scalar", lt_scalar),
    ("less.Tensor", lt),
    ("less_equal.Scalar", less_equal_scalar),
    ("less_equal.Tensor", less_equal),
    ("lgamma", lgamma),
    ("lgamma_", lgamma_),
    ("lift", lift),
    ("lift.out", lift_out),
    ("lift_fresh_copy", lift_fresh_copy),
    ("linalg_cholesky", linalg_cholesky),
    ("linalg_ldl_factor", ldl_factor),
    ("linalg_ldl_factor_ex", ldl_factor_ex),
    ("linalg_slogdet", linalg_slogdet),
    ("linalg_vector_norm", vector_norm),
    ("linear", linear),
    ("linear_backward", linear_backward),
    ("linspace", linspace),
    ("log", log),
    ("log10", log10),
    ("log10.out", log10_out),
    ("log10_", log10_),
    ("log1p", log1p),
    ("log1p.out", log1p_out),
    ("log1p_", log1p_),
    ("log2", log2),
    ("log2_", log2_),
    ("log_normal_", log_normal_),
    ("log_sigmoid", log_sigmoid),
    ("logaddexp", logaddexp),
    ("logaddexp.out", logaddexp_out),
    ("logaddexp2", logaddexp2),
    ("logaddexp2.out", logaddexp2_out),
    ("logcumsumexp", logcumsumexp),
    ("logcumsumexp.out", logcumsumexp_out),
    ("logical_and", logical_and),
    ("logical_and_", logical_and_),
    ("logical_not", logical_not),
    ("logical_not_", logical_not_),
    ("logical_or", logical_or),
    ("logical_or_", logical_or_),
    ("logical_xor", logical_xor),
    ("logical_xor_", logical_xor_),
    ("logit", logit),
    ("logit.out", logit_out),
    ("logit_", logit_),
    ("logspace", logspace),
    ("logsumexp", logsumexp),
    ("lt.Scalar", lt_scalar),
    ("lt.Tensor", lt),
    ("lt_.Scalar", lt_scalar_),
    ("lt_.Tensor", lt_),
    ("margin_ranking_loss", margin_ranking_loss),
    ("masked_fill.Scalar", masked_fill),
    ("masked_fill.Tensor", masked_fill),
    ("masked_fill_.Scalar", masked_fill_),
    ("masked_fill_.Tensor", masked_fill_),
    ("masked_scatter", masked_scatter),
    ("masked_scatter_", masked_scatter_),
    ("masked_scatter_backward", masked_scatter_backward),
    ("masked_select", masked_select),
    ("matmuladd", matmuladd),
    ("max", max),
    ("max.dim", max_dim),
    ("max_pool2d_backward", max_pool2d_backward),
    ("max_pool2d_with_indices", max_pool2d_with_indices),
    ("max_pool3d_backward", max_pool3d_backward),
    ("max_pool3d_with_indices", max_pool3d_with_indices),
    ("max_unpool2d", max_unpool2d),
    ("maximum", maximum),
    ("mean", mean),
    ("mean.dim", mean_dim),
    ("median", median),
    ("median.dim", median_dim),
    ("median.dim_values", median_dim_values),
    ("median.out", median_out),
    ("min", min),
    ("min.dim", min_dim),
    ("minimum", minimum),
    ("mish", mish),
    ("mish_", mish_),
    ("mm", mm),
    ("mm.out", mm_out),
    ("mode", mode),
    ("mse_loss", mse_loss),
    ("mse_loss_backward", mse_loss_backward),
    ("mul.Tensor", mul),
    ("mul_.Tensor", mul_),
    ("multinomial", multinomial),
    ("multiply_", multiply_),
    ("multiply_.Scalar", multiply_),
    ("multiply_.Tensor", multiply_),
    ("mv", mv),
    ("mvlgamma_", mvlgamma_),
    ("nan_to_num", nan_to_num),
    ("nanmedian", nanmedian),
    ("nanmedian.dim", nanmedian_dim),
    ("nanmedian.dim_values", nanmedian_dim_values),
    ("nanmedian.out", nanmedian_out),
    ("narrow", narrow),
    ("narrow_copy", narrow_copy),
    ("native_batch_norm", batch_norm),
    ("native_batch_norm_backward", batch_norm_backward),
    ("native_dropout", dropout),
    ("native_dropout_backward", dropout_backward),
    ("native_group_norm", group_norm),
    ("native_group_norm_backward", group_norm_backward),
    ("native_layer_norm", layer_norm),
    ("native_layer_norm_backward", layer_norm_backward),
    ("ne.Scalar", ne_scalar),
    ("ne.Tensor", ne),
    ("neg", neg),
    ("neg_", neg_),
    ("negative", negative),
    ("new_full", new_full),
    ("new_ones", new_ones),
    ("nextafter", nextafter),
    ("nextafter_", nextafter_),
    ("nll_loss2d_backward", nll_loss2d_backward),
    ("nll_loss2d_forward", nll_loss2d_forward),
    ("nll_loss_backward", nll_loss_backward),
    ("nll_loss_forward", nll_loss_forward),
    ("nll_loss_nd_backward", nll_loss_nd_backward),
    ("nll_loss_nd_forward", nll_loss_nd_forward),
    ("nonzero", nonzero),
    ("nonzero_numpy", nonzero_numpy),
    ("norm", norm),
    ("norm.Scalar", norm_scalar),
    ("norm.ScalarOpt_dim", norm_scalaropt_dim),
    ("normal.Tensor_Tensor", normal_tensor_tensor),
    ("normal.Tensor_float", normal_tensor_float),
    ("normal.float_Tensor", normal_float_tensor),
    ("normal_", normal_),
    ("normed_cumsum", normed_cumsum),
    ("not_equal.Scalar", not_equal_scalar),
    ("not_equal.Tensor", not_equal),
    ("one_hot", one_hot),
    ("ones", ones),
    ("ones_like", ones_like),
    ("pad", pad),
    ("pdist", pdist),
    ("permute_copy", permute_copy),
    ("pixel_shuffle", pixel_shuffle),
    ("pixel_unshuffle", pixel_unshuffle),
    ("pixel_unshuffle.out", pixel_unshuffle_out),
    ("poisson", poisson),
    ("polar", polar),
    ("pow.Scalar", pow_scalar),
    ("pow.Tensor_Scalar", pow_tensor_scalar),
    ("pow.Tensor_Tensor", pow_tensor_tensor),
    ("pow_.Scalar", pow_tensor_scalar_),
    ("pow_.Tensor", pow_tensor_tensor_),
    ("prelu", prelu),
    ("prod", prod),
    ("prod.dim_int", prod_dim),
    ("quantile", quantile),
    ("rad2deg", rad2deg),
    ("rad2deg_", rad2deg_),
    ("rand", rand),
    ("rand_like", rand_like),
    ("randint", randint),
    ("randint_like", randint_like),
    ("randn", randn),
    ("randn_like", randn_like),
    ("randperm", randperm),
    ("range", range_op),
    ("reciprocal", reciprocal),
    ("reciprocal_", reciprocal_),
    ("reflection_pad1d", reflection_pad1d),
    ("reflection_pad1d.out", reflection_pad1d_out),
    ("reflection_pad1d_backward", reflection_pad1d_backward),
    ("reflection_pad2d", reflection_pad2d),
    ("reflection_pad2d.out", reflection_pad2d_out),
    ("reflection_pad3d", reflection_pad3d),
    ("reflection_pad3d.out", reflection_pad3d),
    ("reflection_pad3d_backward", reflection_pad3d_backward),
    ("relu", relu),
    ("relu6", relu6),
    ("relu_", relu_),
    ("remainder", remainder),
    ("remainder.Scalar", remainder),
    ("remainder.Scalar_Tensor", remainder),
    ("remainder.Tensor", remainder),
    ("remainder_.Scalar", remainder_),
    ("remainder_.Tensor", remainder_),
    ("renorm", renorm),
    ("renorm_", renorm_),
    ("repeat", repeat),
    ("repeat_interleave.Tensor", repeat_interleave_tensor),
    ("repeat_interleave.self_Tensor", repeat_interleave_self_tensor),
    ("repeat_interleave.self_int", repeat_interleave_self_int),
    ("replication_pad1d", replication_pad1d),
    ("replication_pad1d.out", replication_pad1d_out),
    ("replication_pad3d", replication_pad3d),
    ("resize", resize),
    ("resize_", resize_),
    ("resize_as", resize_as),
    ("resize_as_", resize_as_),
    ("resolve_conj", resolve_conj),
    ("resolve_neg", resolve_neg),
    ("rms_norm", rms_norm),
    ("rnn_relu", rnn_relu),
    ("roll", roll),
    ("rot90", rot90),
    ("round", round),
    ("round.out", round_out),
    ("round_", round_),
    ("rrelu_with_noise_backward", rrelu_with_noise_backward),
    ("rrelu_with_noise_functional", rrelu_with_noise_functional),
    ("rsqrt", rsqrt),
    ("rsqrt_", rsqrt_),
    ("rsub.Scalar", rsub_scalar),
    ("rsub.Tensor", rsub_tensor),
    ("scalar_tensor", scalar_tensor),
    ("scaled_softmax_backward", scaled_softmax_backward),
    ("scaled_softmax_forward", scaled_softmax_forward),
    ("scatter.reduce", scatter),
    ("scatter.src", scatter),
    ("scatter_.reduce", scatter_),
    ("scatter_.src", scatter_),
    ("scatter_add_", scatter_add_),
    ("scatter_reduce.two", scatter_reduce),
    ("scatter_reduce.two_out", scatter_reduce_out),
    ("scatter_reduce_.two", scatter_reduce_),
    ("searchsorted.Scalar", searchsorted_scalar),
    ("searchsorted.Scalar_out", searchsorted_scalar_out),
    ("searchsorted.Tensor", searchsorted),
    ("searchsorted.Tensor_out", searchsorted_out),
    ("segment_reduce", segment_reduce),
    ("segment_reduce.out", segment_reduce_out),
    ("select_backward", select_backward),
    ("select_scatter", select_scatter),
    ("selu", selu),
    ("selu_", selu_),
    ("sgn_", sgn_),
    ("sigmoid", sigmoid),
    ("sigmoid_", sigmoid_),
    ("sigmoid_backward", sigmoid_backward),
    ("signbit", signbit),
    ("signbit.out", signbit_out),
    ("silu", silu),
    ("silu_", silu_),
    ("silu_backward", silu_backward),
    ("sin", sin),
    ("sin_", sin_),
    ("sinc", sinc),
    ("sinc_", sinc_),
    ("sinh", sinh),
    ("sinh_", sinh_),
    ("slice_backward", slice_backward),
    ("slice_scatter", slice_scatter),
    ("smooth_l1_loss", smooth_l1_loss),
    ("smooth_l1_loss.out", smooth_l1_loss_out),
    ("smooth_l1_loss_backward", smooth_l1_loss_backward),
    ("soft_margin_loss", soft_margin_loss),
    ("soft_margin_loss_backward", soft_margin_loss_backward),
    ("softplus", softplus),
    ("softplus_backward", softplus_backward),
    ("softshrink", softshrink),
    ("softshrink.out", softshrink_out),
    ("sort", sort),
    ("sort.stable", sort_stable),
    ("special_airy_ai", special_airy_ai),
    ("special_airy_ai.out", special_airy_ai_out),
    ("special_chebyshev_polynomial_u", special_chebyshev_polynomial_u),
    ("special_chebyshev_polynomial_u.n_scalar", special_chebyshev_polynomial_u),
    ("special_chebyshev_polynomial_v", special_chebyshev_polynomial_v),
    ("special_chebyshev_polynomial_w", special_chebyshev_polynomial_w),
    ("special_chebyshev_polynomial_w_out", special_chebyshev_polynomial_w_out),
    ("special_digamma", special_digamma),
    ("special_erfinv", special_erfinv),
    ("special_erfinv.out", special_erfinv_out),
    ("special_erfinv_", special_erfinv_),
    ("special_gammainc", special_gammainc),
    ("special_gammaln", special_gammaln),
    ("special_gammaln.out", special_gammaln_out),
    ("special_hermite_polynomial_h", special_hermite_polynomial_h),
    ("special_i0e", special_i0e),
    ("special_i0e_out", special_i0e_out),
    ("special_i1", special_i1),
    ("special_i1_out", special_i1_out),
    ("special_log1p", special_log1p),
    ("special_log1p.out", special_log1p_out),
    ("special_logsumexp", special_logsumexp),
    ("special_log_softmax", special_log_softmax),
    ("special_modified_bessel_k0", special_modified_bessel_k0),
    ("special_modified_bessel_k0.out", special_modified_bessel_k0_out),
    ("special_modified_bessel_k1", special_modified_bessel_k1),
    ("special_modified_bessel_k1.out", special_modified_bessel_k1_out),
    ("special_scaled_modified_bessel_k1", special_scaled_modified_bessel_k1),
    ("special_scaled_modified_bessel_k1.out", special_scaled_modified_bessel_k1_out),
    ("special_shifted_chebyshev_polynomial_u", special_shifted_chebyshev_polynomial_u),
    (
        "special_shifted_chebyshev_polynomial_u_",
        special_shifted_chebyshev_polynomial_u_,
    ),
    ("special_shifted_chebyshev_polynomial_v", special_shifted_chebyshev_polynomial_v),
    ("special_shifted_chebyshev_polynomial_w", special_shifted_chebyshev_polynomial_w),
    ("special_sinc", special_sinc),
    ("special_xlog1py", special_xlog1py),
    ("split_with_sizes_copy", split_with_sizes_copy),
    ("sqrt", sqrt),
    ("sqrt_", sqrt_),
    ("square", square),
    ("square.out", square_out),
    ("square_", square_),
    ("squeeze_copy", squeeze_copy),
    ("stack", stack),
    ("std.correction", std),
    ("sub.Tensor", sub),
    ("sub_.Tensor", sub_),
    ("subtract_.Tensor", subtract_),
    ("sum", sum),
    ("sum.IntList_out", sum_dim_out),
    ("sum.dim_IntList", sum_dim),
    ("sum.out", sum_out),
    ("svd", svd),
    ("sym_stride", sym_stride),
    ("t_copy", t_copy),
    ("t_copy.out", t_copy_out),
    ("tan", tan),
    ("tan_", tan_),
    ("tanh", tanh),
    ("tanh_", tanh_),
    ("tanh_backward", tanh_backward),
    ("tensor_split", tensor_split),
    ("threshold", threshold),
    ("threshold_", threshold_),
    ("threshold_backward", threshold_backward),
    ("tile", tile),
    ("topk", topk),
    ("trace", trace),
    ("transpose.int", transpose),
    ("tril", tril),
    ("tril.out", tril_out),
    ("tril_", tril_),
    ("triu", triu),
    ("triu_", triu_),
    ("true_divide.Scalar", true_divide),
    ("true_divide.Tensor", true_divide),
    ("true_divide.out", true_divide_out),
    ("true_divide_.Scalar", true_divide_),
    ("true_divide_.Tensor", true_divide_),
    ("trunc", trunc),
    ("trunc_", trunc_),
    ("unbind.int", unbind),
    ("unbind_copy", unbind_copy),
    ("unfold_backward", unfold_backward),
    ("unfold_copy", unfold_copy),
    ("uniform_", uniform_),
    ("unique_consecutive", unique_consecutive),
    ("unique_dim", unique_dim),
    ("unsqueeze", unsqueeze),
    ("unsqueeze_", unsqueeze_),
    ("upsample_bicubic2d", upsample_bicubic2d),
    ("upsample_linear1d", upsample_linear1d),
    ("upsample_linear1d_backward", upsample_linear1d_backward),
    ("upsample_nearest1d", upsample_nearest1d),
    ("upsample_nearest2d", upsample_nearest2d),
    ("upsample_nearest3d", upsample_nearest3d),
    ("upsample_trilinear3d", upsample_trilinear3d),
    ("var", var),
    ("var.correction", var_correction),
    ("var.dim", var_dim),
    ("var_mean.correction", var_mean),
    ("vdot", vdot),
    ("view_copy", view_copy),
    ("vstack", vstack),
    ("where.self", where_self),
    ("where.self_out", where_self_out),
    ("xlogy.OutScalar_Other", xlogy_tensor_scalar_out),
    ("xlogy.OutScalar_Self", xlogy_scalar_tensor_out),
    ("xlogy.OutTensor", xlogy_out),
    ("xlogy.Scalar_Other", xlogy_tensor_scalar),
    ("xlogy.Scalar_Self", xlogy_scalar_tensor),
    ("xlogy.Tensor", xlogy),
    ("zero", zero),
    ("zero.out", zero_out),
    ("zero_", zero_),
    ("zeros", zeros),
    ("zeros_like", zeros_like),
)

# Cache mapping from function name -> list of _FULL_CONFIG entries for quick lookup
FULL_CONFIG_BY_FUNC = {}
for _item in _FULL_CONFIG:
    if not _item or len(_item) < 2:
        continue
    fn = _item[1]
    func_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
    FULL_CONFIG_BY_FUNC.setdefault(func_name, []).append(_item)

# Friendly names for only_enable(include=[...]) when the registered impl is *.out
for _alias, _target in (
    ("softmax", "softmax_out"),
    ("softmax_backward", "softmax_backward_out"),
    ("log_softmax", "log_softmax_out"),
    ("log_softmax_backward", "log_softmax_backward_out"),
):
    if _target in FULL_CONFIG_BY_FUNC:
        FULL_CONFIG_BY_FUNC.setdefault(_alias, []).extend(FULL_CONFIG_BY_FUNC[_target])


def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register all FlagGems ops except those explicitly excluded.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        unused: Which ops to skip. Supported forms:
            - list/tuple/set of function names (e.g., ["masked_fill", "mul"]).
            - str path to a YAML file ending with .yml/.yaml containing an
              `exclude:` list.
            - "default" or None: auto-load vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagGems logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Notes:
        - If the exclude list/YAML resolves to empty, all ops are registered.
    """
    global current_work_registrar
    exclude_ops = resolve_user_setting(unused, "exclude")
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=[],
        user_exclude_ops=exclude_ops,
        cpp_patched_ops=list(set(aten_patch_list)),
        lib=lib,
    )
    setup_flaggems_logging(path=path, record=record, once=once)


def only_enable(
    lib=aten_lib,
    include=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register only the specified FlagGems ops and skip the rest.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        include: Which ops to register. Supported forms:
            - list/tuple/set of function names (e.g., ["rms_norm", "softmax"]).
            - str path to a YAML file ending with .yml/.yaml (expects a list or
              an `include:` key).
            - "default" or None: auto-load vendor/arch-specific
                runtime/backend/_<vendor>/[<arch>/]only_enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagGems logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Classic usage:
        - Only register a few ops:
            only_enable(include=["rms_norm", "softmax"])
        - Use vendor default YAML:
            only_enable(include="default")  # or include=None
        - Use a custom YAML:
            only_enable(include="/path/to/only_enable.yaml")

    Notes:
        - If the include list/YAML resolves to empty or none of the names match
          known ops, the function warns and returns without registering.
    """
    include_ops = resolve_user_setting(include, "include")
    if not include_ops:
        warnings.warn(
            "only_enable failed: No include entries resolved from list or yaml."
        )
        return

    global current_work_registrar
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=include_ops,
        user_exclude_ops=[],
        cpp_patched_ops=list(set(aten_patch_list)),
        full_config_by_func=FULL_CONFIG_BY_FUNC,
        lib=lib,
    )
    setup_flaggems_logging(path=path, record=record, once=once)


class use_gems:
    """
    The 'include' parameter has higher priority than 'exclude'.
    When 'include' is not None, use_gems will not process 'exclude'.
    """

    def __init__(self, exclude=None, include=None, record=False, once=False, path=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.exclude = exclude if isinstance(exclude, (list, tuple, set, str)) else []
        self.include = include if isinstance(include, (list, tuple, set, str)) else []
        self.registrar = GeneralOpRegistrar
        self.record = record
        self.once = once
        self.path = path

    def __enter__(self):
        if self.include:
            only_enable(
                lib=self.lib,
                include=self.include,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )
        else:
            enable(
                lib=self.lib,
                unused=self.exclude,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        if torch.__version__ >= "2.5":
            self.lib._destroy()
        del self.lib
        del self.exclude
        del self.include
        del self.registrar
        del current_work_registrar
        if self.record:
            teardown_flaggems_logging()

    @property
    def experimental_ops(self):
        import flag_gems.experimental_ops

        return flag_gems.experimental_ops


def all_registered_ops():
    return current_work_registrar.get_all_ops()


def all_registered_keys():
    return current_work_registrar.get_all_keys()


__all__ = [
    "all_registered_keys",
    "all_registered_ops",
    "enable",
    "flagtune",
    "only_enable",
    "use_gems",
]
