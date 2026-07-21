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

import importlib.util
import os
import re

from backend_utils import VendorDescriptor

# NOTE: transfer_to_gcu is not used anywhere
# try:
#     from torch_gcu import transfer_to_gcu  # noqa: F401
# except Exception:
#    logger.warning("torch_gcu not installed")

# TODO: Revise the following imports to be exception free
if importlib.util.find_spec("triton.backends.enflame") is None:
    from triton_gcu.triton.driver import _GCUDriver
else:
    from triton.backends.enflame.driver import _GCUDriver

driver = _GCUDriver()
arch = driver.get_arch()
arch_version = int(re.search(r"gcu(\d+)", arch).group(1))

vendor_info = VendorDescriptor(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=False,
    tle_enabled=True,
)

os.environ["ARCH"] = str(arch_version)
ARCH_MAP = {"3": "gcu300", "4": "gcu400"}
# i64 to/copy is not supported in gcu300
if arch_version == 300:
    CUSTOMIZED_UNUSED_OPS = (
        "concat",
        "concatenate",
        "_to_copy",
        "to_copy",
        "copy_",
        "__and__.Scalar",
        "__and__.Tensor",
        "__and__",
        "_amp_foreach_non_finite_check_and_unscale_",
        "_batch_norm_no_update",
        "_functional_sym_constrain_range",
        "_fused_adam",
        "_fused_adam_",
        "_jagged_to_padded_dense_forward",
        "_linalg_eigvals",
        "_masked_scale",
        "_prelu_kernel_backward",
        "_sparse_semi_structured_mm",
        "_thnn_fused_lstm_cell",
        "_thnn_fused_lstm_cell_backward_impl",
        "_unsafe_masked_index_put_accumulate",
        "_unsafe_view",
        "addmm",
        "addmm_dtype",
        "addmm_dtype_out",
        "addmm_out",
        "pad",
        "bmm",
        "bmm_out",
        "gelu_backward",
        "gelu",
        "gelu_",
        "cat",
        "cat_out",
        "_upsample_bilinear2d_aa",
        "_upsample_nearest_exact2d_backward",
        "acosh",
        "acosh_",
        "adaptive_max_pool3d_backward",
        "addcdiv_",
        "addcmul_",
        "addmm_",
        "alpha_dropout",
        "amin",
        "amin_",
        "arccos",
        "arccos_",
        "arcsin",
        "arcsin_out",
        "arcsin_",
        "arctan",
        "arctan_",
        "asin",
        "asin_",
        "baddbmm_out",
        "beam_search_score",
        "beam_search_score_",
        "bitwise_right_shift_",
        "broadcast_to",
        "bucketize_Tensor",
        "channel_shuffle",
        "deg2rad_out",
        "deg2rad_",
        "dequantize",
        "dequantize_self",
        "diagonal_copy",
        "digamma",
        "empty",
        "erfinv",
        "erfinv_",
        "expand",
        "expand_",
        "frac",
        "frac_",
        "greater_equal_Scalar",
        "greater_equal_Tensor",
        "greater_equal__Tensor",
        "im2col",
        "is_nonzero",
        "kthvalue",
        "less_Scalar",
        "less_Tensor",
        "less_equal_Scalar",
        "less_equal_Tensor",
        "lgamma",
        "lgamma_",
        "linalg_cholesky",
        "log_normal_",
        "logical_not_",
        "lt_Scalar",
        "lt_Tensor",
        "matmuladd",
        "max_unpool2d",
        "mish",
        "mish_",
        "mse_loss_backward",
        "multiply_",
        "multiply_Scalar",
        "multiply_Tensor",
        "mvlgamma_",
        "narrow_copy",
        "new_ones",
        "nextafter_",
        "permute_copy",
        "range",
        "reflection_pad3d",
        "reflection_pad3d_out",
        "resize",
        "resize_",
        "rnn_relu",
        "rrelu_with_noise_functional",
        "scalar_tensor",
        "sinc",
        "sinc_",
        "soft_margin_loss_backward",
        "softplus_backward",
        "special_chebyshev_polynomial_v",
        "special_chebyshev_polynomial_w",
        "special_chebyshev_polynomial_w_out",
        "special_gammainc",
        "special_hermite_polynomial_h",
        "special_i0e_out",
        "special_i1_out",
        "special_log_softmax",
        "special_shifted_chebyshev_polynomial_u",
        "special_shifted_chebyshev_polynomial_u_",
        "subtract_Tensor",
        "sym_stride",
        "threshold_",
        "true_divide_out",
        "unbind_copy",
    )
elif arch_version == 400 or arch_version == 410:
    CUSTOMIZED_UNUSED_OPS = (
        "to_copy",
        "copy_",
        "_to_copy",
    )

__all__ = ["*"]
