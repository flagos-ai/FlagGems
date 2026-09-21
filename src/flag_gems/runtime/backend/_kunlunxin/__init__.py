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

from backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "atan2_out",
    "cumsum",
    "grid_sampler_3d_backward",
    "randperm",
    "topk",
    "unique",
    "slice",
    "conv_transpose1d",
    "mkldnn_rnn_layer",
    "_linalg_eigvals",
    "linalg_eig",
    "linalg_eigvals",
    "linalg_eigvals.out",
    "linalg_eigvals_out",
)


__all__ = ["*"]


# Reference-side monkey patches (ported from the _sunrise backend template,
# 2026-09-04): try the original aten op first and fall back to a CPU reference
# only on NotImplementedError while outside flag_gems.use_gems(). This repairs
# the reference side of official tests on this platform, where ops like
# aten::_flash_attention_forward / aten::_conv_depthwise2d /
# aten::cudnn_convolution have no usable kernel.
from . import monkey_patch as _monkey_patch  # noqa: E402

_monkey_patch.apply_kunlunxin_monkey_patches()
