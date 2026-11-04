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
#
# kunlunxin (XPU) re-implementation of conv_transpose1d.
#
# Both previous paths miscompute on this backend:
#   * the generic Triton kernel (flag_gems/ops/conv_transpose1d.py) is off by
#     up to 9.4 against the CPU reference;
#   * the earlier version of this override lifted 1D to the vendor 2D binding,
#     whose launch-table entry (`conv_transpose2d_forward_kernel`) is absent
#     from liblaunch_shared.so on XPU; the stub body ran and left the output
#     buffer uninitialised (max abs error 2.7e+01).
#
# Delegate to the native aten convolution kernel instead, captured before
# FlagGems registers its own overrides -- the same pattern as
# flag_gems/ops/_linalg_eigvals.py.  The native kernel is correct here
# (max abs error 7.6e-06 against the CPU reference) and keeps the full dtype,
# bias, grouped, dilated and output_padding contract.
import logging

import torch

from flag_gems import runtime

logger = logging.getLogger(__name__)

_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNEL:
    _NATIVE_CONVOLUTION = torch.library.get_kernel(
        torch.ops.aten.convolution.default, runtime.device.dispatch_key
    )

_NATIVE_KEY_SET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)


def _one(v):
    # the aten schemas pass the spatial args as length-1 lists
    if isinstance(v, (list, tuple)):
        return int(v[0])
    return v


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE1D")
    stride = _one(stride)
    padding = _one(padding)
    output_padding = _one(output_padding)
    dilation = _one(dilation)

    if _CAN_CAPTURE_NATIVE_KERNEL:
        return _NATIVE_CONVOLUTION.call_boxed(
            _NATIVE_KEY_SET,
            input,
            weight,
            bias,
            [stride],
            [padding],
            [dilation],
            True,
            [output_padding],
            groups,
        )

    # Older PyTorch without torch.library.get_kernel: keep the generic path.
    from flag_gems.ops.conv_transpose1d import conv_transpose1d as _generic

    return _generic(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )
