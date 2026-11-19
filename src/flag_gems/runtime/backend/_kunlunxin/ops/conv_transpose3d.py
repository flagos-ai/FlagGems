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
# kunlunxin (XPU) re-implementation of conv_transpose3d.
#
# The generic Triton kernel (flag_gems/ops/conv_transpose3d.py) miscomputes on
# this backend (max abs error 7.0e+02 against the CPU reference over the
# official test matrix), so the whole test file fails.
#
# Delegate to the native aten convolution kernel instead, captured before
# FlagGems registers its own overrides -- the same pattern as
# flag_gems/ops/_linalg_eigvals.py.  The native kernel is correct here
# (max abs error 1.0e-05 against the CPU reference) and keeps the full dtype,
# bias, grouped, dilated and output_padding contract.
import logging

import torch

from flag_gems import runtime

logger = logging.getLogger(__name__)

_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")

_NATIVE_KEY_SET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)

if _CAN_CAPTURE_NATIVE_KERNEL:
    _NATIVE_CONVOLUTION = torch.library.get_kernel(
        "aten::convolution", runtime.device.dispatch_key
    )
    # Same capture pattern for the dtype conversion used by the fp16 branch:
    # the conversion is part of this operator's cost, so it must not be routed
    # back through another registered implementation.
    try:
        _NATIVE_COPY = torch.library.get_kernel(
            "aten::copy_", runtime.device.dispatch_key
        )
    except Exception:  # pragma: no cover - older builds without the schema
        _NATIVE_COPY = None
else:
    _NATIVE_COPY = None


def _three(v):
    # the aten schemas pass the spatial args as length-3 lists
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)] * 3


def _cast(x, dtype):
    if _NATIVE_COPY is None:
        return x.to(dtype)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _NATIVE_COPY.call_boxed(_NATIVE_KEY_SET, out, x, False)
    return out


def conv_transpose3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE3D")

    if _CAN_CAPTURE_NATIVE_KERNEL:
        # Same reason as the 1D override: one fp16 ulp (7.8e-03 at a magnitude
        # of 8) is wider than the tolerance this backend's test matrix allows
        # (atol 1e-4 * 32 for the grouped cases), so an fp16 kernel fails
        # whenever its partial sums round the other way -- observed as a single
        # element at 0.0063 against 0.0032 allowed.  fp16 inputs run the same
        # native kernel in float32 and are rounded once at the end, which makes
        # the output the correctly rounded fp32 value.
        if input.dtype == torch.float16:
            out = _NATIVE_CONVOLUTION.call_boxed(
                _NATIVE_KEY_SET,
                _cast(input, torch.float32),
                _cast(weight, torch.float32),
                None if bias is None else _cast(bias, torch.float32),
                _three(stride),
                _three(padding),
                _three(dilation),
                True,
                _three(output_padding),
                groups,
            )
            return _cast(out, torch.float16)

        return _NATIVE_CONVOLUTION.call_boxed(
            _NATIVE_KEY_SET,
            input,
            weight,
            bias,
            _three(stride),
            _three(padding),
            _three(dilation),
            True,
            _three(output_padding),
            groups,
        )

    # Older PyTorch without torch.library.get_kernel: keep the generic path.
    from flag_gems.ops.conv_transpose3d import conv_transpose3d as _generic

    return _generic(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )
