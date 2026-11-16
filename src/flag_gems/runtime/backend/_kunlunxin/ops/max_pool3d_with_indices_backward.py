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
"""Kunlunxin override for ``aten::max_pool3d_with_indices_backward``.

This module only fixes one defect of the existing shared implementation in
``max_pool3d_backward.py``: the small-tensor ("flat gather") path allocates

    grad_input = torch.zeros_like(input, dtype=torch.float32)

with the *default* ``memory_format=torch.preserve_format``.  For a
``channels_last_3d`` input that produces an NDHWC-strided tensor while the
kernel addresses ``grad_input`` as a linear NCDHW plane, so every scattered
gradient lands at the wrong address (9/36 accuracy cells failed at HEAD).

The dispatch below is intentionally a faithful copy of
``_kunlunxin/ops/max_pool3d_backward.max_pool3d_backward`` (which must not be
modified) reusing its kernels/helpers, with the single fix being that both
grad_input allocations are pinned to NCDHW-contiguous storage.  The slot path
already allocated ``torch.empty((n, c, d, h, w))`` (contiguous) and is
therefore unchanged.

Note: max-pool backward is index-driven; ``input`` contributes only its shape
(and the zero-initialisation), never its values.
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

# Reuse the proven kernels / helpers of the shared backend implementation
# without touching that file.
from flag_gems.runtime.backend._kunlunxin.ops.max_pool3d_backward import (
    _max_pool3d_backward_slot,
    _parse_pool3d_params,
    max_pool3d_backward_flat_kernel,
)

logger = logging.getLogger(__name__)


def max_pool3d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for 3-D max pooling with indices (Kunlunxin).

    Matches the ATen signature of aten::max_pool3d_with_indices_backward, the
    op invoked by the PyTorch autograd formula of max_pool3d_with_indices.
    """
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_WITH_INDICES_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    if in_n * in_c * in_d * in_h * in_w == 0:
        return torch.zeros_like(
            self, dtype=original_dtype, memory_format=torch.contiguous_format
        )

    # Two-phase slot path (no atomics, deterministic) for the common small
    # slot-count case on large-enough tensors; falls back to the flat gather
    # kernel otherwise.  Mirrors the dispatch thresholds of the shared
    # implementation.
    KDN = ((kd - 1) * dd) // sd + 1
    KHN = ((kh - 1) * dh) // sh + 1
    KWN = ((kw - 1) * dw) // sw + 1
    N_SLOT = KDN * KHN * KWN
    n_out_total = in_n * in_c * out_d * out_h * out_w
    if N_SLOT <= 9 and n_out_total >= 8192:
        return _max_pool3d_backward_slot(
            grad_output,
            indices,
            in_n,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            dd,
            dh,
            dw,
        ).to(original_dtype)

    # Flat gather fallback. The one behavioural difference from the shared
    # implementation: pin NCDHW-contiguous storage so a channels_last_3d
    # `self` cannot leak its physical layout into the linear scatter.
    grad_input = torch.zeros_like(
        self, dtype=torch.float32, memory_format=torch.contiguous_format
    )
    total = grad_input.numel()
    block = 64
    grid = (triton.cdiv(total, block),)

    with torch_device_fn.device(grad_input.device):
        max_pool3d_backward_flat_kernel[grid](
            grad_output,
            indices,
            grad_input,
            total,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            dd,
            dh,
            dw,
            block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input.to(original_dtype)


# Dedicated tests/benchmarks may call ``flag_gems.ops.<op>`` directly, which
# bypasses ``SpecOpRegistrar`` (it only rewrites the top-level ``flag_gems``
# namespace). Publish the backend impl there too
# (wiki/patterns/xpu3-dedicated-tests-call-flag-gems-ops.md).
import sys as _sys

_generic_ops_module = _sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(
        _generic_ops_module,
        "max_pool3d_with_indices_backward",
        max_pool3d_with_indices_backward,
    )
