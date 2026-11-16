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

import logging
import math
import struct
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK = 1024
# One extra candidate absorbs the (at most one-step) gap between ATen's
# ``ceil(idx * scale)`` start/end window and the analytic window size.
_EXTRA_CANDIDATES = 1


@libentry()
@triton.jit
def _upsample_nearest1d_backward_kernel(
    grad_output,
    grad_input,
    total,
    channels,
    IW,
    OW,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_w,
    rec: tl.constexpr,
    scale: tl.constexpr,
    cand: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_UINT8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather backward: one lane per ``grad_input`` element.

    For every input position ``iw`` the kernel scans the candidate output
    window and accepts a candidate only when ATen's *forward* source-index
    formula maps it back to ``iw``:

        src = min(int(floor(ow * reciprocal_scale)), IW - 1)

    Using the official forward formula as the acceptance test keeps the
    topmost-input clamp tail (every ``ow`` past the clamp belongs to
    ``IW - 1``) and the floating-point boundaries bit-compatible with the
    native scatter+atomic algorithm, without needing atomics.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    # The store mask is not honoured on this backend (it degrades to an address
    # select and every lane writes), so tail lanes decode a *valid* location
    # (element 0) and recompute the same value as lane 0: their write is then
    # harmless instead of clobbering an unrelated element.
    rem = tl.where(mask, offsets, 0)

    iw = rem % IW
    nc = rem // IW
    c = nc % channels
    n = nc // channels

    # Candidate window start; slide the topmost input so that its window ends
    # at OW and therefore absorbs every output position held by the clamp.
    w_lo = tl.ceil(iw.to(tl.float32) * scale).to(tl.int32) - 1
    w_lo = tl.where(iw == IW - 1, tl.maximum(w_lo, OW - cand), w_lo)

    go_base = n * go_stride_n + c * go_stride_c
    gi_base = n * gi_stride_n + c * gi_stride_c

    if IS_UINT8:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    elif IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in tl.static_range(cand):
        ow = w_lo + k
        src = tl.minimum((ow.to(tl.float32) * rec).to(tl.int32), IW - 1)
        ok = (ow >= 0) & (ow < OW) & (src == iw)
        # Masked loads are honoured inconsistently on this backend; clamp every
        # address and neutralize rejected lanes in registers instead.
        sa = tl.where(ok, ow, 0)
        value = tl.load(
            grad_output + go_base + sa * go_stride_w,
            mask=ok,
            other=0,
        )
        if IS_UINT8:
            acc += tl.where(ok, value.to(tl.int32), 0)
        elif IS_FP64:
            acc += tl.where(ok, value.to(tl.float64), 0.0)
        else:
            acc += tl.where(ok, value.to(tl.float32), 0.0)

    tl.store(
        grad_input + gi_base + iw * gi_stride_w,
        acc.to(grad_input.dtype.element_ty),
        mask=mask,
    )


def _validate_args(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
) -> tuple[int, int, int, int]:
    if grad_output.device.type != device.name:
        raise RuntimeError(
            f"Expected grad_output on {device.name}, but got {grad_output.device.type}"
        )
    if grad_output.ndim != 3:
        raise RuntimeError("Expected grad_output to be a 3D tensor")
    if len(output_size) != 1:
        raise RuntimeError("Expected output_size to contain one element")
    if len(input_size) != 3:
        raise RuntimeError("Expected input_size to contain three elements")
    if not grad_output.is_floating_point() and grad_output.dtype != torch.uint8:
        raise RuntimeError(
            f'"upsample_nearest1d_backward_out_frame" not implemented for '
            f"'{grad_output.dtype}'"
        )

    output_w = int(output_size[0])
    batch, channels, input_w = (int(value) for value in input_size)
    if input_w <= 0 or output_w <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if tuple(grad_output.shape) != (batch, channels, output_w):
        raise RuntimeError(
            f"Expected grad_output shape {(batch, channels, output_w)}, "
            f"but got {tuple(grad_output.shape)}"
        )
    return batch, channels, input_w, output_w


def _as_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _reciprocal_scale(in_size, out_size, scale):
    """Match ATen ``compute_scales_value<float>`` (forward reciprocal)."""
    if scale is not None and scale > 0:
        return _as_float32(1.0 / float(scale))
    return _as_float32(float(in_size) / float(out_size))


def _upsample_nearest1d_backward_impl(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float],
    grad_input: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, channels, input_w, output_w = _validate_args(
        grad_output, output_size, input_size
    )
    if grad_input is None:
        grad_input = torch.empty(
            (batch, channels, input_w),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected grad_input on {grad_output.device}, "
                f"but got {grad_input.device}"
            )
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected grad_input dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype}"
            )
        grad_input.resize_((batch, channels, input_w))

    total = grad_input.numel()
    if total == 0:
        return grad_input

    backward_scale = _as_float32(
        float(scales) if scales is not None and scales > 0.0 else output_w / input_w
    )
    rec = _reciprocal_scale(input_w, output_w, scales)
    cand = max(1, math.ceil(backward_scale)) + _EXTRA_CANDIDATES
    # The topmost input also absorbs every output position clamped onto it, so
    # its window can be much wider than ``ceil(scale)`` when an explicit,
    # size-inconsistent scale is supplied.
    top_start = math.ceil((input_w - 1) * backward_scale)
    cand = min(output_w, max(cand, output_w - top_start))

    grid = (triton.cdiv(total, _BLOCK),)
    with torch_device_fn.device(grad_output.device):
        _upsample_nearest1d_backward_kernel[grid](
            grad_output,
            grad_input,
            total,
            channels,
            input_w,
            output_w,
            *grad_output.stride(),
            *grad_input.stride(),
            rec=rec,
            scale=backward_scale,
            cand=cand,
            IS_FP64=grad_output.dtype == torch.float64,
            IS_UINT8=grad_output.dtype == torch.uint8,
            BLOCK=_BLOCK,
            num_warps=1,
            buffer_size_limit=2048,
        )
    return grad_input


def upsample_nearest1d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST1D_BACKWARD")
    return _upsample_nearest1d_backward_impl(
        grad_output, output_size, input_size, scales, None
    )


def upsample_nearest1d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT")
    return _upsample_nearest1d_backward_impl(
        grad_output, output_size, input_size, scales, grad_input
    )


# benchmark/test_upsample_nearest1d_backward.py binds its candidate from the
# *generic* module path, and that generic Triton kernel does not lower on XPU3
# (TritonXPUUnrollControl / uni_sram).  Rebind the generic module attributes to
# this vendor implementation so both call paths run the kernel that works here
# (same pattern as _kunlunxin/ops/_batch_norm_impl_index.py).
import sys as _sys  # noqa: E402

_generic_module = _sys.modules.get("flag_gems.ops.upsample_nearest1d_backward")
if _generic_module is not None:
    _generic_module.upsample_nearest1d_backward = upsample_nearest1d_backward
    _generic_module.upsample_nearest1d_backward_grad_input = (
        upsample_nearest1d_backward_grad_input
    )
