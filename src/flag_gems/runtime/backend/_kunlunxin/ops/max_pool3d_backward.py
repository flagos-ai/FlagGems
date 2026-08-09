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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_pool3d_backward_flat_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling parameters
    BLOCK: tl.constexpr,
):
    """Backward kernel for 3-D max pooling (gather, per-iteration accumulate).

    Grid: (cdiv(N * C * D * H * W, BLOCK),)
    For each input position, iterate over every kernel offset to find the
    output positions that could have pooled it, and accumulate the gradient
    contribution of the window whose stored index equals this input's flat
    spatial index (d * H * W + h * W + w).

    Design constraints verified on Kunlunxin XPU (2026-08-20):
    - runtime trip-count loops crash the backend compiler (uni_sram OOM);
    - tl.sum over a [BLOCK, KK] window tile miscompiles for this workload
      (where-in-reduce and masked-load variants both diverge);
    - tl.atomic_add scatter loses updates (~1e-5 per op, seed-dependent);
    - a fully-unrolled static kernel with `grad_acc += tl.where(...)` matches
      the proven Kunlunxin 2-D pattern and is exact and deterministic.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_mask = offsets < total
    in_hw = in_h * in_w
    in_spatial = in_d * in_hw
    nc_idx = offsets // in_spatial
    rem = offsets % in_spatial
    id_ = rem // in_hw
    rem2 = rem % in_hw
    ih = rem2 // in_w
    iw = rem2 % in_w
    nc_safe = tl.where(input_mask, nc_idx, 0)
    input_flat_idx = id_ * in_hw + ih * in_w + iw
    out_dhw = out_d * out_h * out_w

    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kd in tl.static_range(0, kernel_d):
        d_num = id_ + padding_d - kd * dilation_d
        d_nonnegative = d_num >= 0
        d_num_safe = tl.where(d_nonnegative, d_num, 0)
        od = d_num_safe // stride_d
        d_rem = d_num_safe - od * stride_d
        for kh in tl.static_range(0, kernel_h):
            h_num = ih + padding_h - kh * dilation_h
            h_nonnegative = h_num >= 0
            h_num_safe = tl.where(h_nonnegative, h_num, 0)
            oh = h_num_safe // stride_h
            h_rem = h_num_safe - oh * stride_h
            for kw in tl.static_range(0, kernel_w):
                w_num = iw + padding_w - kw * dilation_w
                w_nonnegative = w_num >= 0
                w_num_safe = tl.where(w_nonnegative, w_num, 0)
                ow = w_num_safe // stride_w
                w_rem = w_num_safe - ow * stride_w
                valid = (
                    input_mask
                    & d_nonnegative
                    & (d_rem == 0)
                    & (od < out_d)
                    & h_nonnegative
                    & (h_rem == 0)
                    & (oh < out_h)
                    & w_nonnegative
                    & (w_rem == 0)
                    & (ow < out_w)
                )
                od_safe = tl.where(valid, od, 0)
                oh_safe = tl.where(valid, oh, 0)
                ow_safe = tl.where(valid, ow, 0)
                out_offset = (
                    nc_safe * out_dhw
                    + od_safe * (out_h * out_w)
                    + oh_safe * out_w
                    + ow_safe
                )
                index_value = tl.load(
                    indices_ptr + out_offset, mask=valid, other=-1
                ).to(tl.int32)
                match = valid & (index_value == input_flat_idx)
                grad_value = tl.load(
                    grad_output_ptr + out_offset, mask=valid, other=0.0
                ).to(tl.float32)
                grad_acc += tl.where(match, grad_value, 0.0)

    tl.store(grad_input_ptr + offsets, grad_acc, mask=input_mask)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    """Backward pass for 3-D max pooling (Kunlunxin)."""
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(original_dtype)

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
    return max_pool3d_backward(
        grad_output,
        self,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )