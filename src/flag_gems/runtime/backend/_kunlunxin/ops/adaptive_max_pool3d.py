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
def adaptive_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    n_elems,  # in_n * in_c * out_d * out_h * out_w
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    WIN_D: tl.constexpr,
    WIN_H: tl.constexpr,
    WIN_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adaptive max pool 3d forward (Kunlunxin/XPU).

    One lane per output position.  The adaptive window
    [start, end) = [floor(o * in / out), ceil((o + 1) * in / out)) is scanned
    in row-major (d, h, w) order with the same tie-break as the ATen reference
    (first tap attaining the strict max; NaNs win over any value).  Indices are
    stored as the flat spatial index d * H * W + h * W + w, matching ATen.

    This replaces the vendor ``adaptive_max_pool3d`` for all dtypes: the
    vendor XDNN wrapper rejects bfloat16 (``scalar type of (ret, ret_indices) :
    (kbfloat16, kint64) combined is unsupported``) and returns uninitialized
    index memory for float16/float32 on this stack, so every
    ``adaptive_max_pool3d_backward`` generated against those indices either
    faults or disagrees with the CPU reference.

    Windows are provably non-empty; candidate addresses are clamped to the
    window start so the (unmasked) loads can never leave the (n, c) plane
    (the XPU backend treats compound i1 masked loads as a slow path).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    out_per_nc = out_d * out_h * out_w
    nc = safe_offsets // out_per_nc
    rem = safe_offsets % out_per_nc
    od = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    d_start = (od * in_d) // out_d
    d_end = tl.minimum(((od + 1) * in_d + out_d - 1) // out_d, in_d)
    h_start = (oh * in_h) // out_h
    h_end = tl.minimum(((oh + 1) * in_h + out_h - 1) // out_h, in_h)
    w_start = (ow * in_w) // out_w
    w_end = tl.minimum(((ow + 1) * in_w + out_w - 1) // out_w, in_w)

    plane_base = input_ptr + nc * (in_d * in_h * in_w)
    in_hw = in_h * in_w

    acc_val = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    acc_idx = tl.full((BLOCK,), -1, dtype=tl.int32)

    for kd in range(0, WIN_D):
        d = d_start + kd
        d_ok = d < d_end
        d_safe = tl.where(d_ok, d, d_start)
        for kh in range(0, WIN_H):
            h = h_start + kh
            h_ok = h < h_end
            h_safe = tl.where(h_ok, h, h_start)
            for kw in range(0, WIN_W):
                w = w_start + kw
                w_ok = w < w_end
                w_safe = tl.where(w_ok, w, w_start)
                value = tl.load(
                    plane_base + d_safe * in_hw + h_safe * in_w + w_safe
                ).to(tl.float32)
                active = d_ok & h_ok & w_ok
                is_new = active & (
                    (value > acc_val) | (value != value) | (acc_idx < 0)
                )
                acc_val = tl.where(is_new, value, acc_val)
                acc_idx = tl.where(is_new, d * in_hw + h * in_w + w, acc_idx)

    tl.store(
        output_ptr + offsets,
        acc_val.to(output_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(indices_ptr + offsets, acc_idx.to(tl.int64), mask=mask)


def adaptive_max_pool3d(input: torch.Tensor, output_size, return_indices=False):
    """Adaptive max pool 3d forward (Kunlunxin/XPU implementation)."""
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D")
    input = input.contiguous()

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)
    output_size = tuple(output_size)
    if len(output_size) != 3:
        raise ValueError(f"Invalid output_size: {output_size}")

    in_n, in_c, in_d, in_h, in_w = input.shape

    output = torch.empty(
        (in_n, in_c) + output_size,
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (in_n, in_c) + output_size,
        device=input.device,
        dtype=torch.int64,
    )

    if output.numel() == 0 or input.numel() == 0:
        if return_indices:
            return output, indices
        return output

    out_d, out_h, out_w = output_size
    n_elems = in_n * in_c * out_d * out_h * out_w

    # Exact upper bound of the adaptive window length: exactly in/out when the
    # ratio is an integer, at most floor(in/out) + 2 otherwise.  The naive
    # ``(in + out - 1) // out + 1`` spends (3/2)^3 = 3.4x of the loop bodies
    # on the common integer-ratio shapes (in = 2 * out).
    win_d = in_d // out_d + (0 if in_d % out_d == 0 else 2)
    win_h = in_h // out_h + (0 if in_h % out_h == 0 else 2)
    win_w = in_w // out_w + (0 if in_w % out_w == 0 else 2)

    # 64-lane tiles with num_warps=1 (see adaptive_max_pool2d: the XPU unroll
    # control pass keeps the window scan within its vrf budget only for small
    # tiles; do not raise BLOCK_SIZE without re-measuring every shape).  For
    # the common window of at most 2 per dim (out = in / 2) the scan also
    # fits 128-lane tiles / num_warps=2, which is measurably faster.
    if max(win_d, win_h, win_w) <= 2:
        block, num_warps = 128, 2
    else:
        block, num_warps = 64, 1
    grid = (triton.cdiv(n_elems, block),)

    with torch_device_fn.device(input.device):
        adaptive_max_pool3d_forward_kernel[grid](
            input,
            output,
            indices,
            n_elems,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            WIN_D=win_d,
            WIN_H=win_h,
            WIN_W=win_w,
            BLOCK=block,
            num_warps=num_warps,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    if return_indices:
        return output, indices
    return output