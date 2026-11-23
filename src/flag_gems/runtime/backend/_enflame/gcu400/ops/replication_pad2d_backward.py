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

"""replication_pad2d backward (Enflame GCU400): single launch, no atomics, row-grid striding implementation.

2026-11-15 fix + optimization
=====================
1) **grid.x overflow crash (functional defect)**
   The old implementation had two paths: with interior_elements <= 16384 a "one program per row" fused kernel,
   otherwise a fallback of "interior copy + edge atomic add" two kernels. The fallback grid is
   `nc_total * H * cdiv(W, INTERIOR_BLOCK)` / `cdiv(interior_elements, BLOCK)`,
   which needs 131072 for (4,64,256,512) and more for (8,128,1024,1024), exceeding the hardware grid.x <= 65535,
   directly raising `OutOfResources: out of resource: grid.x` (all 18 cases of 4x64x256x512 in tests
   failed). The new implementation puts the "input row" loop inside the kernel as grid-stride
   (`for row in range(pid, ROWS, G)`), so grid depends only on row count and tuning caps and no shape
   can overflow; the two-kernel fallback is also removed (its accuracy defect is point 2).
2) **fp16/bf16 edge accuracy**
   The old edge kernel used `tl.atomic_add` to accumulate directly into low-precision output, rounding once per pad
   element
   (comparing random grad_output with a CPU float64 reference, 0.1% of boundary elements differ by
   0.047). The new implementation accumulates all output elements mapping to one input element in fp32 in registers,
   casts once at the end, and needs no atomics at all.
3) **ph dimension changed to sequential accumulation (main performance source)**
   A `[BLOCK_PH, BLOCK_W]` 2D tile followed by `tl.sum(axis=0)` becomes a cross-lane reduction on GCU,
   the biggest cost of this kernel; switching to `tl.static_range(BLOCK_PH)` sequential accumulation into a 1D register
   vector raised the same case group's geomean from 17.19x to 37.59x (+2.2x, see report). Column direction is blocked by
   `ELEM_CAP`, so extra-wide rows (W > BLOCK_W) do not blow up registers.

Mapping semantics (replication backward): input row (nc, ih) receives output rows in the closed interval
[oh_lo, oh_hi] = [ih == 0 ? 0 : PT + ih, ih == H-1 ? OH-1 : PT+ih]; in the column direction left pad
columns fold to iw = 0 and right pad columns fold to iw = W-1. Each grad_output element is read once,
so read/write traffic is OH*OW reads + H*W writes (optimal).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ===== Tuning constants (2026-11-15 interleaved A/B calibration, min-of-3 rounds) =====
# grid = min(rows, SPC * GRID_MULT, 65535): GRID_MULT=1 (48 programs) is optimal,
# 2/4/8x give -2.7%/-5.8%/-11.9% respectively.
GRID_MULT = 1
SPC = 48
MAX_GRID = 65535
# Element budget of a single load: 8192 ties 16384/32768 (bounded by next_pow2(W)),
# 1024/2048 give -31%/-11% respectively.
ELEM_CAP = 8192
# num_warps: 1 is optimal, 2 gives -26%.
WARPS = 1
# ==============================================================


@triton.jit
def replication_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    ROWS,
    H,
    W,
    OH,
    OW,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    GO_ROW: tl.constexpr,
    GO_NC: tl.constexpr,
    GI_ROW: tl.constexpr,
    GI_NC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    NCHUNK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    iw0 = tl.arange(0, BLOCK_W)
    pw = tl.arange(0, BLOCK_PAD)

    for row in range(pid, ROWS, G):
        nc = row // H
        ih = row - nc * H
        # Closed interval of output rows received by this input row (H == 1 merges both where branches to cover all OH)
        oh_lo = tl.where(ih == 0, 0, PT + ih)
        oh_hi = tl.where(ih == H - 1, OH - 1, PT + ih)
        go_nc = grad_output_ptr + nc * GO_NC
        gi_row = grad_input_ptr + row * GI_ROW

        for chunk in tl.static_range(NCHUNK):
            iw = chunk * BLOCK_W + iw0
            iwv = iw < W
            total = tl.zeros([BLOCK_W], dtype=tl.float32)
            # Sequential accumulation along ph (avoiding the 2D tile's cross-lane reduction)
            for p in tl.static_range(BLOCK_PH):
                oh = oh_lo + p
                if oh <= oh_hi:
                    total += tl.load(
                        go_nc + oh * GO_ROW + (PL + iw), mask=iwv, other=0.0
                    ).to(tl.float32)
            # Left/right pad columns: only iw=0 / iw=W-1 of the whole row receive them
            if PL > 0:
                if chunk == 0:
                    left_sum = 0.0
                    for p in tl.static_range(BLOCK_PH):
                        oh = oh_lo + p
                        if oh <= oh_hi:
                            left_sum += tl.sum(
                                tl.load(
                                    go_nc + oh * GO_ROW + pw,
                                    mask=pw < PL,
                                    other=0.0,
                                ).to(tl.float32)
                            )
                    total += tl.where(iw0 == 0, left_sum, 0.0)
            if PR > 0:
                if chunk == NCHUNK - 1:
                    right_sum = 0.0
                    for p in tl.static_range(BLOCK_PH):
                        oh = oh_lo + p
                        if oh <= oh_hi:
                            right_sum += tl.sum(
                                tl.load(
                                    go_nc + oh * GO_ROW + (PL + W + pw),
                                    mask=pw < PR,
                                    other=0.0,
                                ).to(tl.float32)
                            )
                    total += tl.where(iw == W - 1, right_sum, 0.0)
            tl.store(gi_row + iw, total.to(OUT_DTYPE), mask=iwv)


def _dtype_constexpr(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    return tl.float32


def _next_pow2(n):
    value = 1
    while value < n:
        value *= 2
    return value


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.view(
            1, C, grad_output.shape[-2], grad_output.shape[-1]
        )
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError(
            "replication_pad2d_backward expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    expected_grad_shape = (N, C, OH, OW)
    if tuple(grad_output_4d.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output_4d.shape)} does not match "
            f"expected shape {expected_grad_shape}"
        )

    # Without padding, backward is an identity copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    rows = N * C * H
    if rows == 0:
        if out is not None:
            return out
        return torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)

    # Kernel writes the contiguous buffer directly with constant strides; non-contiguous out is computed into a
    # contiguous buffer then copied back
    write_back = None
    if out is None:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
    elif out.is_contiguous():
        target = out
    else:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
        write_back = out

    block_ph = _next_pow2(1 + pad_top + pad_bottom)
    block_w = min(_next_pow2(W), max(16, ELEM_CAP // block_ph))
    nchunk = triton.cdiv(W, block_w)
    block_pad = _next_pow2(max(pad_left, pad_right, 2))

    grid_rows = min(rows, SPC * GRID_MULT, MAX_GRID)
    with torch_device_fn.device(self.device):
        replication_pad2d_backward_kernel[(grid_rows,)](
            grad_output_4d,
            target,
            rows,
            H,
            W,
            OH,
            OW,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            OW,
            OH * OW,
            W,
            H * W,
            BLOCK_W=block_w,
            BLOCK_PH=block_ph,
            BLOCK_PAD=block_pad,
            NCHUNK=nchunk,
            OUT_DTYPE=_dtype_constexpr(self.dtype),
            num_warps=WARPS,
        )

    result = target.view(C, H, W) if is_3d else target
    if write_back is not None:
        write_back.copy_(result)
        return write_back
    return result


def replication_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding)


def replication_pad2d_backward_grad_input(grad_output, self, padding, *, grad_input):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
