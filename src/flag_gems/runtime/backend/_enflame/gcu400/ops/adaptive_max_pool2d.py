# Copyright 2026, The FlagOS Contributors.
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
"""gcu400 adaptive_max_pool2d forward specialisation.

Differences from the generic ``flag_gems.ops.adaptive_max_pool2d``:

All argmax/index bookkeeping is staged through **int32** tensors (same
approach as the gcu400 ``adaptive_max_pool3d``).  On this torch_gcu build,
without ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` a Triton kernel cannot
store values into an int64 tensor parameter: the runtime silently narrows
the long storage to 4 bytes per element, so the 8-byte strides the kernel
computes land on the wrong bytes — the indices come out interleaved
garbage *and* the high words are written past the end of the buffer
(>2 GiB-corruption window that can even fault the device).  The kernels
therefore write int32 staging tensors and the host converts the final
index tensor to the API-required ``torch.int64`` with a plain aten cast
before returning.

Launch structure: 1D flattened grid-stride over the flat output index
(gcu400 caps grid.x at 65535) — each program scans OUT_PER_BLOCK output
elements' windows directly, replacing the generic 2D (N*C,
BLOCK_HxBLOCK_W-tile) grid whose serial int64 where-chain over
(MAX_WIN_H x MAX_WIN_W) masked iterations dominated the runtime.

Divisible fast path: when ``in_h % out_h == 0`` and ``in_w % out_w == 0``
every window is exactly KH x KW, so the window masks disappear entirely
(only the tail-of-grid lane mask remains).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# gcu400 hardware launch-grid cap along x.
_MAX_GRID_X = 65535


def _capped_grid_1d(num_blocks: int) -> int:
    return min(num_blocks, _MAX_GRID_X)


def _compute_max_win(in_size: int, out_size: int) -> int:
    return (in_size + out_size - 1) // out_size + 1


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("adaptive_max_pool2d"),
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def _adaptive_max_pool2d_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    total_outputs,
    in_h,
    in_w,
    out_h,
    out_w,
    MAX_WIN_H,
    MAX_WIN_W,
    OUT_PER_BLOCK: tl.constexpr,
):
    """Direct scan: each lane owns one output element and scans its window.

    gcu400 caps the hardware grid at 65535 along x, so the kernel runs as a
    grid-stride loop: the launch grid is clamped and each program strides
    over the remaining output blocks."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(total_outputs, OUT_PER_BLOCK)
    tid = tl.arange(0, OUT_PER_BLOCK)

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)

    for block_id in range(pid, num_blocks, num_programs):
        flat_idx = block_id * OUT_PER_BLOCK + tid
        valid = flat_idx < total_outputs
        safe = tl.where(valid, flat_idx, 0)

        # Decode (nc, h_out, w_out) from the flat output index.  The output
        # is contiguous (N, C, out_h, out_w), so flat_idx is also the output
        # offset.
        w_out = safe % out_w
        tmp = safe // out_w
        h_out = tmp % out_h
        nc = tmp // out_h

        # Adaptive window bounds (PyTorch semantics):
        #   start = floor(i * in / out), end = ceil((i + 1) * in / out)
        h_start = h_out * in_h // out_h
        win_h = ((h_out + 1) * in_h + out_h - 1) // out_h - h_start
        w_start = w_out * in_w // out_w
        win_w = ((w_out + 1) * in_w + out_w - 1) // out_w - w_start

        # gcu400: widen the (n, c) slab base offset to int64 before the
        # multiply — a >2 GiB input's element offset overflows 32-bit
        # address math and faults the device otherwise.
        slab = in_h * in_w
        in_base = in_ptr + nc.to(tl.int64) * slab

        acc_val = tl.full((OUT_PER_BLOCK,), min_val, dtype=dtype)
        # gcu400: indices staged as int32 (see module docstring) — the flat
        # input offset inside one (n, c) slab always fits int32.
        acc_idx = tl.full((OUT_PER_BLOCK,), -1, dtype=tl.int32)

        for kh in range(MAX_WIN_H):
            h_raw = h_start + kh
            h_ok = (kh < win_h) & (h_raw < in_h) & valid
            h_s = tl.where(h_ok, h_raw, 0)
            h_off = h_s * in_w
            for kw in range(MAX_WIN_W):
                w_raw = w_start + kw
                w_ok = (kw < win_w) & (w_raw < in_w) & valid
                in_mask = h_ok & w_ok
                w_s = tl.where(w_ok, w_raw, 0)
                cv = tl.load(
                    in_base + h_off + w_s,
                    mask=in_mask,
                    other=min_val,
                    cache_modifier=".ca",
                )
                # ATen keeps the first equal maximum but the last
                # encountered NaN; ``acc_idx < 0`` makes the first in-window
                # element win even when it equals the dtype minimum.
                is_new = (cv > acc_val) | (cv != cv) | (acc_idx < 0)
                acc_val = tl.where(is_new, cv, acc_val)
                acc_idx = tl.where(is_new & in_mask, h_off + w_s, acc_idx)

        tl.store(out_ptr + flat_idx, acc_val, mask=valid)
        tl.store(idx_ptr + flat_idx, acc_idx, mask=valid)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("adaptive_max_pool2d"),
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def _adaptive_max_pool2d_div_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    total_outputs,
    in_h,
    in_w,
    out_h,
    out_w,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OUT_PER_BLOCK: tl.constexpr,
):
    """Divisible fast path: in_h % out_h == 0 and in_w % out_w == 0, so every
    window is exactly KH x KW and h_in/w_in are always in range for valid
    lanes — the per-iteration window masks disappear (only the tail-of-grid
    lane mask remains)."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(total_outputs, OUT_PER_BLOCK)
    tid = tl.arange(0, OUT_PER_BLOCK)

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)

    for block_id in range(pid, num_blocks, num_programs):
        flat_idx = block_id * OUT_PER_BLOCK + tid
        valid = flat_idx < total_outputs
        safe = tl.where(valid, flat_idx, 0)

        w_out = safe % out_w
        tmp = safe // out_w
        h_out = tmp % out_h
        nc = tmp // out_h

        slab = in_h * in_w
        in_base = in_ptr + nc.to(tl.int64) * slab

        acc_val = tl.full((OUT_PER_BLOCK,), min_val, dtype=dtype)
        acc_idx = tl.full((OUT_PER_BLOCK,), -1, dtype=tl.int32)

        h0 = h_out * KH
        w0 = w_out * KW
        for kh in range(KH):
            h_in = h0 + kh
            h_off = h_in * in_w
            for kw in range(KW):
                w_in = w0 + kw
                cv = tl.load(
                    in_base + h_off + w_in,
                    mask=valid,
                    other=min_val,
                    cache_modifier=".ca",
                )
                is_new = (cv > acc_val) | (cv != cv) | (acc_idx < 0)
                acc_val = tl.where(is_new, cv, acc_val)
                acc_idx = tl.where(is_new & valid, h_off + w_in, acc_idx)

        tl.store(out_ptr + flat_idx, acc_val, mask=valid)
        tl.store(idx_ptr + flat_idx, acc_idx, mask=valid)


def _parse_output_size(output_size):
    if isinstance(output_size, int):
        return output_size, output_size
    if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
        return output_size[0], output_size[1]
    raise ValueError(f"Invalid output_size: {output_size}")


def adaptive_max_pool2d(
    input: torch.Tensor,
    output_size,
):
    logger.debug("GEMS ENFLAME ADAPTIVE_MAX_POOL2D")
    input = input.contiguous()

    out_h, out_w = _parse_output_size(output_size)

    if input.dim() == 3:
        # Unbatched: (C, H, W) -> add batch dim
        input = input.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    in_n, in_c, in_h, in_w = input.shape

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    # gcu400: int32 staging tensor (see module docstring); converted to the
    # API-required int64 with a plain aten cast after the launch.
    indices_i32 = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int32
    )

    total_outputs = in_n * in_c * out_h * out_w
    if total_outputs == 0:
        indices = indices_i32.to(torch.int64)
        if squeeze_batch:
            return output.squeeze(0), indices.squeeze(0)
        return output, indices

    def grid(meta):
        return (_capped_grid_1d(triton.cdiv(total_outputs, meta["OUT_PER_BLOCK"])),)

    if in_h % out_h == 0 and in_w % out_w == 0:
        _adaptive_max_pool2d_div_kernel[grid](
            input,
            output,
            indices_i32,
            total_outputs,
            in_h,
            in_w,
            out_h,
            out_w,
            KH=in_h // out_h,
            KW=in_w // out_w,
        )
    else:
        _adaptive_max_pool2d_kernel[grid](
            input,
            output,
            indices_i32,
            total_outputs,
            in_h,
            in_w,
            out_h,
            out_w,
            MAX_WIN_H=_compute_max_win(in_h, out_h),
            MAX_WIN_W=_compute_max_win(in_w, out_w),
        )

    indices = indices_i32.to(torch.int64)
    if squeeze_batch:
        return output.squeeze(0), indices.squeeze(0)
    return output, indices
