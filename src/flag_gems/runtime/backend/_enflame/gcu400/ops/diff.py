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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# GCU400 tile sweep (2026-11-13, per-launch event median, fp16 large shape):
# bandwidth rises monotonically with BLOCK_M*BLOCK_DIFF total tile elements; 128K elements (DSM 768KB < 896KB
# hard limit) reaches 2.05 TB/s (32K tile only 1.42 TB/s). The optimum is BLOCK_DIFF filling the row width and
# BLOCK_M filling the tile budget: long rows (N>=4096) use 32x4096, medium rows (N~1024) use 128x1024,
# short rows (N~256) use 512x256. num_warps is insensitive (<1.5%). DSM = tile*itemsize*3,
# so a 128K tile exceeds 896KB for fp32/int64 and is skipped by OutOfResources, with a 64K tier as fallback;
# 32x16 is the fallback for extremely short rows.
_GCU400_DIFF_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 16}, num_warps=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_DIFF": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_DIFF": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_DIFF": 2048}, num_warps=4),
    triton.Config({"BLOCK_M": 8, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 4096}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_DIFF": 4096}, num_warps=8),
]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff_1d"),
    key=["N"],
)
@triton.jit
def diff_kernel_1d(in_ptr, out_ptr, N, BLOCK_DIFF: tl.constexpr):
    pid = tle.program_id(0)

    in_offsets = pid * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in = in_offsets < N - 1
    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + in_offsets, next_block - in_block, mask_in)


@libentry()
@libtuner(
    configs=_GCU400_DIFF_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def diff_kernel_2d(
    in_ptr,
    out_ptr,
    M,
    N,
    M_STRIDE_IN,
    M_STRIDE_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_DIFF: tl.constexpr,
):
    pid_M = tle.program_id(0)
    pid_diff = tle.program_id(1)

    M_offsets = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = M_offsets < M

    in_offsets_diff = pid_diff * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in_diff = in_offsets_diff < N - 1

    in_offsets = M_offsets[:, None] * M_STRIDE_IN + in_offsets_diff
    out_offsets = M_offsets[:, None] * M_STRIDE_OUT + in_offsets_diff
    mask_in = mask_M[:, None] & mask_in_diff

    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + out_offsets, next_block - in_block, mask_in)


# GCU400 flat-path configs (2026-11-16 sweep, do_bench median w1000/r100, L600 48SP):
# the flat kernel streams the whole (M, N) tensor as one contiguous 2R+1W flow and
# reaches 2.76-3.27 TB/s raw (fp16 (8,4096,4096) 277us, fp32 492us) vs 2.15 TB/s for
# the 2D tiled form above (375/746us); aten's binary_tensor_slice_kernel<SUB> sits at
# 315/631us. Winners by regime: fp16/bf16 large -> B=131072 nw2; fp32 large ->
# B=65536 nw2; long-row (1024,65536) -> B=65536 nw2 ns2; mid/small -> B=4096..16384
# nw4. OOR configs (B=131072 with fp32/int64 DSM) are skipped by the tuner.
_GCU400_DIFF_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 1024, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 2048, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 4096, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 8192, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 16384, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 32768, "NS": 1}, num_warps=2),
    triton.Config({"BLOCK": 32768, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 65536, "NS": 1}, num_warps=2),
    triton.Config({"BLOCK": 65536, "NS": 1}, num_warps=4),
    triton.Config({"BLOCK": 65536, "NS": 2}, num_warps=2),
    triton.Config({"BLOCK": 131072, "NS": 1}, num_warps=2),
    triton.Config({"BLOCK": 131072, "NS": 1}, num_warps=4),
]

_GCU400_FLAT_GRID_CAP = 48
# i32 safe domain with headroom for the loop-variable overshoot: the last
# failing tl.range condition check can reach TOTAL + 48*BLOCK (<= 2^23) and the
# final tile's arange can reach TOTAL + BLOCK - 2 before the mask clamps, so
# keep 2^23 elements of slack below 2^31 to rule out signed wraparound.
_FLAT_INT32_LIMIT = 0x7F7FFFFF


@libentry()
@libtuner(
    configs=_GCU400_DIFF_FLAT_CONFIGS,
    key=["TOTAL"],
)
@triton.jit
def diff_kernel_flat(in_ptr, out_ptr, TOTAL, BLOCK: tl.constexpr, NS: tl.constexpr):
    # Flat difference over the whole tensor: out[j] = in[j + 1] - in[j] for
    # j < TOTAL. Row-crossing lanes land in the caller's padded dummy column
    # and the loads never go past in[TOTAL] (TOTAL = numel - 1).
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    for j in tl.range(pid * BLOCK, TOTAL, nprog * BLOCK, num_stages=NS):
        offs = j + tl.arange(0, BLOCK)
        msk = offs < TOTAL
        cur = tl.load(in_ptr + offs, msk)
        nxt = tl.load(in_ptr + offs + 1, msk)
        tl.store(out_ptr + offs, nxt - cur, msk)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N
    total = M * N

    is_1d = len(shape) == 1

    # GCU400 flat fast path: stream the whole tensor as one contiguous flow
    # (probe 2026-11-16: 2.76-3.27 TB/s raw vs 2.15 TB/s for the 2D tiled form).
    # Multi-row outputs are allocated at the input's row width N and returned
    # as a [..., :N-n] view; the trailing dummy column absorbs the row-crossing
    # lanes so flat index j maps to the same (row, col) in src and dst.
    if not is_1d and total <= _FLAT_INT32_LIMIT:

        def _launch_flat(src, dst, total_out):
            if total_out <= 0:
                return
            grid = lambda meta: (
                min(_GCU400_FLAT_GRID_CAP, triton.cdiv(total_out, meta["BLOCK"])),
            )
            with torch_device_fn.device(src.device):
                diff_kernel_flat[grid](src, dst, total_out)

        if M == 1:
            # Single row: the flat layout is exact, so keep the real
            # (contiguous) post-diff shape [..., N-n] without padding.
            out_shape = list(input.shape)
            out_shape[-1] = N - n
            output = torch.empty(out_shape, device=input.device, dtype=input.dtype)
            if n == 1:
                _launch_flat(input, output, N - 1)
                return torch.moveaxis(output, -1, dim)
            scratch_a_shape = list(input.shape)
            scratch_a_shape[-1] = N - 1
            scratch_a = torch.empty(
                scratch_a_shape, device=input.device, dtype=input.dtype
            )
            if n >= 3:
                scratch_b_shape = list(input.shape)
                scratch_b_shape[-1] = N - 2
                scratch_b = torch.empty(
                    scratch_b_shape, device=input.device, dtype=input.dtype
                )
            _launch_flat(input, scratch_a, N - 1)
            src, src_width = scratch_a, N - 1
            for k in range(1, n):
                if k == n - 1:
                    dst = output
                elif k % 2 == 1:
                    dst = scratch_b
                else:
                    dst = scratch_a
                _launch_flat(src, dst, src_width - 1)
                src, src_width = dst, src_width - 1
            return torch.moveaxis(output, -1, dim)

        output = torch.empty_like(input)
        if n == 1:
            _launch_flat(input, output, total - 1)
            return torch.moveaxis(output[..., : N - 1], -1, dim)
        scratch_a = torch.empty_like(input)
        if n >= 3:
            scratch_b = torch.empty_like(input)
        _launch_flat(input, scratch_a, total - 1)
        src = scratch_a
        for k in range(1, n):
            if k == n - 1:
                dst = output
            elif k % 2 == 1:
                dst = scratch_b
            else:
                dst = scratch_a
            _launch_flat(src, dst, total - 1)
            src = dst
        return torch.moveaxis(output[..., : N - n], -1, dim)

    def _launch(src, dst, in_stride_m, out_stride_m, n_bound):
        if is_1d:
            grid = lambda meta: (triton.cdiv(n_bound - 1, meta["BLOCK_DIFF"]),)
            with torch_device_fn.device(src.device):
                diff_kernel_1d[grid](src, dst, n_bound)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(n_bound - 1, meta["BLOCK_DIFF"]),
            )
            with torch_device_fn.device(src.device):
                diff_kernel_2d[grid](src, dst, M, n_bound, in_stride_m, out_stride_m)

    # Allocate the final output at its exact post-diff size [..., N-n] so
    # that the last kernel writes directly into it. This eliminates the
    # tail `output[..., :N-n].contiguous()` copy.
    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N, N - 1, N)
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers (sized N-1 and N-2 for the diff dim),
    # writing the last iteration directly into `output` (size N-n).
    # Avoid _copy().
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    # iter 0: input -> scratch_a
    _launch(input, scratch_a, N, N - 1, N)
    src, src_stride = scratch_a, N - 1

    # iter 1 to (n - 1)
    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
