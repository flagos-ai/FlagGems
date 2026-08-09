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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# XPU codegen hints / caps (harness docs: tl.sum full up to 8192 lanes without
# buffer_size_limit and 32768 with it; masked tails are only reliable with
# mega tiles, so masked tiling pins TILE to TILE_CAP).
buf_len_per_core = 2048
TILE_CAP = 8192
BM_CAP = 64
PARTIAL_CAP = 256


def _word_parts(x):
    """Return (cnt_per_word, is_float) for an int32-word view, or (0, 0)."""
    es = x.element_size()
    if es not in (1, 2, 4) or (x.numel() * es) % 4 != 0:
        return 0, 0
    return 4 // es, int(x.dtype.is_floating_point)


@libentry()
@triton.jit
def count_nonzero_kernel_flat(
    inp,
    partial,
    N,  # units === elements when EPW == 0, int32 words when EPW > 0
    OFF: tl.constexpr,  # start offset in units (tail pass reuses the kernel)
    TILE: tl.constexpr,
    GRID: tl.constexpr,  # program count (power of two, <= PARTIAL_CAP)
    EPW: tl.constexpr,  # elements per int32 word: 0 = element mode, else 1/2/4
    EFZ: tl.constexpr,  # -0.0 bit-patterns (0x8000/0x80000000) count as zero
    NEED_MASK: tl.constexpr,
):
    # Grid-stride over TILE units. Masked lanes (tail pass only, GRID == 1)
    # use clamped offsets plus an ok-multiplier (masked `other` values are
    # not trusted on this backend).
    pid = ext.program_id(0)
    acc = tl.zeros([TILE], dtype=tl.int32)
    last = OFF + N - 1
    for off in range(pid * TILE, N, GRID * TILE):
        cols = OFF + off + tl.arange(0, TILE)
        if NEED_MASK:
            cclamp = tl.minimum(cols, last)
            ok = (cols < OFF + N).to(tl.int32)
            w = tl.load(inp + cclamp)
        else:
            w = tl.load(inp + cols)
            ok = 1
        if EPW == 0:
            cnt = (w != 0).to(tl.int32)
        elif EPW == 1:
            if EFZ:
                cnt = ((w != 0) & (w != 0x80000000)).to(tl.int32)
            else:
                cnt = (w != 0).to(tl.int32)
        elif EPW == 2:
            lo = w & 0xFFFF
            hi = (w >> 16) & 0xFFFF
            if EFZ:
                cnt = ((lo != 0) & (lo != 0x8000)).to(tl.int32) + (
                    (hi != 0) & (hi != 0x8000)
                ).to(tl.int32)
            else:
                cnt = (lo != 0).to(tl.int32) + (hi != 0).to(tl.int32)
        else:  # EPW == 4
            b0 = w & 0xFF
            b1 = (w >> 8) & 0xFF
            b2 = (w >> 16) & 0xFF
            b3 = (w >> 24) & 0xFF
            cnt = (
                (b0 != 0).to(tl.int32)
                + (b1 != 0).to(tl.int32)
                + (b2 != 0).to(tl.int32)
                + (b3 != 0).to(tl.int32)
            )
        acc += cnt * ok
    s = tl.sum(acc, axis=0).to(tl.int64)
    tl.store(partial + pid, s)


@libentry()
@triton.jit
def count_nonzero_reduce_partial(partial, out, BLOCK: tl.constexpr):
    # BLOCK == grid size (power of two <= PARTIAL_CAP); every slot is written
    # by the mask-free main kernel => no masked load is ever needed here.
    idx = tl.arange(0, BLOCK)
    v = tl.load(partial + idx)
    total = tl.sum(v, axis=0).to(tl.int64)
    tl.store(out, total)


@libentry()
@triton.jit
def count_nonzero_kernel_rows(
    inp,  # word pointer when EPW > 0, element pointer otherwise
    out,  # flat int64 output, one entry per row
    M,
    N,  # row length in units of `inp` (words or elements)
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPW: tl.constexpr,
    EFZ: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Fast rows: either TILE divides N (no mask), or TILE == TILE_CAP with
    # per-row tails handled by clamp + ok-multiplier (large-tile mask only).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rclamp = tl.minimum(rows, M - 1)
    base = rclamp * N
    acc = tl.zeros([BLOCK_M, TILE], dtype=tl.int32)
    last = N - 1
    for off in range(0, N, TILE):
        cols = off + tl.arange(0, TILE)
        if NEED_MASK:
            cclamp = tl.minimum(base[:, None] + cols[None, :], base[:, None] + last)
            ok = (cols < N).to(tl.int32)
            w = tl.load(inp + cclamp)
        else:
            w = tl.load(inp + base[:, None] + cols[None, :])
            ok = 1
        if EPW == 0:
            cnt = (w != 0).to(tl.int32)
        elif EPW == 1:
            if EFZ:
                cnt = ((w != 0) & (w != 0x80000000)).to(tl.int32)
            else:
                cnt = (w != 0).to(tl.int32)
        elif EPW == 2:
            lo = w & 0xFFFF
            hi = (w >> 16) & 0xFFFF
            if EFZ:
                cnt = ((lo != 0) & (lo != 0x8000)).to(tl.int32) + (
                    (hi != 0) & (hi != 0x8000)
                ).to(tl.int32)
            else:
                cnt = (lo != 0).to(tl.int32) + (hi != 0).to(tl.int32)
        else:  # EPW == 4
            b0 = w & 0xFF
            b1 = (w >> 8) & 0xFF
            b2 = (w >> 16) & 0xFF
            b3 = (w >> 24) & 0xFF
            cnt = (
                (b0 != 0).to(tl.int32)
                + (b1 != 0).to(tl.int32)
                + (b2 != 0).to(tl.int32)
                + (b3 != 0).to(tl.int32)
            )
        acc += cnt * ok
    s = tl.sum(acc, axis=1).to(tl.int64)
    tl.store(out + rclamp, s, mask=rows < M)


@libentry()
@triton.jit
def count_nonzero_scalar_rows(inp, out, M, N, BLOCK_M: tl.constexpr):
    # Rows with odd/non-pow2 short lengths: a per-row scalar loop, no masks
    # at all (small-tile masking and masked `other` values are unreliable on
    # this backend). Only reached for N small (< TILE_CAP).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rclamp = tl.minimum(rows, M - 1)
    acc = tl.zeros([BLOCK_M], dtype=tl.int32)
    for c in range(0, N):
        a = tl.load(inp + rclamp * N + c)
        acc += (a != 0).to(tl.int32)
    tl.store(out + rclamp, acc.to(tl.int64), mask=rows < M)


def _count_flat(x, data, epw, efz):
    """dim=None: whole-tensor count via a two-stage reduction.

    The masked-tail loop is ~25-30x slower than the mask-free loop on this
    backend, so the scan is split: a mask-free grid-stride pass over all
    complete tiles plus a single masked program for the remainder(s); both
    counts are reduced independently and added (each reduce reads only slots
    its own pass wrote).
    """
    numel = x.numel()
    if numel == 0:
        return torch.zeros((), dtype=torch.int64, device=x.device)
    n = numel if epw == 0 else numel // epw
    tiles = n // TILE_CAP
    grid = 1
    want = max(1, triton.cdiv(max(1, tiles), 8))
    while grid < min(PARTIAL_CAP, want):
        grid *= 2
    tiles_main = (tiles // grid) * grid
    n_main = tiles_main * TILE_CAP
    n_tail = n - n_main
    out = torch.empty((), dtype=torch.int64, device=x.device)
    with torch_device_fn.device(x.device):
        if n_main:
            partial = torch.empty(grid, dtype=torch.int64, device=x.device)
            count_nonzero_kernel_flat[(grid,)](
                data,
                partial,
                n_main,
                0,
                TILE=TILE_CAP,
                GRID=grid,
                EPW=epw,
                EFZ=efz,
                NEED_MASK=False,
            )
            count_nonzero_reduce_partial[(1,)](
                partial, out, BLOCK=grid
            )
        if n_tail:
            # remainder (< grid*TILE units): a single masked program
            tail = torch.empty(1, dtype=torch.int64, device=x.device)
            count_nonzero_kernel_flat[(1,)](
                data,
                tail,
                n_tail,
                n_main,
                TILE=TILE_CAP,
                GRID=1,
                EPW=epw,
                EFZ=efz,
                NEED_MASK=True,
            )
            if n_main:
                out2 = torch.empty((), dtype=torch.int64, device=x.device)
                count_nonzero_reduce_partial[(1,)](
                    tail, out2, BLOCK=1
                )
                out = out + out2
            else:
                count_nonzero_reduce_partial[(1,)](
                    tail, out, BLOCK=1
                )
    return out


def _count_rows_fast(data, out, M, n, tile, blk_m, epw, efz):
    grid = (triton.cdiv(M, blk_m),)
    with torch_device_fn.device(data.device):
        count_nonzero_kernel_rows[grid](
            data,
            out,
            M,
            n,
            TILE=tile,
            BLOCK_M=blk_m,
            EPW=epw,
            EFZ=efz,
            NEED_MASK=n % tile != 0,
            buffer_size_limit=buf_len_per_core,
        )
    return out


def count_nonzero(x, dim=None):
    logger.debug("GEMS_KUNLUNXIN COUNT_NONZERO")

    if dim is not None:
        shape = x.shape
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        if dim < 0:
            dim += x.ndim
        out_shape = list(shape)
        del out_shape[dim]
        N = shape[dim]
        if x.numel() == 0:
            return torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
        if dim == x.ndim - 1:
            xf = x if x.is_contiguous() else x.contiguous()
        elif x.ndim == 2 and dim == 0:
            # transpose (view) + native strided copy, then count rows.
            xt = x.transpose(0, 1)
            xf = torch.empty(
                (x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device
            )
            torch.ops.aten._copy_from(xt, xf, False)
            N = x.shape[0]
        else:
            xf = dim_compress(x, dim).reshape(-1, N)
        M = xf.numel() // N
        if M == 0 or N == 0:
            return torch.empty(out_shape, dtype=torch.int64, device=x.device)
        epw, efz = _word_parts(xf)
        if epw and N % epw == 0:
            data = xf.reshape(-1).view(torch.uint8).view(torch.int32)
            n = N // epw
        else:
            data, epw, n = xf, 0, N
        tile = min(TILE_CAP, max(1, triton.next_power_of_2(n)))
        blk_m = min(BM_CAP, max(1, 8192 // tile))
        if n % tile == 0:
            return _count_rows_fast(data, out, M, n, tile, blk_m, epw, efz)
        if tile == TILE_CAP:
            # masked large-tile rows (proven safe on this backend)
            return _count_rows_fast(data, out, M, n, TILE_CAP, 1, epw, efz)
        # small odd-N rows: scalar per-row loop (no masks anywhere)
        blk_l = 128
        with torch_device_fn.device(xf.device):
            count_nonzero_scalar_rows[(triton.cdiv(M, blk_l),)](
                xf, out, M, N, BLOCK_M=blk_l,
            )
        return out
    else:
        xc = x.contiguous().reshape(-1)
        epw, efz = _word_parts(xc)
        if epw:
            data = xc.contiguous().view(torch.uint8).view(torch.int32)
        else:
            data = xc
        return _count_flat(xc, data, epw, efz)