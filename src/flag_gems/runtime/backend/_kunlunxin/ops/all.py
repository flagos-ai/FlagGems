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

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16


# Tile budget = the current max tile (BLOCK_M=64 * BLOCK_N=512). We keep this
# constant so the [BLOCK_M, BLOCK_N] tile never grows past the size that already
# compiles cleanly (no XPU struct explosion), we only RESHAPE it.
TILE_BUDGET = 64 * 512


# Large flat torch.all(inp) reductions reshape to a [M, K] grid and reduce via the
# wider all_kernel_dim (axis=1) path, which is ~1.75x the flat all_kernel_1 at 1G.
# The reshape wins only above ~8M elements (below that the extra launch dominates).
_GLOBAL_2D_MIN = 1 << 23


def _pick_2d_cols(n):
    # Largest power-of-2 column width in [8192, 65536] that divides n, so the flat
    # buffer can be view()'d as a dense [n // K, K] grid with no copy. 0 = no clean fit.
    for K in (65536, 32768, 16384, 8192):
        if n % K == 0:
            return K
    return 0


def _heur_n_raw(N):
    # For N <= 8192 keep the historical cap of 512 (square / small-N shapes are
    # already near the reduce-bandwidth ceiling with BLOCK_M=64, BLOCK_N=512).
    # For very wide N, a 512-wide tile forces N/512 serial chunks (e.g. 128 for
    # N=65536); widening BLOCK_N to 4096 cuts the loop count ~8x. Measured on XPU
    # (proto): [1024,65536] 113 -> 165 GB/s (+46%) at the SAME tile budget.
    if N <= 8192:
        block_n = min(N, 512)
    else:
        block_n = min(triton.next_power_of_2(N), 4096)
    return triton.next_power_of_2(max(block_n, 1))


def heur_m_block_size(args):
    M = args["M"]
    block_n = _heur_n_raw(args["N"])
    # For very small M, use minimum BLOCK_M of 1
    block_m = min(triton.cdiv(M, cluster_num), core_num)
    # Keep BLOCK_M * BLOCK_N <= TILE_BUDGET: if BLOCK_N was widened for large N,
    # shrink BLOCK_M so the tile stays the same size (constant compile footprint).
    block_m = min(block_m, max(TILE_BUDGET // block_n, 1))
    return triton.next_power_of_2(max(block_m, 1))


def heur_n_block_size(args):
    return _heur_n_raw(args["N"])


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1 of the global-all reduction: each program reduces one
    BLOCK_SIZE-sized chunk of the flattened input into a single bool in `mid`.
    Splitting the work across `cdiv(n_elements, BLOCK_SIZE)` programs restores
    parallelism (the old single-program loop ran at ~7 GB/s on one core)."""
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp + offset, mask=mask, other=1)
    # masked-out lanes must be True (identity for AND); do not rely on `other`.
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


@libentry()
@triton.jit
def all_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    """Stage 2: a single program reduces the per-chunk bools from stage 1."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


@libentry()
@triton.jit
def all_kernel_dim_split(
    inp,
    mid,
    M,
    N,
    SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Column-split stage 1: program (pid_m, pid_s) AND-reduces the N-chunks
    # strided by SPLIT*BLOCK_N for its BLOCK_M rows into one partial bool per
    # row, scattered to mid[row, pid_s]. This restores N-parallelism that
    # all_kernel_dim serializes inside every program's chunk loop.
    pid_m = ext.program_id(0)
    pid_s = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    inp = inp + rows * N

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(pid_s * BLOCK_N, N, SPLIT * BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        a = tl.load(inp + cols, row_mask and col_mask, other=1.0)
        _all = _all and (a != 0)
    partial = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(mid + rows * SPLIT + pid_s, partial[:, None], row_mask)


@libentry()
@triton.jit
def all_kernel_dim_split_pair(
    inp,
    mid,
    M,
    NP,
    SPLIT,
    HBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Packed column-split stage 1: program (pid_m, pid_s) AND-reduces the
    # NP-chunks strided by SPLIT*BLOCK_N of the packed integer view for its
    # BLOCK_M rows into one partial bool per row. Halves/quarters the element
    # count vs all_kernel_dim_split and replaces the per-element float compare
    # with one SWAR op (2x-4x measured on deep-loop shapes). Callers must
    # guarantee M % BLOCK_M == 0 and NP % BLOCK_N == 0. Literal masks/shifts
    # per HBITS branch (see all_kernel_dim_pair note).
    pid_m = ext.program_id(0)
    pid_s = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * NP

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(pid_s * BLOCK_N, NP, SPLIT * BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols)
        if HBITS == 15:
            z = (a & 0x7FFF7FFF) + 0x7FFF7FFF
            pair_nz = ((z >> 15) & 0x00010001) == 0x00010001
        else:
            z = (a & 0x7FFFFFFF7FFFFFFF) + 0x7FFFFFFF7FFFFFFF
            pair_nz = ((z >> 31) & 0x0000000100000001) == 0x0000000100000001
        _all = _all and pair_nz
    partial = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(mid + rows * SPLIT + pid_s, partial[:, None])


@libentry()
@triton.jit
def all_kernel_dim_merge(
    mid,
    out,
    M,
    SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    # Stage 2: AND the SPLIT partial bools per row. mid rows are contiguous
    # ([M, SPLIT] layout), so each program loads a [BLOCK_M, BLOCK_MID] tile.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_MID)[None, :]
    col_mask = cols < SPLIT
    val = tl.load(mid + rows * SPLIT + cols, row_mask and col_mask, other=1)
    result = tl.reduce(val != 0, axis=1, combine_fn=reduce_all)
    tl.store(out + rows, result[:, None], row_mask)


# dtypes eligible for the packed pair-nonzero fast path: (packed dtype,
# half bits, elements per packed element).
_PACK_DTYPES = {
    torch.float16: (torch.int32, 15, 2),
    torch.bfloat16: (torch.int32, 15, 2),
    torch.float32: (torch.int64, 31, 2),
}


@libentry()
@triton.jit
def all_kernel_dim_pair(
    inp,
    out,
    M,
    NP,
    HBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Mask-free axis=1 reduction over a packed integer view (2 original values
    # per element). SWAR per element: the top bit of each half is set iff that
    # half != 0 (correct for -0.0 and NaN); pair passes iff both bits are set.
    # Callers must guarantee M % BLOCK_M == 0 and NP % BLOCK_N == 0.
    # NOTE: masks/shift amounts must be LITERAL constants per branch — the XPU
    # backend miscompiles int64 mask/shift ops whose operands are derived from
    # constexpr arithmetic (ops silently become identity).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * NP
    out = out + rows
    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, NP, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols)
        if HBITS == 15:
            z = (a & 0x7FFF7FFF) + 0x7FFF7FFF
            pair_nz = ((z >> 15) & 0x00010001) == 0x00010001
        else:
            z = (a & 0x7FFFFFFF7FFFFFFF) + 0x7FFFFFFF7FFFFFFF
            pair_nz = ((z >> 31) & 0x0000000100000001) == 0x0000000100000001
        _all = _all and pair_nz
    all_ = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all_[:, None])


@libentry()
@triton.jit
def all_kernel_1_pair(
    inp,
    mid,
    n_packed,
    HBITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Flat packed stage 1 (mask-free): each program reduces one BLOCK_SIZE
    # chunk of the packed view into a single bool. Callers must guarantee
    # n_packed % BLOCK_SIZE == 0. Literal masks/shifts per HBITS branch (see
    # all_kernel_dim_pair note).
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(inp + offset)
    if HBITS == 15:
        z = (a & 0x7FFF7FFF) + 0x7FFF7FFF
        pair_nz = ((z >> 15) & 0x00010001) == 0x00010001
    else:
        z = (a & 0x7FFFFFFF7FFFFFFF) + 0x7FFFFFFF7FFFFFFF
        pair_nz = ((z >> 31) & 0x0000000100000001) == 0x0000000100000001
    result = tl.reduce(pair_nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, result)


def _pack_view(inp):
    """(packed view, half bits, elements per packed element) or None."""
    pack_dt, hbits, ratio = _PACK_DTYPES[inp.dtype]
    if inp.numel() % ratio != 0:
        return None
    if (inp.storage_offset() * inp.element_size()) % (ratio * inp.element_size()) != 0:
        return None
    return inp.reshape(-1).view(pack_dt), hbits, ratio


def _all_packed_flat(inp):
    """Packed flat/2D reduction mirroring all()'s two paths; None = not eligible.
    All eligibility checks are pure integer math; tensor views are created only
    when the packed launch is certain (host overhead matters for small ops)."""
    pack_dt, hbits, ratio = _PACK_DTYPES[inp.dtype]
    n = inp.numel()
    if n % ratio != 0:
        return None
    if (inp.storage_offset() * inp.element_size()) % (ratio * inp.element_size()) != 0:
        return None
    n_p = n // ratio
    packed_elem_size = ratio * inp.element_size()
    with torch_device_fn.device(inp.device):
        if n_p >= _GLOBAL_2D_MIN:
            k2 = _pick_2d_cols(n_p)
            if k2:
                m2 = n_p // k2
                block_n = _heur_n_raw(k2)
                block_m = heur_m_block_size({"M": m2, "N": k2})
                if m2 % block_m == 0 and k2 % block_n == 0:
                    packed = inp.reshape(-1).view(pack_dt)
                    mid = torch.empty((m2,), dtype=torch.bool, device=inp.device)
                    all_kernel_dim_pair[(triton.cdiv(m2, block_m),)](
                        packed.view(m2, k2),
                        mid,
                        m2,
                        k2,
                        hbits,
                        block_m,
                        block_n,
                        buffer_size_limit=2048,
                    )
                    block_mid = triton.next_power_of_2(m2)
                    out = torch.empty([], dtype=torch.bool, device=inp.device)
                    all_kernel_2[(1, 1, 1)](mid, out, m2, block_mid, buffer_size_limit=2048)
                    return out
        block_size = get_block_size_1d(n_p, packed_elem_size)
        if block_size > 0 and n_p % block_size == 0:
            packed = inp.reshape(-1).view(pack_dt)
            mid_size = n_p // block_size
            mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
            all_kernel_1_pair[(mid_size, 1, 1)](
                packed, mid, n_p, hbits, block_size, buffer_size_limit=2048
            )
            if mid_size == 1:
                return mid.reshape([])
            block_mid = triton.next_power_of_2(mid_size)
            out = torch.empty([], dtype=torch.bool, device=inp.device)
            all_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
            return out
    return None


def _all_dims_packed_dim(inp, out, M, N):
    """Mask-free packed axis=1 reduction for the [M, N] contiguous buffer.
    Writes `out` (flat M bools) and returns True; False = not eligible.
    Integer-math checks only until the launch is certain (no wasted tensor
    views on the launch-bound small shapes)."""
    if inp.dtype not in _PACK_DTYPES or M * N < (1 << 20):
        return False
    _, _, ratio = _PACK_DTYPES[inp.dtype]
    if N % ratio != 0:
        return False
    n_packed = N // ratio
    block_n = _heur_n_raw(n_packed)
    block_m = heur_m_block_size({"M": M, "N": n_packed})
    if M % block_m != 0 or n_packed % block_n != 0:
        return False
    packed_info = _pack_view(inp)
    if packed_info is None:
        return False
    packed, hbits, _ = packed_info
    all_kernel_dim_pair[(triton.cdiv(M, block_m),)](
        packed,
        out,
        M,
        n_packed,
        hbits,
        block_m,
        block_n,
        buffer_size_limit=2048,
    )
    return True


def _all_dims_packed_split(inp, out, M, N):
    """Packed column-split two-stage reduction (stage 1 = all_kernel_dim_split_pair
    on the packed integer view, stage 2 = all_kernel_dim_merge). Writes `out`
    (flat M bools) and returns True; False = not eligible. Only invoked on the
    deep-loop use_split path, so no host-overhead risk for launch-bound shapes."""
    if inp.dtype not in _PACK_DTYPES:
        return False
    _, _, ratio = _PACK_DTYPES[inp.dtype]
    if N % ratio != 0:
        return False
    n_packed = N // ratio
    block_n = _heur_n_raw(n_packed)
    if n_packed < block_n or n_packed % block_n != 0:
        return False
    block_m = heur_m_block_size({"M": M, "N": n_packed})
    # Shrink BLOCK_M to a divisor of M: the packed split kernel is mask-free.
    while block_m > 1 and M % block_m != 0:
        block_m //= 2
    if M % block_m != 0:
        return False
    grid_m = M // block_m
    packed_chunks = n_packed // block_n
    split = min(packed_chunks, max(1, (cluster_num * core_num) // grid_m))
    if split <= 1:
        return False
    packed_info = _pack_view(inp)
    if packed_info is None:
        return False
    packed, hbits, _ = packed_info
    mid = torch.empty((M, split), dtype=torch.bool, device=inp.device)
    block_m2 = triton.next_power_of_2(
        max(min(triton.cdiv(M, cluster_num), core_num), 1)
    )
    all_kernel_dim_split_pair[(grid_m, split)](
        packed,
        mid,
        M,
        n_packed,
        split,
        hbits,
        block_m,
        block_n,
        buffer_size_limit=2048,
    )
    all_kernel_dim_merge[(triton.cdiv(M, block_m2),)](
        mid,
        out,
        M,
        split,
        block_m2,
        triton.next_power_of_2(split),
        buffer_size_limit=2048,
    )
    return True


def all(inp):
    logger.debug("GEMS_KUNLUNXIN ALL")
    n_elements = inp.numel()

    # Packed pair-nonzero fast path for 2-byte/4-byte float dtypes. The XPU is
    # element-op bound in this reduction family (~17-27 G elem/s for fp16/bf16
    # compares vs ~59 G/s for int32), so viewing the buffer as int32 (fp16/bf16)
    # or int64 (fp32) halves/quarters the element count. Per pair, SWAR detects
    # "both halves nonzero" exactly (incl. -0.0 -> zero, NaN -> nonzero):
    #   z = (a & low_mask_pair) + low_mask_pair;  top bit of each half set iff
    #   that half != 0. Falls back to the generic paths when the buffer cannot
    # be viewed/packed or the tiles do not divide evenly (mask-free loads).
    if inp.dtype in _PACK_DTYPES and inp.is_contiguous() and n_elements >= (1 << 16):
        out_packed = _all_packed_flat(inp)
        if out_packed is not None:
            return out_packed

    # Fast path for large flat reductions. The 1D all_kernel_1 (flat BLOCK_SIZE tile +
    # axis=0 reduce) tops out ~115-230 GB/s on XPU. Viewing the contiguous buffer as a
    # [M, K] grid and reducing along axis=1 via all_kernel_dim coalesces far better
    # (measured ~1.75x at 1G, crossover ~8M elements). Falls back to the flat path when
    # the buffer is small, non-contiguous, or has no clean power-of-2 column width.
    if n_elements >= _GLOBAL_2D_MIN and inp.is_contiguous():
        K = _pick_2d_cols(n_elements)
        if K:
            M = n_elements // K
            inp2d = inp.view(M, K)
            mid = torch.empty((M,), dtype=torch.bool, device=inp.device)
            out = torch.empty([], dtype=torch.bool, device=inp.device)
            block_mid = triton.next_power_of_2(M)
            grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
            with torch_device_fn.device(inp.device):
                all_kernel_dim[grid](inp2d, mid, M, K, buffer_size_limit=2048)
                all_kernel_2[(1, 1, 1)](mid, out, M, block_mid, buffer_size_limit=2048)
            return out

    block_size = get_block_size_1d(n_elements, inp.element_size())
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1, 1)](
            inp, mid, n_elements, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        all_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    orig_ndim = inp.ndim

    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * orig_ndim)
        return out

    assert dim >= -orig_ndim and dim < orig_ndim, "Invalid dim"
    dim = dim % orig_ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    shape[dim] = 1
    M = inp.numel() // N

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim and out.ndim > 0:
        out = out.squeeze(dim) if dim < out.ndim else out
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    # Full-reduction fast path (all non-reduced dims collapse to M == 1): the
    # generic all_kernel_dim call below would launch a single program that
    # serializes the whole reduced volume inside one CTA (e.g. 342ms for a
    # 655M-element fp16 tensor). dim_compress() returned a contiguous tensor,
    # so delegate to the tuned flat `all()` two-stage split reduction and just
    # reshape the scalar to the torch.all(dim=list) output shape. Small
    # reductions stay on the single-kernel path (no extra merge launch).
    if M == 1 and N >= 65536:
        out = all(inp)
        if keepdim:
            return out.reshape(shape)
        return out.reshape([shape[i] for i in range(orig_ndim) if i not in dim])

    if inp.dtype != torch.bool and M * N <= 64:
        inp = inp != 0

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    block_n = _heur_n_raw(N)
    block_m = heur_m_block_size({"M": M, "N": N})
    grid_m = max(triton.cdiv(M, block_m), 1)
    total_chunks = triton.cdiv(N, block_n)
    split = min(total_chunks, max(1, (cluster_num * core_num) // grid_m))
    # Only split when the per-program chunk loop is long enough to pay for the
    # extra merge launch: short chunk loops (e.g. N=32768 -> 8 chunks) measured
    # SLOWER with the two-stage path ([64,512,512] 0.83 -> 0.98ms), while deep
    # loops (N=6.55M -> 1600 chunks, split=59) gain ~37% ([100,65536,100]).
    use_split = (
        M > 1 and total_chunks >= 32 and M * N >= (1 << 20) and split > 1
    )

    with torch_device_fn.device(inp.device):
        if use_split and _all_dims_packed_split(inp, out, M, N):
            # Packed column-split two-stage reduction ran both stages.
            pass
        elif use_split:
            # Column-split two-stage reduction: stage 1 spreads the N-chunk
            # loop over `split` programs per row-block (restoring the
            # parallelism the single-program chunk loop serializes), stage 2
            # AND-merges the per-row partials.
            mid = torch.empty((M, split), dtype=torch.bool, device=inp.device)
            block_m2 = triton.next_power_of_2(
                max(min(triton.cdiv(M, cluster_num), core_num), 1)
            )
            all_kernel_dim_split[(grid_m, split)](
                inp, mid, M, N, split, block_m, block_n, buffer_size_limit=2048
            )
            all_kernel_dim_merge[(triton.cdiv(M, block_m2),)](
                mid,
                out,
                M,
                split,
                block_m2,
                triton.next_power_of_2(split),
                buffer_size_limit=2048,
            )
        elif not _all_dims_packed_dim(inp, out, M, N):
            grid = lambda meta: (max(triton.cdiv(M, meta["BLOCK_M"]), 1),)
            all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim:
        # Squeeze reduced axes from highest to lowest. Removing a low axis first
        # shifts the positions of the remaining (still size-1) reduced axes, so a
        # later `squeeze(dim=d)` would target the wrong axis and silently leave a
        # leading size-1 dim (e.g. dim=[1,0] on (7,4,11,1) gave [1,11,1] vs [11,1]).
        for d in sorted(dim, reverse=True):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
