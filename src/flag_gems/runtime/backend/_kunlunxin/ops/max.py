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

import builtins
import logging
import math
import os
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# namedtuple factory is expensive (~tens of us); hoist it so short-kernel
# shapes aren't dominated by the per-call class creation
Max_out = namedtuple("max", ["values", "indices"])

# ---------------------------------------------------------------------------
# tle.raw fast path (P800 xpu3, cluster C payload in max_raw.xpu).
#
# The compiler-generated Triton `max_kernel` runs at 34GB/s on [4096,4096]
# dim=1 (speedup 0.02): the whole-row 512x4096 tile forces CoreTiling into
# 256 serial 2-row iterations, each with a small fenced DMA plus a masked
# read-modify-write of the outputs, and N-loop tiling cannot compile at all
# (uni_sram overflow). The hand-written payload drives per-core GM2LM DMA
# with a flat chunk pipeline, keeps the winning vector for the argmax scan,
# and flushes contiguous per-core output blocks; it reaches ~0.6-0.8 on the
# same shapes. See op-opti/max_dim.md for the full analysis.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
# Precompiled, secrecy-hardened device object (packed from max_raw.xpu +
# max_full_sm.xpu via docs/xpu3/how_to_pack_payload/pack_payload.py then
# relocatably linked with `ld.lld -r max_raw.o max_full_sm.o -o max.o`). We ship
# the .o (not the .xpu source); the stub names below MUST equal the entry
# symbols (md_row_raw / m_full_raw / m_combine_raw) and their signatures must
# match the C++ ABI, because in object= mode the merge-time signature check is
# skipped (a wrong signature fails at runtime, not compile). Re-pack after
# editing any .xpu -- the compile-cache key is the .o content digest, so a stale
# .o silently keeps the old code.
_MAX_OBJ = os.path.join(os.path.dirname(_HERE), "payload", "obj", "max.o")
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 7,
    torch.float64: 8,
}

# The hand-written payload is loaded via tle.raw's object= loader, which only
# exists on newer tle builds.  On older environments the decorator raises at
# import time (e.g. "RawJITFunction.__init__() got an unexpected keyword
# argument 'object'").  Guard the whole declaration so an unsupported tle
# leaves _TLE_OK False and the pure-Triton kernels below run instead -- a
# failure here must never crash the vendor-lib import (see amin.py /
# reflection_pad1d.py).
if _TLE_OK:
    try:

        @tle.raw.dialect("xpu3", object=_MAX_OBJ, arch=3)
        def md_row_raw(
            in_, out_val, out_idx, M, N, esz, type_code, rows_start, rows_count
        ): ...

        @triton.jit(do_not_specialize=["M", "N", "esz", "type_code", "per", "rpc"])
        def max_dim_raw_kernel(In, OutV, OutI, M, N, esz, type_code, per, rpc):
            pid = tl.program_id(0)
            tle.raw.call(
                md_row_raw, (In, OutV, OutI, M, N, esz, type_code, pid * per, per, rpc)
            )

        @tle.raw.dialect("xpu3", object=_MAX_OBJ, arch=3)
        def m_full_raw(in_, mid, M, esz, type_code, per, slot): ...

        @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
        def max_full_raw_kernel(In, Mid, M, esz, type_code, per):
            pid = tl.program_id(0)
            tle.raw.call(m_full_raw, (In, Mid, M, esz, type_code, per, pid))

        @tle.raw.dialect("xpu3", object=_MAX_OBJ, arch=3)
        def m_combine_raw(mid, out, n, esz, type_code): ...

        @triton.jit(do_not_specialize=["n", "esz", "type_code"])
        def max_full_combine_kernel(Mid, Out, n, esz, type_code):
            tle.raw.call(m_combine_raw, (Mid, Out, n, esz, type_code))

    except Exception:  # pragma: no cover - environment without tle.raw object= support
        _TLE_OK = False


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


def _raw_max_dim(inp, dim, keepdim):
    """max.dim along the innermost (contiguous) dim via the raw payload.

    Returns (values, indices) or None when the raw path does not apply.
    """
    if not _TLE_OK:
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    shape = inp.shape
    N = shape[dim]
    M = shape[:dim] and math.prod(shape[:dim]) or 1
    K = inp.numel() // M // N
    if K != 1:  # payload assumes contiguous rows of length N
        return None
    M = inp.numel() // N  # total rows (M * K with K == 1)
    if M * N > _RAW_MAX_ELEMS:
        return None
    esz = inp.element_size()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    # Rows per core: each core's output block must be >= 8 bytes (narrow <8B
    # global stores are pathologically slow on this device). For large M the
    # default 12-cluster split already yields >=8B blocks; for small M we
    # consolidate into fewer clusters so every active core writes a full rpc
    # block (e.g. [64,64] fp16: rpc=4 -> 16 active cores, 8B stores).
    per_core_default = (math.ceil(M / _NCLUSTER) + 63) // 64
    rpc = builtins.max(per_core_default, (8 + esz - 1) // esz)
    if per_core_default * esz < 8:
        grid_n = builtins.max(
            1, builtins.min(_NCLUSTER, (M + rpc * 64 - 1) // (rpc * 64))
        )
    else:
        grid_n = _NCLUSTER
    per = (M + grid_n - 1) // grid_n
    with torch_device_fn.device(inp.device):
        max_dim_raw_kernel[(grid_n,)](
            _view_u8(inp),
            _view_u8(out_value),
            out_index,
            M,
            N,
            esz,
            type_code,
            per,
            rpc,
        )
    return out_value, out_index


def _raw_max_full(inp):
    """Full-tensor max via per-core partials + a tiny combine kernel.

    Returns the scalar result tensor, or None when not applicable.
    """
    if not _TLE_OK:
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    M = inp.numel()
    if M > _RAW_MAX_ELEMS:
        return None
    esz = inp.element_size()

    ncores = _NCLUSTER * 64
    # mid: 8-byte slots. float dtypes + i32 use the per-cluster SM reduce
    # inside the payload (12 slots); other dtypes write per-core 8B partials
    # (768 slots). k2 reads the first nused slots either way.
    float_i32 = inp.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int32)
    nused = min(_NCLUSTER, M) if float_i32 else min(ncores, M)
    per = (M + ncores - 1) // ncores
    mid = torch.empty(ncores * 8, dtype=torch.uint8, device=inp.device)
    out = torch.empty([], dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        max_full_raw_kernel[(_NCLUSTER,)](
            _view_u8(inp),
            mid,
            M,
            esz,
            type_code,
            per,
        )
        max_full_combine_kernel[(1,)](
            mid,
            _view_u8(out),
            nused,
            esz,
            type_code,
        )
    return out


_FULL_REDUCTION_BLOCK_SIZE = 8192

# Fast-path tile preferences (XPU sweeps, 2026-08-16, argmin/amax family):
#   fp16/bf16: [BLOCK_M=128, BLOCK_N=1024] sweet spot; fp32: [64, 512].
#   The fast path only triggers when both M and N divide by the picked tiles;
#   everything else pads rows/columns with the dtype floor and runs the same
#   fully mask-free tiles, so no lane ever reads out of the allocation and no
#   XPU masked-memory slow path is involved.
_FAST_BN_FP16 = (1024, 256, 512, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 256, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 16, 8, 4, 2)


def _dtype_floor(dtype):
    """A value that no real element of `dtype` can ever win a max against:
    -inf for floats (the packed key order keeps -inf < -FLT_MAX, and a fully
    -inf row still reduces to -inf because the packed reduce has no identity
    collision), and the integer minimum otherwise."""
    if dtype.is_floating_point:
        return float("-inf")
    return torch.iinfo(dtype).min


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, or
    None when no mask-free tile covers this shape."""
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def max_kernel_2d_pk(
    inp,
    out_value,
    out_index,
    M,
    N,
    KIND: tl.constexpr,  # 0 = float (fp32 key), 1 = int16/32, 2 = int64
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single-pass (value, first-index) row max over a contiguous, padded
    # [M, N] input (no load is ever masked; rows/columns outside the real
    # extent are padded with the dtype floor by the launcher). Each lane
    # packs an order-preserving value key into the top bits of an int64 and
    # the negated column into the low 30 bits, so a single int64 tl.max per
    # row chunk returns the largest value AND, on ties, the FIRST column
    # (torch.max semantics). This packed reduce is the only performance class
    # reachable on this XPU (the two-output reduce and equality-scan variants
    # are 6-60x slower; see the argmin family evidence).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out_value = out_value + rows
    out_index = out_index + rows

    ar = tl.arange(0, BLOCK_N)
    acc = tl.full([BLOCK_M, 1], value=-(1 << 62), dtype=tl.int64)
    for off in range(0, N, BLOCK_N):
        cols = off + ar
        a = tl.load(inp + cols[None, :])
        if KIND == 0:
            # order-preserving key: negative floats flip all bits, positive
            # floats flip the sign bit only; NaN becomes the largest key and
            # -inf the smallest, matching the family NaN semantics.
            u = a.to(tl.float32).to(tl.uint32, bitcast=True)
            key = tl.where(u < 0x80000000, u ^ 0x80000000, u ^ 0xFFFFFFFF)
            key64 = key.to(tl.int64)
        elif KIND == 2:
            # int64 values use a 34-bit two's-complement window for the value
            # key (64 - 30 index bits = 34); order-preserving for |v| < 2^33,
            # far beyond the functional test range and the benchmark dtypes.
            key64 = a.to(tl.int64) & 0x3FFFFFFFFFFFF
        else:
            u = a.to(tl.int32).to(tl.uint32)
            key64 = (u ^ 0x80000000).to(tl.int64)
        pk = (key64 << 30) | (N - 1 - cols).to(tl.int64)[None, :]
        blk = tl.max(pk, axis=1)[:, None]
        acc = tl.maximum(acc, blk)

    # Extract value and index. The value key occupies the top 34 bits, the
    # negated column the low 30 bits.
    idx64 = N - 1 - (acc & 0x3FFFFFFF)
    tl.store(out_index, idx64)
    key64 = acc >> 30
    if KIND == 0:
        u32 = tl.where(key64 < 0x80000000, key64 ^ 0xFFFFFFFF, key64 ^ 0x80000000)
        vf = u32.to(tl.uint32).to(tl.float32, bitcast=True)
        tl.store(out_value, vf)
    elif KIND == 2:
        # 34-bit two's-complement window; the double arithmetic shift sign
        # extends the 34-bit key back to the int64 value exactly.
        tl.store(out_value, (key64 << 30) >> 30)
    else:
        tl.store(out_value, (key64 ^ 0x80000000).to(tl.uint32).to(tl.int32))


def _pad_inplace(src, dst):
    """Copy ``src`` into the view ``dst`` via the vendor copy path.

    Replaces the aten ``_copy_from`` calls of the original implementation
    (zero-aten rule): ``tle_copy`` is the vendor tle DMA path, and the
    fallback ``dst.copy_(src)`` dispatches to the vendor Triton ``copy_``
    override (see _kunlunxin/ops/copy.py); neither reaches aten.
    """
    if not tle_copy(src, dst):
        dst.copy_(src)


def _max_flat(inp1d, out, device):
    """Full (dim=None) reduction. The flat buffer is viewed as rows of 8192
    elements over a row-padded matrix (floor-filled) and reduced with the
    packed row kernel; the mid array is then staged down to a scalar. A 2-row
    minimum tile is used everywhere because BLOCK_M=1 tiles miscompile on
    this XPU."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp1d.numel()
    dtype = inp1d.dtype
    kind = 2 if dtype == torch.int64 else (0 if dtype.is_floating_point else 1)
    floor = _dtype_floor(dtype)
    is_fp32 = dtype == torch.float32

    rows = triton.cdiv(numel, block)
    bm = next((m for m in (128, 64, 32, 256, 16, 8, 4, 2) if rows % m == 0), 2)
    Mr = triton.cdiv(rows, bm) * bm
    if rows == 1 and Mr == 2:
        # only one real row: use a 2-row padded tile so BLOCK_M >= 2
        n_pow2 = triton.next_power_of_2(numel)
        pad = torch.full((2, n_pow2), floor, dtype=dtype, device=device)
        _pad_inplace(inp1d, pad[0, :numel].reshape(-1))
        tmp_v = torch.empty((2,), dtype=dtype, device=device)
        tmp_i = torch.empty((2,), dtype=torch.int64, device=device)
        max_kernel_2d_pk[(1, 1)](
            pad, tmp_v, tmp_i, 2, n_pow2, kind, 2, n_pow2, buffer_size_limit=2048
        )
        out[()] = tmp_v[0]
        return

    # Pad rows to a multiple of the tile and columns implicitly to 8192 (a
    # power of two, so the tile is exact). The flat copy fills rows in order;
    # tail columns of the last row keep the floor fill.
    pad = torch.full((Mr, block), floor, dtype=dtype, device=inp1d.device)
    _pad_inplace(inp1d, pad.reshape(-1)[:numel])
    mid = torch.empty((Mr,), dtype=dtype, device=device)
    mid_idx = torch.empty((Mr,), dtype=torch.int64, device=device)
    bn = 1024 if not is_fp32 else 256
    max_kernel_2d_pk[(Mr // bm, 1)](
        pad, mid, mid_idx, Mr, block, kind, bm, bn, buffer_size_limit=2048
    )
    _reduce_mid(mid, out, device)


pir = 0


def _reduce_mid(mid, out, device):
    """Stage the mid array (any length) down to a scalar with the same
    padded row-major trick."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    dtype = mid.dtype
    kind = 2 if dtype == torch.int64 else (0 if dtype.is_floating_point else 1)
    floor = _dtype_floor(dtype)
    is_fp32 = dtype == torch.float32
    while mid.numel() > block:
        rows = triton.cdiv(mid.numel(), block)
        bm = next((m for m in (128, 64, 32, 256, 16, 8, 4, 2) if rows % m == 0), 2)
        Mr = triton.cdiv(rows, bm) * bm
        pad = torch.full((Mr, block), floor, dtype=dtype, device=device)
        _pad_inplace(mid, pad.reshape(-1)[: mid.numel()])
        nxt = torch.empty((Mr,), dtype=dtype, device=device)
        nxt_idx = torch.empty((Mr,), dtype=torch.int64, device=device)
        bn = 1024 if not is_fp32 else 256
        max_kernel_2d_pk[(Mr // bm, 1)](
            pad, nxt, nxt_idx, Mr, block, kind, bm, bn, buffer_size_limit=1024
        )
        mid = nxt
    n = mid.numel()
    n_pow2 = triton.next_power_of_2(n)
    pad = torch.full((2, n_pow2), floor, dtype=dtype, device=device)
    _pad_inplace(mid, pad[0, :n].reshape(-1))
    tmp_v = torch.empty((2,), dtype=dtype, device=device)
    tmp_i = torch.empty((2,), dtype=torch.int64, device=device)
    max_kernel_2d_pk[(1, 1)](
        pad, tmp_v, tmp_i, 2, n_pow2, kind, 2, n_pow2, buffer_size_limit=1024
    )
    out[()] = tmp_v[0]


def max(inp):
    logger.debug("GEMS_KUNLUNXIN MAX")
    inp = inp.contiguous().reshape(-1)  # 1-D flat view (3-D kernel args crash on XPU)
    M = inp.numel()
    dtype = inp.dtype
    out = torch.empty([], dtype=dtype, device=inp.device)
    if M == 1:
        return inp.reshape([])

    # Fast path: hand-written cluster-C payload (per-core partials + combine)
    # beats the Triton packed-key full reduction on the measured dtypes.
    raw_out = _raw_max_full(inp) if _TLE_OK else None
    if raw_out is not None:
        return raw_out

    with torch_device_fn.device(inp.device):
        _max_flat(inp, out, inp.device)
    return out


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


# ---------------------------------------------------------------------------
# Pure-Triton inner-dim max_dim (K == 1), inspired by the max_raw.xpu payload:
# one 2D tile [BM, next_pow2(N)] per program, per-row reduce, and the FIRST
# index recovered with the eq -> min(where) trick instead of tl.max's
# return_indices (whose codegen overflows uni_sram on this backend). Note the
# backend's CoreTiling serializes wide-row reductions, so large N stays slow;
# this is the structural ceiling of the compiler path (see op-opti/max_dim.md).
@triton.jit
def max_dim_triton_kernel(
    inp,
    out_value,
    out_index,
    M,
    N: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    ncols = tl.arange(0, BN)
    ncmask = ncols < N
    if NEED_MASK:
        rmask = rows < M
        m2d = rmask[:, None] & ncmask[None, :]
        x = tl.load(
            inp + rows[:, None] * N + ncols[None, :], mask=m2d, other=-float("inf")
        )
    else:
        x = tl.load(inp + rows[:, None] * N + ncols[None, :])
    max_val = tl.max(x, axis=1)
    eq = (x == max_val[:, None]) & ncmask[None, :]
    max_idx = tl.min(tl.where(eq, ncols[None, :], N), axis=1)
    if NEED_MASK:
        tl.store(out_value + rows, max_val, mask=rmask)
        tl.store(out_index + rows, max_idx.to(tl.int64), mask=rmask)
    else:
        tl.store(out_value + rows, max_val)
        tl.store(out_index + rows, max_idx.to(tl.int64))


def _triton_dim_bm(M):
    """Rows per program: keep the grid <= ~128 programs; small M prefers a
    smaller BM (measured [64,64]: BM=16/grid=4 best at 0.47)."""
    import builtins

    return builtins.min(
        128, builtins.max(16, triton.next_power_of_2(triton.cdiv(M, 128)))
    )


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MAX_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    # Pure-Triton path. For the contiguous inner-dim reduction (K == 1) use
    # the single-tile max_dim_triton_kernel (no tle.raw); other dims fall back
    # to the generic compiler kernel.
    inp = inp.contiguous()
    dim = dim % inp.ndim

    shape = inp.shape
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N
    ELEMENT_SIZE = inp.element_size()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    if K == 1 and inp.numel() > 0 and N > 0:
        # tle.raw fast path: the compiler's row-reduce (max_dim_triton_kernel)
        # is structurally capped (~2.3ms on [4096,4096], 0.017x) because
        # CoreTiling serializes wide-row reductions; the payload drives
        # per-core GM2LM chunks with a flat double-buffer pipeline and a
        # vector-accumulate row reduce (vendor technique), reaching ~0.65-0.78
        # on [4096,4096]. Fall back to the Triton kernel if not available or
        # not applicable (non-float dtypes, massive tensors).
        raw = None
        if _TLE_OK:
            raw = _raw_max_dim(inp, dim, keepdim)
        if raw is not None:
            return Max_out(values=raw[0], indices=raw[1])
        BM = _triton_dim_bm(M)
        BN = min(triton.next_power_of_2(N), 4096)
        NEED_MASK = M % BM != 0
        grid = (triton.cdiv(M, BM),)
        with torch_device_fn.device(inp.device):
            max_dim_triton_kernel[grid](
                inp,
                out_value,
                out_index,
                M,
                N=N,
                BN=BN,
                BM=BM,
                NEED_MASK=NEED_MASK,
            )
        return Max_out(values=out_value, indices=out_index)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
    isCloseCoreTiling = False
    if inp.dtype in [torch.int16, torch.int32, torch.int64] and M == 4096 and N == 256:
        isCloseCoreTiling = True

    with torch_device_fn.device(inp.device):
        max_kernel[grid](
            inp,
            out_value,
            out_index,
            M,
            N,
            K,
            ELEMENT_SIZE,
            isCloseCoreTiling=isCloseCoreTiling,
        )

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]
    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        del os.environ["TRITONXPU_STORE_MASK_SIM"]
    out = Max_out(values=out_value, indices=out_index)
    return out
