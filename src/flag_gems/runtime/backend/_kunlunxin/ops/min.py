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
from flag_gems.utils.limits import get_dtype_max

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tle.raw fast path (P800 xpu3, cluster C payloads in min_raw.xpu) -- mirrors
# amin.py's layered dispatch: every contiguous inner-dim (K == 1) reduction and
# the full reduction try the hand-written payload first (per-core GM2LM DMA +
# vvmin vector accumulate + SM cluster reduce), the compiler paths are the
# fallback.  min_raw.xpu is the max_raw.xpu mirror with min semantics (running
# min init +inf, strict '<' keeps the FIRST index on ties).
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except Exception:  # pragma: no cover - environment without tle
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
# Precompiled, secrecy-hardened device object (packed from min_raw.xpu +
# min_flat_sm.xpu + min_full_sm.xpu via docs/xpu3/how_to_pack_payload/
# pack_payload.py then relocatably linked with `ld.lld -r *.o -o min.o`). We
# ship the .o (not the .xpu source); the stub names below MUST equal the entry
# symbols and their signatures must match the C++ ABI (object= mode skips the
# merge-time signature check, so a mismatch fails at runtime, not compile).
# Re-pack after editing any .xpu -- the compile-cache key is the .o content
# digest, so a stale .o silently keeps the old code.
_MIN_OBJ = os.path.join(os.path.dirname(_HERE), "payload", "obj", "min.o")
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MIN_ELEMS = 2**31 - 1

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

# The hand-written min payloads are loaded via tle.raw's object= loader, which
# only exists on newer tle builds.  On older/other environments the decorator
# raises at import time (e.g. tle.raw missing, or
# "RawJITFunction.__init__() got an unexpected keyword argument 'object'").
# Guard the whole declaration so an unsupported tle leaves _HAS_RAW False and
# the pure-Triton reduction kernels below run instead -- a failure here must
# never crash the vendor-lib import.
_HAS_RAW = False
if _TLE_OK:
    try:

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_row_raw(
            in_, out_val, out_idx, M, N, esz, type_code, rows_start, rows_count, rpc
        ): ...

        @triton.jit(do_not_specialize=["M", "N", "esz", "type_code", "per", "rpc"])
        def min_dim_raw_kernel(In, OutV, OutI, M, N, esz, type_code, per, rpc):
            pid = tl.program_id(0)
            tle.raw.call(
                min_row_raw, (In, OutV, OutI, M, N, esz, type_code, pid * per, per, rpc)
            )

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_full_raw(in_, mid, M, esz, type_code, per, slot): ...

        @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
        def min_full_raw_kernel(In, Mid, M, esz, type_code, per):
            pid = tl.program_id(0)
            tle.raw.call(min_full_raw, (In, Mid, M, esz, type_code, per, pid))

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_combine_raw(mid, out, n, esz, type_code): ...

        @triton.jit(do_not_specialize=["n", "esz", "type_code"])
        def min_full_combine_kernel(Mid, Out, n, esz, type_code):
            tle.raw.call(min_combine_raw, (Mid, Out, n, esz, type_code))

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_flat_idx_raw(in_, mid, M, esz, type_code, per, slot): ...

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_flat_idx_combine_raw(mid, out_val, out_idx, n, esz, type_code): ...

        @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
        def min_flat_idx_kernel(In, Mid, M, esz, type_code, per):
            pid = tl.program_id(0)
            tle.raw.call(min_flat_idx_raw, (In, Mid, M, esz, type_code, per, pid))

        @triton.jit(do_not_specialize=["n", "esz", "type_code"])
        def min_flat_idx_combine_kernel(Mid, OutV, OutI, n, esz, type_code):
            tle.raw.call(min_flat_idx_combine_raw, (Mid, OutV, OutI, n, esz, type_code))

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_flat_sm_raw(in_, mid, M, esz, type_code, per, slot): ...

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_flat_sm_combine_raw(mid, out_val, out_idx, n, esz, type_code): ...

        @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
        def min_flat_sm_kernel(In, Mid, M, esz, type_code, per):
            pid = tl.program_id(0)
            tle.raw.call(min_flat_sm_raw, (In, Mid, M, esz, type_code, per, pid))

        @triton.jit(do_not_specialize=["n", "esz", "type_code"])
        def min_flat_sm_combine_kernel(Mid, OutV, OutI, n, esz, type_code):
            tle.raw.call(min_flat_sm_combine_raw, (Mid, OutV, OutI, n, esz, type_code))

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_full_sm_raw(in_, mid, M, esz, type_code, per, slot): ...

        @tle.raw.dialect("xpu3", object=_MIN_OBJ, arch=3)
        def min_full_sm_combine_raw(mid, out, n, esz, type_code): ...

        @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
        def min_full_sm_kernel(In, Mid, M, esz, type_code, per):
            pid = tl.program_id(0)
            tle.raw.call(min_full_sm_raw, (In, Mid, M, esz, type_code, per, pid))

        @triton.jit(do_not_specialize=["n", "esz", "type_code"])
        def min_full_sm_combine_kernel(Mid, Out, n, esz, type_code):
            tle.raw.call(min_full_sm_combine_raw, (Mid, Out, n, esz, type_code))

        _HAS_RAW = True
    except Exception:  # pragma: no cover - environment without tle.raw object= support
        _HAS_RAW = False


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


def _raw_min_dim(inp, dim, keepdim):
    """min.dim along the innermost (contiguous) dim via the raw payload.

    Returns (values, indices) or None when the raw path does not apply.
    """
    if not _HAS_RAW:
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
    if M * N > _RAW_MIN_ELEMS:
        return None
    esz = inp.element_size()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    if M == 1:
        # Single row: 1D argmin spread over all 768 cores.  Each cluster SM-
        # reduces its 64 cores' (value, column) pairs (sync_all) and writes one
        # 16-byte slot; the combine reduces 12 slots in a single 192B GM2LM.
        ncores = _NCLUSTER * 64
        nused = builtins.min(ncores, N)
        if nused < ncores:
            # Fewer elements than payload cores: the empty clusters' mid slots
            # stay uninitialized, which makes the 12-slot combine result
            # environment-dependent (same class as the _raw_min_full M <
            # _NCLUSTER * 64 guard; test_min (1,2) hit this on the kunlunxin
            # CI).  The compiler single-row kernel handles these tiny rows
            # correctly, so keep the raw payload off the sub-core-count range.
            return None
        per = (N + nused - 1) // nused
        mid = torch.empty(_NCLUSTER * 16, dtype=torch.uint8, device=inp.device)
        with torch_device_fn.device(inp.device):
            min_flat_sm_kernel[(_NCLUSTER,)](_view_u8(inp), mid, N, esz, type_code, per)
            min_flat_sm_combine_kernel[(1,)](
                mid, _view_u8(out_value), out_index, _NCLUSTER, esz, type_code
            )
        return out_value, out_index

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
        min_dim_raw_kernel[(grid_n,)](
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


def _raw_min_full(inp):
    """Full-tensor min via per-core partials + a tiny combine kernel.

    Returns the scalar result tensor, or None when not applicable.
    """
    if not _HAS_RAW:
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    M = inp.numel()
    if M < _NCLUSTER * 64 or M > _RAW_MIN_ELEMS:
        # Fewer elements than payload cores (12 clusters x 64): the empty
        # cores' SM-reduce and the uninitialized mid slots make the combine
        # result environment-dependent (NaN / +inf / fp16-max instead of the
        # true min; test_min shape0 (1,2) hit this on the kunlunxin CI).  The
        # compiler flat path (padded with +inf) handles these tiny tensors
        # correctly, so keep the raw payload off the sub-core-count range.
        return None
    esz = inp.element_size()

    per = (M + _NCLUSTER * 64 - 1) // (_NCLUSTER * 64)
    mid = torch.empty(_NCLUSTER * 8, dtype=torch.uint8, device=inp.device)
    out = torch.empty([], dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        min_full_sm_kernel[(_NCLUSTER,)](_view_u8(inp), mid, M, esz, type_code, per)
        min_full_sm_combine_kernel[(1,)](mid, _view_u8(out), _NCLUSTER, esz, type_code)
    return out


_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MIN_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
_VIEW_BN_FP16 = (512, 256, 128, 64, 1024)
_VIEW_BN_FP32 = (512, 256, 128, 64, 1024)
_VIEW_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_VIEW_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
_MIN_FAST_TILE_LANES = 8192
_NEW_MIN_VR = 65536


def _is_fast_dtype(dtype):
    return dtype in (torch.float16, torch.float32, torch.bfloat16)


def _pick_chunk_tile(VR, CHUNK, is_fp32):
    """Return (BLOCK_M, BLOCK_N) for pass 1 over the chunk-major view
    (VR, CHUNK) with VR % BLOCK_M == 0, CHUNK % BLOCK_N == 0,
    BLOCK_M * BLOCK_N >= _MIN_FAST_TILE_LANES, and CHUNK / BLOCK_N >= 2 loop
    trips (single-trip kernels do not pipeline on this XPU, so they are only
    a last resort), or None when no such mask-free tile covers this shape
    (small shapes keep the legacy masked kernel)."""
    bns = _VIEW_BN_FP32 if is_fp32 else _VIEW_BN_FP16
    bms = _VIEW_BM_FP32 if is_fp32 else _VIEW_BM_FP16
    for min_trips in (2, 1):
        for bn in bns:
            if CHUNK % bn != 0 or CHUNK // bn < min_trips:
                continue
            for bm in bms:
                if VR % bm == 0 and bm * bn >= _MIN_FAST_TILE_LANES:
                    return bm, bn
    return None


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0, N % BLOCK_N == 0 and
    BLOCK_M * BLOCK_N >= _MIN_FAST_TILE_LANES, or None when no such mask-free
    tile covers this shape (small shapes keep the legacy masked kernel)."""
    if N < _FAST_MIN_N:
        return None
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    for bn in bns:
        if N % bn != 0:
            continue
        for bm in bms:
            if M % bm != 0:
                continue
            if bm * bn >= _MIN_FAST_TILE_LANES:
                return bm, bn


@libentry()
@triton.jit
def min_split_kernel(
    inp,
    part_val,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    row_mask = rows < M
    ic = 0
    for off in range(0, N, BLOCK_CHUNK):
        cols = off + tl.arange(0, BLOCK_CHUNK)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        blk = tl.min(a, axis=1)[:, None]
        if NEED_MASK:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk, mask=row_mask)
        else:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk)
        ic += 1


@libentry()
@triton.jit
def min_chunk_kernel_col(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None] + c_off[None, :].to(tl.int64) * M,
        mask=row_mask[:, None] and cmask[None, :],
        other=float("inf"),
    )
    _, bc = tl.min(vals, axis=1, return_indices=True)
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def min_scan_kernel_col(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("inf"),
    ).to(tl.float32)
    u = a.to(tl.int32, bitcast=True)
    neg = u < 0
    ordered = tl.where(neg, ~u, u ^ -2147483648)
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | cols.to(tl.int64)
    blk = tl.min(pack, axis=1)
    pos = (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    m = tl.load(part_val + rows.to(tl.int64) + c.to(tl.int64) * M, mask=row_mask)
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    if NEED_MASK:
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    else:
        inp_val = tl.load(inp_ptrs)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


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
def min_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


@libentry()
@triton.jit
def min_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    acc = tl.full([BLOCK_M, 1], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(inp + cols, mask, other=float("inf")).to(tl.float32)
            a = tl.where(mask, a, float("inf"))
            blk = tl.min(a, axis=1)[:, None]
        else:
            a = tl.load(inp + cols).to(tl.float32)
            blk = tl.min(a, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


def _pad_value(dtype):
    """Identity value for the min reduction: +inf for floats, dtype max otherwise."""
    if dtype.is_floating_point:
        return float("inf")
    return torch.iinfo(dtype).max


def _pad_buffer(src, pad_to, device):
    """Return a buffer of `pad_to` elements: [0, n) = src, [n, pad_to) filled
    with the reduction identity (+inf for floats, dtype-max for ints)."""
    n = src.numel()
    buf = torch.full((pad_to,), _pad_value(src.dtype), dtype=src.dtype, device=device)
    if n:
        if not tle_copy(src, buf[:n]):
            buf[:n].copy_(src)
    return buf


def _reduce_to_scalar(src, out):
    """Repeatedly reduce `src` with 8192-wide blocks until a scalar remains.
    Every masked lane is eliminated by padding each stage to a multiple of
    8192 with the reduction identity (XPU masked tails read OOB -- backend
    limitation; see the module note)."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = src.numel()
    data = src
    while n > block:
        n_pad = triton.cdiv(n, block) * block
        if n_pad > n:
            data = _pad_buffer(data, n_pad, data.device)
        mid_size = triton.cdiv(n, block)
        mid = torch.empty((mid_size,), dtype=data.dtype, device=data.device)
        min_kernel_1[(mid_size, 1)](
            data,
            mid,
            n,
            block,
            False,
            buffer_size_limit=2048,
        )
        data = mid
        n = mid_size
    if n < block:
        data = _pad_buffer(data, block, data.device)
    min_kernel_1[(1, 1)](
        data,
        out,
        n,
        block,
        False,
        buffer_size_limit=2048,
    )


def _min_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel). Mask-free 2-D row
    tiles over rows-of-8192 plus a staged scalar reduce; every tail is reads
    are padded so no masked/OOB load can happen on the XPU backend."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    pad_val = _pad_value(inp.dtype)
    if numel <= block:
        buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
        if not tle_copy(inp, buf[:numel]):
            buf[:numel].copy_(inp)
        min_kernel_1[(1, 1)](
            buf,
            out,
            numel,
            block,
            False,
            buffer_size_limit=2048,
        )
        return
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    bn = 1024 if not is_fp32 else 256
    if block % bn != 0:
        bn = next((b for b in _FAST_BN_FP32 if block % b == 0), block)
    bm = 128 if not is_fp32 else 64
    if rows % bm != 0:
        bm = next(
            (
                m
                for m in _FAST_BM_FP32
                if rows % m == 0 and m * bn >= _MIN_FAST_TILE_LANES
            ),
            128,
        )
    rows_exact = (rows // bm) * bm
    mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    with torch_device_fn.device(device):
        if rows_exact:
            min_kernel_2d[(rows_exact // bm, 1)](
                inp,
                mid,
                rows_exact,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
        if rows > rows_exact:
            tail_rows = rows - rows_exact
            tail_buf = torch.full(
                (bm * block,), pad_val, dtype=inp.dtype, device=device
            )
            src_tail = inp[rows_exact * block : rows * block]
            if not tle_copy(src_tail, tail_buf[: tail_rows * block]):
                tail_buf[: tail_rows * block].copy_(src_tail)
            tail_mid = torch.empty((bm,), dtype=inp.dtype, device=device)
            min_kernel_2d[(1, 1)](
                tail_buf,
                tail_mid,
                bm,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
            src_h1 = tail_mid[:tail_rows]
            dst_h1 = mid[rows_exact : rows_exact + tail_rows]
            if not tle_copy(src_h1, dst_h1):
                dst_h1.copy_(src_h1)
        if res:
            res_buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
            src_res = inp[rows * block :]
            if not tle_copy(src_res, res_buf[:res]):
                res_buf[:res].copy_(src_res)
            min_kernel_1[(1, 1)](
                res_buf,
                mid[rows:],
                res,
                block,
                False,
                buffer_size_limit=2048,
            )
        _reduce_to_scalar(mid, out)


def min(inp):
    logger.debug("GEMS_KUNLUNXIN MIN")
    dtype = inp.dtype
    out = torch.empty([], dtype=dtype, device=inp.device)
    # tle.raw fast path first (per-cluster SM min, 12-slot combine), compiler
    # flat as fallback
    raw_out = _raw_min_full(inp)
    if raw_out is not None:
        return raw_out
    with torch_device_fn.device(inp.device):
        _min_flat(inp, out, inp.device)
    return out


@libentry()
@triton.jit
def min_chunk_val_kernel(
    inp,
    part_val,
    M,
    W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c.to(tl.int64) * W
    out = part_val + rows.to(tl.int64)
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, W, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.minimum(acc, a)
    tl.store(out, tl.min(acc, axis=1)[:, None], row_mask)


@libentry()
@triton.jit
def min_chunk_kernel(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None].to(tl.int64) * NC + c_off[None, :],
        mask=row_mask[:, None] and cmask[None, :],
        other=float("inf"),
    )
    _, bc = tl.min(vals, axis=1, return_indices=True)
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def min_scan_kernel(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("inf"),
    ).to(tl.float32)
    u = a.to(tl.int32, bitcast=True)
    neg = u < 0
    ordered = tl.where(neg, ~u, u ^ -2147483648)
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | cols.to(tl.int64)
    blk = tl.min(pack, axis=1)
    pos = (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    m = tl.load(
        part_val + rows.to(tl.int64) * (N // BLOCK_CHUNK) + c.to(tl.int64),
        mask=row_mask,
    )
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


# ---------------------------------------------------------------------------
# Pure-Triton inner-dim min_dim (K == 1), the min mirror of max.py's
# max_dim_triton_kernel: one 2D tile [BM, next_pow2(N)] per program, per-row
# tl.min reduce, and the FIRST index recovered with the eq -> min(where) trick
# instead of tl.min's return_indices (whose codegen overflows uni_sram on this
# backend). Used as the K == 1 fallback when the tle.raw payload does not apply;
# wide-N stays slow (CoreTiling serializes wide-row reductions) so tle.raw is
# still preferred, but this keeps correctness without routing every shape/dtype
# through the raw payload.
@triton.jit
def min_dim_triton_kernel(
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
            inp + rows[:, None] * N + ncols[None, :], mask=m2d, other=float("inf")
        )
    else:
        x = tl.load(inp + rows[:, None] * N + ncols[None, :])
    min_val = tl.min(x, axis=1)
    eq = (x == min_val[:, None]) & ncmask[None, :]
    min_idx = tl.min(tl.where(eq, ncols[None, :], N), axis=1)
    if NEED_MASK:
        tl.store(out_value + rows, min_val, mask=rmask)
        tl.store(out_index + rows, min_idx.to(tl.int64), mask=rmask)
    else:
        tl.store(out_value + rows, min_val)
        tl.store(out_index + rows, min_idx.to(tl.int64))


def _triton_dim_bm(M):
    """Rows per program: keep the grid <= ~128 programs; small M prefers a
    smaller BM (mirror of max.py)."""
    return builtins.min(
        128, builtins.max(16, triton.next_power_of_2(triton.cdiv(M, 128)))
    )


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MIN_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    Min_out = namedtuple("min", ["values", "indices"])

    # Mirror of max.py's max_dim dispatch: contiguous inner-dim reductions
    # (K == 1) try the tle.raw payload first and fall back to the pure-Triton
    # min_dim_triton_kernel; other dims (K > 1) use the generic compiler kernel.
    # This keeps the raw payload off the shapes/dtypes it does not apply to
    # (non-float dtypes, K > 1, huge tensors) instead of routing everything
    # through tle.raw.
    inp = inp.contiguous()
    dim = dim % inp.ndim

    shape = inp.shape
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if N == 1:
        # The reduced dim has size 1: values are the identity and indices all 0.
        # Native strided copy avoids the [BLOCK_M, N=1] reduction tile (uni_sram
        # OOM in TritonXPUCoreTiling).
        with torch_device_fn.device(inp.device):
            if not tle_copy(inp, out_value):
                out_value.copy_(inp)
        out_index.zero_()
        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
        return Min_out(values=out_value, indices=out_index)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    if K == 1 and inp.numel() > 0 and N > 0:
        # tle.raw fast path (per-core GM2LM pipeline + vvmin accumulate + SM
        # cluster reduce); the compiler row-reduce is structurally capped on
        # wide N (CoreTiling serializes). Fall back to the pure-Triton kernel
        # when the payload does not apply (non-float dtypes, massive tensors).
        raw = _raw_min_dim(inp, dim, keepdim) if _HAS_RAW else None
        if raw is not None:
            return Min_out(values=raw[0], indices=raw[1])
        BM = _triton_dim_bm(M)
        BN = builtins.min(triton.next_power_of_2(N), 4096)
        NEED_MASK = M % BM != 0
        grid = (triton.cdiv(M, BM),)
        with torch_device_fn.device(inp.device):
            min_dim_triton_kernel[grid](
                inp,
                out_value,
                out_index,
                M,
                N=N,
                BN=BN,
                BM=BM,
                NEED_MASK=NEED_MASK,
            )
        return Min_out(values=out_value, indices=out_index)

    # K > 1: generic compiler kernel over the native [M, N, K] layout.
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    isCloseCoreTiling = True
    with torch_device_fn.device(inp.device):
        min_kernel[grid](
            inp, out_value, out_index, M, N, K, isCloseCoreTiling=isCloseCoreTiling
        )
    return Min_out(values=out_value, indices=out_index)
