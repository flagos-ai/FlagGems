import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)


_REDUCE_BLOCK = 8192
_TREE16_BLOCK = 512
_FAST_BN_FP16 = (1024, 256, 512, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 256, 16, 8, 4, 2, 1)
_FAST_BM_FP32 = (64, 128, 32, 256, 16, 8, 4, 2, 1)

# =============================================================================
# tle row-reduce (see _kunlunxin/ops/sum.py for the reference pattern).
# `tritonxpu-tle-core-tiling` hands each of the 64 cores XBLOCK/core_num whole
# rows of a [XBLOCK, YBLOCK] tile, so the axis-1 reduce stays core-local and
# needs no cluster barrier: a plain 2D `tl.reduce` of the same tile measures
# 124us on 4096x4096 f32 against 50us here, and the pointer path's staged
# 2-launch flat reduce measures 26us on 1M f32 against ~13us. The cluster DMA
# (`tle.gpu.copy` -> LM) is also masked: a descriptor clamps the transfer to
# its extents, so an arbitrary [M, N] needs one launch instead of the pointer
# path's mask-free-fast-tile / masked-fallback split.
# =============================================================================
# LM per core. 4 KB is what the reference implementation settles on; 8 KB fails
# to allocate ("TLE kernel stack is over the local-memory budget").
_TLE_LM_BYTES_PER_CORE = 4096
# Cores / clusters on the chip; one program per cluster is the target grid shape.
_TLE_CORE_NUM = 64
_TLE_CLUSTERS = 8
# Rows per program. The reduce is core-local at XBLOCK >= core_num, and wider
# rows amortise the launch (tuned for sum on 1G f32: 512 measured best).
_TLE_XBLOCK = 512
# A row this wide has to buy a long YBLOCK, even at the cost of a smaller XBLOCK.
_TLE_WIDE_ROW_BYTES = 32768
# Below this the reduce is all launch and result write, so take the widest XBLOCK.
_TLE_NARROW_ROW_BYTES = 2048
# Flat reduce: reshape to [n // COLS, COLS] and row-reduce. 256 measured
# 9.7us of stage-1 on 1M f32 against 17.3us at 4096: n is small enough that a
# wider grid (more programs, smaller tiles) wins, unlike the 1G-element sums the
# reference implementation tunes for.
_TLE_FLAT_COLS = 256
# Everything that fits one program's loop is reduced in a single [1, n] launch.
_TLE_FLAT_ONESHOT = 8192

# Only dtypes verified to survive an LM round trip on KL3. float64 is absent
# because it is broken on this path (see sum.py); bool has no LM dtype.
_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}
# Accumulator per input dtype: fp16/bf16 accumulate in f32 (16-bit products lose
# a tree level otherwise), integers in int64 -- both matching the pointer path.
_TLE_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.float32: tl.float32,
    torch.bfloat16: tl.float32,
    torch.int16: tl.int64,
    torch.int32: tl.int64,
    torch.int64: tl.int64,
}
_TLE_ACC_TORCH = {tl.float32: torch.float32, tl.int64: torch.int64}


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()

# TLE kernels are only usable if the installed triton can lower their
# tt.make_range (a tle local_ptr index) -- older flagtree builds cannot (they
# fail with "failed to legalize operation 'tt.make_range' that was explicitly
# marked illegal"), so a launch failure is detected once per process and the
# proofing pointer path is used from then on.
_TLE_OK: bool | None = None  # None = untried, True = works, False = use pointer path


def _tle_launch(launch, *args, **kwargs) -> bool:
    """Run one tle launch, recording whether tle works for this process.

    A compile-time failure (RuntimeError from the triton compiler) leaves the
    kernel un-launched (no partial side effects), so the caller can fall back to
    the pointer path. The failure is sticky so later calls skip the broken tle
    path instead of recompiling on every call.
    """
    global _TLE_OK
    if _TLE_OK is False:
        return False
    try:
        launch(*args, **kwargs)
        _TLE_OK = True
        return True
    except RuntimeError:
        _TLE_OK = False
        return False


def _npo2(x):
    """`triton.next_power_of_2` costs ~3.2us a call; this is 0.08us."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


_TLE_ROW_GEOM = {}
_TLE_GEOM_MISS = object()


def _tle_row_geom(M, N, itemsize):
    """Cached (xblock, yblock) tiling for a [M, N] row-reduce.

    XBLOCK is rows per program and YBLOCK the LM budget divided by it, so the two
    trade off. Favour one program per cluster, then let the row length pull XBLOCK
    down when the row is long enough to need the YBLOCK, or up when the whole
    reduce is small enough that the launch and the result write are all there is.
    XBLOCK may exceed M -- the padding rows are clamped by both descriptor copies.
    """
    geom_key = (M, N, itemsize)
    geom = _TLE_ROW_GEOM.get(geom_key, _TLE_GEOM_MISS)
    if geom is not _TLE_GEOM_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 256
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    yblock = max(1, min(_npo2(N), tile_elems // xblock))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock)
    _TLE_ROW_GEOM[geom_key] = geom
    return geom


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_prod_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """Product of a [XBLOCK, YBLOCK]-tiled slice of a [M, N] input along axis=1.

    Partial tiles need no mask: `tle.gpu.copy` clamps the transfer to the
    descriptor extents, so a tile hanging off the edge leaves stale LM bytes --
    which is why the one partial iteration pre-fills the buffer with 1.0, the
    multiplicative identity (sum uses 0.0; a product needs 1.0).
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    acc = tl.full([XBLOCK], value=1, dtype=ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.full([XBLOCK, YBLOCK], value=1, dtype=IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(ACC_DTYPE)
        blk = tl.reduce(v, axis=1, combine_fn=reduce_mul)
        acc = acc * blk
    tl.store(c_ptrs, acc.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_row_reduce(a, c, xblock, yblock):
    """`c[m] = prod(a[m, :])` for a 2-D contiguous `a` and 1-D `c`."""
    consts = (
        xblock,
        yblock,
        _TLE_TL_DTYPE[a.dtype],
        _TLE_ACC_DTYPE[a.dtype],
        _TLE_TL_DTYPE[c.dtype],
        a.shape[1] % yblock != 0,
    )
    grid = (-(-a.shape[0] // xblock),)
    _tle_prod_row_kernel[grid](
        TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
        TensorDescriptor.from_tensor(c, block_shape=[xblock]),
        a.shape[1],
        *consts,
    )


def _tle_prod_flat(inp, out, device):
    """Full product as repeated row-reduces of a [n // COLS, COLS] view.

    Feeding the output back in is the whole algorithm. The final residue (at most
    `_TLE_FLAT_ONESHOT` partials) is finished by the pointer-path `_reduce_partials`,
    which for this size is one launch and is cheaper than another tle stage.
    """
    with torch_device_fn.device(device):
        cur = inp.reshape(-1)
        while cur.numel() > _TLE_FLAT_ONESHOT:
            n = cur.numel()
            cols = (
                _TLE_FLAT_COLS
                if n >= 2 * _TLE_FLAT_COLS
                else max(2, _npo2(n) // _TLE_CORE_NUM)
            )
            rows = n // cols
            if rows < 2:
                break
            main = rows * cols
            tail = n - main
            acc_dtype = _TLE_ACC_DTYPE[cur.dtype]
            nxt = torch.empty(
                (rows + tail,), dtype=_TLE_ACC_TORCH[acc_dtype], device=device
            )
            xblock, yblock = _tle_row_geom(rows, cols, cur.element_size())
            _tle_row_reduce(cur[:main].view(rows, cols), nxt[:rows], xblock, yblock)
            if tail:
                nxt[rows:].copy_(cur[main:])
            cur = nxt
        _reduce_partials(cur, cur.numel(), out, device, inp.dtype.is_floating_point)


def _tle_prod_dim(inp, out_flat, rows, N, device):
    """`out_flat[m] = prod(inp[m, :])` for the contiguous [rows, N] reduce view."""
    with torch_device_fn.device(device):
        xblock, yblock = _tle_row_geom(rows, N, inp.element_size())
        _tle_row_reduce(inp.view(rows, N), out_flat, xblock, yblock)


@triton.jit
def reduce_mul(a, b):
    return a * b


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, so
    the whole reduction runs mask-free, or None."""
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


def _tree16(dt):
    return dt == torch.float16


def _work_dtype(dt):
    return torch.float32 if dt.is_floating_point else torch.int64


@libentry()
@triton.jit
def prod_row2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    ACC32: tl.constexpr,
    TREE16: tl.constexpr = False,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    if ACC32:
        acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if TREE16:
                if NEED_MASK:
                    mask = row_mask and (cols < N)
                    a = tl.load(inp + cols, mask, other=1.0)
                else:
                    a = tl.load(inp + cols)
                blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
                acc = acc * blk.to(tl.float32)
            else:
                if NEED_MASK:
                    mask = row_mask and (cols < N)
                    a = tl.load(inp + cols, mask, other=1.0).to(tl.float32)
                else:
                    a = tl.load(inp + cols).to(tl.float32)
                blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
                acc = acc * blk
    else:
        acc = tl.full([BLOCK_M, 1], value=1, dtype=tl.int64)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if NEED_MASK:
                mask = row_mask and (cols < N)
                a = tl.load(inp + cols, mask, other=1).to(tl.int64)
            else:
                a = tl.load(inp + cols).to(tl.int64)
            blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
            acc = acc * blk
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


@libentry()
@triton.jit
def prod_mid_block(inp, mid, BLOCK: tl.constexpr, ACC32: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(mid + pid, p)


@libentry()
@triton.jit
def prod_tailk(inp, out, START, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    offs = START + tl.arange(0, WIDTH)
    if ACC32:
        v = tl.load(inp + offs).to(tl.float32)
    else:
        v = tl.load(inp + offs).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


@libentry()
@triton.jit
def prod_final_mask(inp, out, n, WIDTH: tl.constexpr, ACC32: tl.constexpr):
    offs = tl.arange(0, WIDTH)
    mask = offs < n
    if ACC32:
        v = tl.load(inp + offs, mask=mask, other=1.0).to(tl.float32)
    else:
        v = tl.load(inp + offs, mask=mask, other=1).to(tl.int64)
    p = tl.reduce(v, axis=0, combine_fn=reduce_mul)
    tl.store(out, p)


def _bm_tail(tr):
    b = 1
    while tr % (b * 2) == 0:
        b *= 2
    return b


@libentry()
@triton.jit
def prod_dim_chunk(
    inp,
    part,
    N,
    B0,
    C0,
    C: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = ext.program_id(0)
    mb = ext.program_id(1)
    rows = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)[:, None]
    inp = inp + rows * N + B0 + c * CHUNK
    acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
        acc = acc * blk
    tl.store(part + rows * C + C0 + c, acc)


@libentry()
@triton.jit
def prod_dim_single(
    inp,
    out,
    N,
    CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mb = ext.program_id(1)
    rows = (mb * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)[:, None]
    inp = inp + rows * N
    acc = tl.full([BLOCK_M, 1], value=1.0, dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        blk = tl.reduce(a, axis=1, combine_fn=reduce_mul)[:, None]
        acc = acc * blk
    tl.store(out + rows, acc)


def _pow2_decomp(r):
    parts = []
    while r:
        p = 1 << (r.bit_length() - 1)
        parts.append(p)
        r -= p
    return parts


def _reduce_partials(data, n, out, device, acc32):
    """Staged product of `n` fp32/int partials -> scalar `out`, mask-free
    (tails pow2-decomposed; final pad of 1.0s)."""
    with torch_device_fn.device(device):
        while n > _REDUCE_BLOCK:
            k = n // _REDUCE_BLOCK
            r = n - k * _REDUCE_BLOCK
            chunks = _pow2_decomp(r)
            sz = k + len(chunks)
            midn = torch.empty((sz,), dtype=data.dtype, device=device)
            if k:
                prod_mid_block[(k, 1)](
                    data, midn, _REDUCE_BLOCK, acc32, buffer_size_limit=2048
                )
            pos = k * _REDUCE_BLOCK
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    data,
                    midn[k + i : k + i + 1],
                    pos,
                    w,
                    acc32,
                    buffer_size_limit=2048,
                )
                pos += w
            data = midn
            n = sz
        width = triton.next_power_of_2(n)
        prod_final_mask[(1, 1)](data, out, n, width, acc32, buffer_size_limit=2048)


def _prod_flat(inp, out, device):
    numel = inp.numel()
    block = _REDUCE_BLOCK
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    acc32 = inp.dtype.is_floating_point
    tree16 = _tree16(inp.dtype)
    wdt = _work_dtype(inp.dtype)
    with torch_device_fn.device(device):
        if rows:
            tile = _pick_fast_tile(rows, block, is_fp32)
            bm = tile[0] if tile else 2
            if tree16:
                bn = _TREE16_BLOCK
            else:
                bn = 512
            chunks = _pow2_decomp(res) if res else []
            mid = torch.empty((rows + len(chunks),), dtype=wdt, device=device)
            prod_row2d[(max(rows // bm, 1), 1)](
                inp,
                mid,
                rows,
                block,
                bm,
                bn,
                False,
                acc32,
                tree16,
                buffer_size_limit=2048,
            )
            pos = rows * block
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp,
                    mid[rows + i : rows + i + 1],
                    pos,
                    w,
                    acc32,
                    buffer_size_limit=2048,
                )
                pos += w
        else:
            chunks = _pow2_decomp(res)
            mid = torch.empty((len(chunks),), dtype=wdt, device=device)
            pos = 0
            for i, w in enumerate(chunks):
                prod_tailk[(1, 1)](
                    inp, mid[i : i + 1], pos, w, acc32, buffer_size_limit=2048
                )
                pos += w
        _reduce_partials(mid, mid.numel(), out, device, acc32)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD")
    if dtype is None:
        dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    numel = inp.numel()
    out = torch.empty([], dtype=dtype, device=inp.device)
    if numel == 0:
        out.fill_(1)
        return out
    if numel == 1:
        with torch_device_fn.device(inp.device):
            out.copy_(inp.reshape([]))
        return out
    if _TLE_AVAILABLE and inp.dtype in _TLE_TL_DTYPE and dtype in _TLE_TL_DTYPE:
        if _tle_launch(_tle_prod_flat, inp, out, inp.device):
            return out
        # tle compile failure (older flagtree): fall through to the pointer path.
    with torch_device_fn.device(inp.device):
        _prod_flat(inp, out, inp.device)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN PROD_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if dtype is None:
        dtype = torch.int64 if not inp.dtype.is_floating_point else inp.dtype
    shape = list(inp.shape)
    d = dim % inp.ndim
    N = shape[d]
    M = 1
    for s in shape[:d]:
        M *= s
    K = 1
    for s in shape[d + 1 :]:
        K *= s

    out_shape = shape.copy()
    out_shape[d] = 1
    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    if M == 0 or K == 0:
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 1:
        with torch_device_fn.device(inp.device):
            out.copy_(inp)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    order = [i for i in range(inp.dim()) if i != d] + [d]
    view = inp.permute(order)
    if view.is_contiguous():
        src = view
    else:
        src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            src.copy_(view)

    rows = M * K
    out_flat = out.reshape(rows)
    is_fp32 = dtype == torch.float32
    acc32 = dtype.is_floating_point
    tree16 = False
    tle_used = _TLE_AVAILABLE and src.dtype in _TLE_TL_DTYPE and dtype in _TLE_TL_DTYPE
    if tle_used and not _tle_launch(_tle_prod_dim, src, out_flat, rows, N, inp.device):
        # tle compile failure (older flagtree): fall through to the pointer path.
        tle_used = False
    if not tle_used:
        with torch_device_fn.device(inp.device):
            tile = _pick_fast_tile(rows, N, is_fp32)
            if tile is not None and (not tree16 or N % _TREE16_BLOCK == 0):
                bm, bn = tile
                if tree16:
                    bn = _TREE16_BLOCK
                prod_row2d[(max(rows // bm, 1), 1)](
                    src,
                    out_flat,
                    rows,
                    N,
                    bm,
                    bn,
                    False,
                    acc32,
                    tree16,
                    buffer_size_limit=2048,
                )
            else:
                bn = min(triton.next_power_of_2(N), _REDUCE_BLOCK)
                if tree16:
                    bn = min(bn, _TREE16_BLOCK)
                bm = triton.next_power_of_2(min(triton.cdiv(rows, 12), 65536 // bn))
                grid = (triton.cdiv(rows, bm),)
                prod_row2d[grid](
                    src,
                    out_flat,
                    rows,
                    N,
                    bm,
                    bn,
                    True,
                    acc32,
                    tree16,
                    buffer_size_limit=2048,
                )
    if not keepdim:
        out = torch.squeeze(out, d)
    return out
