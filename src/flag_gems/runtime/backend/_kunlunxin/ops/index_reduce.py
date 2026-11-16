import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# reduce ids: 0=prod, 1=mean, 2=amax, 3=amin
_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}

# Metadata (per-output match count + unique owner) tuning.  Measured on XPU3:
# 16 sequential d's per program is the best trade-off for out_n >= 1024.
_META_BLOCK_D = 16
_META_BLOCK_C_CAP = 4096
_BLOCK = 512


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _index_reduce_meta_kernel(
    index,
    cnt,
    owner,
    N,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-output count and (smallest) owning source position.

    `index` is shared by every (pre, post) coordinate, so the many-to-one
    structure only has to be decoded once per call.  `cnt`/`owner` are
    allocated padded to `grid_d * BLOCK_D`, so the last program may write
    past `OUT_N` without touching memory outside the buffers.
    """
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    cmask = c < N
    cs = tl.where(cmask, c, 0)
    idx = tl.load(index + cs).to(tl.int32)
    for j in tl.static_range(0, BLOCK_D):
        d = pid * BLOCK_D + j
        m = cmask & (idx == d)
        n = tl.sum(m.to(tl.int32), axis=0)
        own = tl.min(tl.where(m, c, BLOCK_C), axis=0)
        tl.store(cnt + d, n)
        tl.store(owner + d, tl.minimum(own, N - 1))


@libentry()
@triton.jit
def _index_reduce_dup_flag_kernel(cnt, flag, OUT_N, BLOCK: tl.constexpr):
    mx = 0
    for start in tl.range(0, OUT_N, BLOCK):
        d = start + tl.arange(0, BLOCK)
        dmask = d < OUT_N
        ds = tl.where(dmask, d, 0)
        v = tl.load(cnt + ds)
        v = tl.where(dmask, v, 0)
        mx = tl.maximum(mx, tl.max(v, axis=0))
    tl.store(flag, mx)


@libentry()
@triton.jit(do_not_specialize=["N", "OUT_N", "POST", "PRE", "TOTAL"])
def _index_reduce_unique_kernel(
    inp,
    src,
    res,
    cnt,
    owner,
    N,
    OUT_N,
    POST,
    PRE,
    TOTAL,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One writer per output element (index has no duplicates).

    Each output element has at most one matching source entry, so this is a
    plain elementwise gather: no atomics, no read-modify-write races.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TOTAL
    post_i = offs % POST
    tmp = offs // POST
    d = tl.minimum(tmp % OUT_N, OUT_N - 1)
    pre_i = tl.minimum(tmp // OUT_N, PRE - 1)

    self_v = tl.load(inp + tl.where(valid, offs, 0)).to(tl.float32)
    n = tl.load(cnt + d)
    c = tl.load(owner + d)
    src_off = (pre_i.to(tl.int64) * N + c.to(tl.int64)) * POST + post_i
    sv = tl.load(src + src_off).to(tl.float32)

    if REDUCE == 0:
        acc = self_v * sv
    elif REDUCE == 1:
        acc = (self_v + sv) * 0.5
    elif REDUCE == 2:
        acc = tl.maximum(self_v, sv)
    else:
        acc = tl.minimum(self_v, sv)
    if not INCLUDE_SELF:
        acc = sv

    out = tl.where(n > 0, acc, self_v)
    tl.store(res + offs.to(tl.int64), out.to(res.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["N", "OUT_N", "POST", "PRE", "TOTAL"])
def _index_reduce_dup_kernel(
    inp,
    index,
    src,
    res,
    N,
    OUT_N,
    POST,
    PRE,
    TOTAL,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """General (duplicate-safe) path: output-centric scan over the index.

    Only used when the index actually contains duplicates; the benchmark and
    most real workloads hit the unique fast path above.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # `res` is capacity-padded, so every lane may write; reads are clamped.
    post_i = offs % POST
    tmp = offs // POST
    d = tl.minimum(tmp % OUT_N, OUT_N - 1)
    pre_i = tl.minimum(tmp // OUT_N, PRE - 1)

    self_v = tl.load(inp + tl.minimum(offs, TOTAL - 1)).to(tl.float32)
    if REDUCE == 0:
        acc = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    elif REDUCE == 2:
        acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    elif REDUCE == 3:
        acc = tl.full((BLOCK,), float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
    hit = tl.zeros((BLOCK,), dtype=tl.int32)
    if INCLUDE_SELF:
        acc = self_v
        hit = hit + 1

    for c in tl.range(0, N):
        ic = tl.load(index + c).to(tl.int32)
        sel = ic == d
        src_off = (pre_i.to(tl.int64) * N + c.to(tl.int64)) * POST + post_i
        v = tl.load(src + src_off).to(tl.float32)
        if REDUCE == 0:
            acc = tl.where(sel, acc * v, acc)
        elif REDUCE == 1:
            acc = tl.where(sel, acc + v, acc)
        elif REDUCE == 2:
            acc = tl.where(sel, tl.maximum(acc, v), acc)
        else:
            acc = tl.where(sel, tl.minimum(acc, v), acc)
        hit = hit + sel.to(tl.int32)

    if REDUCE == 1:
        acc = acc / tl.maximum(hit, 1).to(tl.float32)
    out = tl.where(hit > 0, acc, self_v)
    tl.store(res + offs.to(tl.int64), out.to(res.dtype.element_ty))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _validate_args(inp, dim, index, source, reduce):
    assert reduce in ("prod", "mean", "amax", "amin"), f"Unsupported reduce: {reduce}"
    assert inp.ndim > 0, "Expected self to have at least one dimension"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim == 1, "Index is supposed to be a vector"
    assert index.dtype in (
        torch.int32,
        torch.int64,
    ), "Expected dtype int32/int64 for index"
    assert (
        inp.is_floating_point()
    ), "index_reduce_(): Expected self to be floating point"
    assert (
        source.dtype == inp.dtype
    ), "index_reduce_(): Expected self and source to have same dtype"
    assert (
        inp.ndim == source.ndim
    ), "Self and source should have the same number of dimensions"
    assert index.numel() == source.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert all(
        inp.size(i) == source.size(i) or i == dim for i in range(inp.ndim)
    ), "source.size(d) == self.size(d) for all dimensions d != dim"


def _prod(values):
    out = 1
    for value in values:
        out *= value
    return out


def _flat_result_slice(buf, total, shape):
    return buf[:total].view(shape)


def _index_reduce_apply(inp, dim, index, source, reduce, include_self):
    """Compute the out-of-place result into a fresh contiguous tensor."""
    dim = dim % inp.ndim
    index = index.contiguous()
    source = source.contiguous()
    out_n = inp.shape[dim]
    pre = _prod(inp.shape[:dim])
    post = _prod(inp.shape[dim + 1 :])
    total = pre * out_n * post
    dtype = inp.dtype
    device = inp.device

    if index.numel() == 0:
        result = torch.empty_like(inp, memory_format=torch.contiguous_format)
        result.copy_(inp)
        return result

    self_w = inp if inp.is_contiguous() else inp.contiguous()

    nprog = triton.cdiv(total, _BLOCK)
    padded = nprog * _BLOCK
    res = torch.empty(padded, dtype=dtype, device=device)

    with torch_device_fn.device(device):
        _launch(
            self_w,
            index,
            source,
            res,
            out_n,
            pre,
            post,
            total,
            reduce,
            include_self,
        )
    return _flat_result_slice(res, total, inp.shape)


def _launch(self_w, index, source, res, out_n, pre, post, total, reduce, include_self):
    n = index.numel()
    block_c = min(_META_BLOCK_C_CAP, triton.next_power_of_2(n))
    grid_d = triton.cdiv(out_n, _META_BLOCK_D)
    cnt = torch.empty(grid_d * _META_BLOCK_D, dtype=torch.int32, device=res.device)
    owner = torch.empty(grid_d * _META_BLOCK_D, dtype=torch.int32, device=res.device)
    _index_reduce_meta_kernel[(grid_d,)](
        index, cnt, owner, n, BLOCK_C=block_c, BLOCK_D=_META_BLOCK_D
    )

    duplicate = n > block_c
    if not duplicate:
        flag = torch.empty(1, dtype=torch.int32, device=res.device)
        _index_reduce_dup_flag_kernel[(1,)](cnt, flag, out_n, BLOCK=1024)
        duplicate = int(flag.item()) > 1

    rid = _REDUCTIONS[reduce]
    if duplicate:
        # The general scan keeps ~6 BLOCK-wide vectors live inside a dynamic
        # loop; on XPU3 `TritonXPUUnrollControl` rejects BLOCK > 64 (uni_sram).
        dup_block = 64
        _index_reduce_dup_kernel[(triton.cdiv(total, dup_block),)](
            self_w,
            index,
            source,
            res,
            n,
            out_n,
            post,
            pre,
            total,
            REDUCE=rid,
            INCLUDE_SELF=include_self,
            BLOCK=dup_block,
        )
    else:
        _index_reduce_unique_kernel[(triton.cdiv(total, _BLOCK),)](
            self_w,
            source,
            res,
            cnt,
            owner,
            n,
            out_n,
            post,
            pre,
            total,
            REDUCE=rid,
            INCLUDE_SELF=include_self,
            BLOCK=_BLOCK,
        )


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    _validate_args(inp, dim, index, source, reduce)
    if index.numel() == 0:
        return inp

    dim = dim % inp.ndim
    out_n = inp.shape[dim]
    pre = _prod(inp.shape[:dim])
    post = _prod(inp.shape[dim + 1 :])
    total = pre * out_n * post
    nprog = triton.cdiv(total, _BLOCK)
    padded = nprog * _BLOCK
    contig = inp.is_contiguous()

    index_c = index.contiguous()
    source_c = source.contiguous()

    with torch_device_fn.device(inp.device):
        if contig and padded == total:
            res = inp
        else:
            res = torch.empty(padded, dtype=inp.dtype, device=inp.device)
        _launch(
            inp if contig else inp.contiguous(),
            index_c,
            source_c,
            res,
            out_n,
            pre,
            post,
            total,
            reduce,
            include_self,
        )
    if res is not inp:
        inp.copy_(_flat_result_slice(res, total, inp.shape))
    return inp


def index_reduce(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE")
    _validate_args(inp, dim, index, source, reduce)
    return _index_reduce_apply(inp, dim, index, source, reduce, include_self)


def index_reduce_out(inp, dim, index, source, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_OUT")
    if out is None:
        return index_reduce(inp, dim, index, source, reduce, include_self=include_self)
    _validate_args(inp, dim, index, source, reduce)
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {inp.device}, but got {out.device} instead"
        )
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)

    if index.numel() == 0:
        out.copy_(inp)
        return out

    dim = dim % inp.ndim
    out_n = inp.shape[dim]
    pre = _prod(inp.shape[:dim])
    post = _prod(inp.shape[dim + 1 :])
    total = pre * out_n * post
    nprog = triton.cdiv(total, _BLOCK)
    padded = nprog * _BLOCK

    index_c = index.contiguous()
    source_c = source.contiguous()
    self_w = inp if inp.is_contiguous() else inp.contiguous()

    with torch_device_fn.device(inp.device):
        direct = out.is_contiguous() and padded == total
        res = (
            out.view(-1)
            if direct
            else torch.empty(padded, dtype=inp.dtype, device=inp.device)
        )
        _launch(
            self_w,
            index_c,
            source_c,
            res,
            out_n,
            pre,
            post,
            total,
            reduce,
            include_self,
        )
    if not direct:
        out.copy_(_flat_result_slice(res, total, inp.shape))
    return out
