import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

# amin mirrors the kunlunxin amax override (2026-08-16 performance closure),
# with the sign flipped max -> min everywhere (+inf identity instead of -inf;
# NaN propagation semantics are identical to amax / torch amin). The generic
# ops/amin.py used a persisted [BLOCK_M, BLOCK_N] accumulator (`_all`) reduced
# only at the end; combined with the naive_reduction tuner that produced a huge
# IR dump (ir-amin-dev3.log). Here we keep a tiny [BLOCK_M, 1] running
# accumulator and reduce each block along N inside the loop (same shape as
# min_dim/max_dim), so the live state stays small and the loop collapses.

_FULL_REDUCTION_BLOCK_SIZE = 8192


@libentry()
@triton.jit
def amin_kernel_1(
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
    amin_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


_BLOCK_N_MAX = 8192
# Master-free fast-path tile preferences (XPU sweeps, 2026-08-16, same picks as
# amax):
#   fp16/bf16: [BLOCK_M=128, BLOCK_N=1024] is the sweet spot on every shape;
#   fp32:      [BLOCK_M=64, BLOCK_N=512].
# Small BLOCK_N (<=16) with a long loop is catastrophic (gather addressing), so
# the fast path only triggers when both M and N divide by the picked tiles
# (i.e. the whole reduction is mask-free); everything else keeps the old
# masked path unchanged.
_FAST_BN_FP16 = (1024, 256, 512, 128, 64, 32, 16)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64, 32, 16)
_FAST_BM_FP16 = (128, 64, 32, 256, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 16, 8, 4, 2)


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0, or
    None when no mask-free tile covers this shape."""
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((b for b in bms if M % b == 0), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def amin_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Keep only a [BLOCK_M, 1] running accumulator and reduce each [BLOCK_M,
    # BLOCK_N] block along N *inside* the loop (reduce-INSIDE). This is the ONLY
    # form that is numerically correct on this XPU: the reduce-OUTSIDE variant
    # (persist a [BLOCK_M, BLOCK_N] tile, tl.min once after the loop) miscompiles
    # for bf16 when full blocks are followed by a masked tail (verified for amax;
    # same kernel family). NEED_MASK=False compiles to a mask-free kernel (all
    # tiles full because M and N both divide by the block sizes), avoiding the
    # XPU masked-memory slow path entirely.
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


def _reduce_to_scalar(src, out):
    """Repeatedly reduce `src` with 8192-wide blocks until a scalar remains."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = src.numel()
    data = src
    while n > block:
        mid_size = triton.cdiv(n, block)
        mid = torch.empty((mid_size,), dtype=data.dtype, device=data.device)
        amin_kernel_1[(mid_size, 1)](
            data,
            mid,
            n,
            block,
            n % block != 0,
            buffer_size_limit=2048,
        )
        data = mid
        n = mid_size
    amin_kernel_1[(1, 1)](
        data,
        out,
        n,
        block,
        n % block != 0,
        buffer_size_limit=2048,
    )


def _amin_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel)."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    if numel <= block:
        amin_kernel_1[(1, 1)](
            inp, out, numel, block, numel % block != 0, buffer_size_limit=2048
        )
        return
    # Main pass: treat the flat buffer as rows of `block` elements and reduce
    # each row with the mask-free 2-D tile kernel (few wide programs instead of
    # the old one-program-per-row staging, which launch-bounds at big numel).
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    bm = 128 if not is_fp32 else 64
    if rows % bm != 0:
        bm = next((m for m in _FAST_BM_FP32 if rows % m == 0), 64)
    bn = 1024 if not is_fp32 else 256
    if block % bn != 0:
        bn = next((b for b in _FAST_BN_FP32 if block % b == 0), block)
    mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    with torch_device_fn.device(device):
        amin_kernel_2d[(rows // bm, 1)](
            inp,
            mid,
            rows,
            block,
            bm,
            bn,
            False,
            buffer_size_limit=2048,
        )
        if res:
            # A single masked program handles the (< block) residue.
            amin_kernel_1[(1, 1)](
                inp[rows * block :],
                mid[rows:],
                res,
                block,
                True,
                buffer_size_limit=2048,
            )
        _reduce_to_scalar(mid, out)


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN")
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        numel = inp.numel()
        dtype = inp.dtype
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            _amin_flat(inp, out, inp.device)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if N == 1:
            # Every reduced dim has size 1: amin over it is the identity. Use the
            # native strided-copy engine (flag_gems does not override
            # `_copy_from`) instead of launching a reduction kernel at all.
            out = torch.empty(shape, dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                torch.ops.aten._copy_from(inp, out, False)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out

        # Reorder so the reduced dims are innermost (same order as
        # dim_compress), then make it contiguous with the native strided-copy
        # engine instead of the much slower gems `.contiguous()` override.
        dim_i = inp.dim()
        stride = inp.stride()
        batch_dim = [i for i in range(dim_i) if i not in dim]
        sorted_reduction_dim = sorted(dim, key=lambda x: stride[x], reverse=True)
        order = batch_dim + sorted_reduction_dim
        view = inp.permute(order)
        if view.is_contiguous():
            src = view
        else:
            src = torch.empty(list(view.shape), dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                torch.ops.aten._copy_from(view, src, False)

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        is_fp32 = dtype == torch.float32
        tile = _pick_fast_tile(M, N, is_fp32)
        with torch_device_fn.device(inp.device):
            if tile is not None:
                block_m, block_n = tile
                grid = (triton.cdiv(M, block_m),)
                amin_kernel_2d[grid](
                    src,
                    out,
                    M,
                    N,
                    block_m,
                    block_n,
                    False,
                    buffer_size_limit=2048,
                )
            else:
                block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
                block_m = triton.next_power_of_2(triton.cdiv(M, 12))
                grid = (triton.cdiv(M, block_m),)
                amin_kernel_2d[grid](
                    src,
                    out,
                    M,
                    N,
                    block_m,
                    block_n,
                    True,
                    buffer_size_limit=2048,
                )
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN_")
    if isinstance(dim, int):
        dim = [dim]
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = get_block_size_1d(M, inp.element_size())
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amin_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
                M % block_size != 0,
                buffer_size_limit=2048,
            )
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
        inp.copy_(out.reshape(inp.shape) if keepdim else out)
        return inp
    else:
        result = amin(inp, dim=dim, keepdim=True)
        inp.copy_(result)
        return inp
