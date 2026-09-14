import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

# amin mirrors the kunlunxin amax override. The generic ops/amin.py used a
# persisted [BLOCK_M, BLOCK_N] accumulator (`_all`) reduced only at the end;
# combined with the naive_reduction tuner that produced a huge IR dump
# (ir-amin-dev3.log). Here we keep a tiny [BLOCK_M, 1] running accumulator and
# reduce each block along N inside the loop (same shape as min_dim/max_dim), so
# the live state stays small and the loop collapses.


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
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


def heur_m_block_size(args):
    # next_pow2(cdiv(M, 12)) (cluster_num) is the historical heuristic and is
    # intentionally left unbounded. A cap experiment (BLOCK_M <= 64) was tested
    # and REVERTED: it sped up a few fp32 bandwidth shapes ([4096,4096]
    # 264->195us) but regressed every small-work shape ~1.1-4x ((64,1,64)
    # 6.0->23.4us fp32, (1024,1) 8.2->14.9us fp32) because the XPU device
    # timeline pays per-program launch cost, so tiny workloads want FEWER,
    # larger programs while huge workloads want more parallelism -- and the
    # crossover is dtype-dependent. Net dtype-balanced effect was -20%.
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


def heur_need_mask(args):
    # HARNESS 1.4: when every load/store block is provably full (M and N both
    # divide into whole BLOCK_M/BLOCK_N tiles) the row/col masks are all-true at
    # runtime; dropping them lets the XPU backend emit the fast unmasked
    # contiguous-DMA path (isolation, [1024,1048576] fp16 31.1ms -> 6.8ms,
    # [1024,65536] fp32 2.03ms -> 0.88ms, [4096,4096] fp32 454us -> 265us,
    # numerically identical). Any partial tile falls back to the masked path.
    #
    # Boundary: the unmasked path is only a win for reasonably wide tiles.
    # BLOCK_N == 1 (N == 1, e.g. [64,1,64]/[1024,1]) regresses 1.8-4.7x on it,
    # so narrow tiles below 32 stay on the masked path.
    return (
        args["N"] == 0
        or heur_n_block_size(args) < 32
        or args["M"] % heur_m_block_size(args) != 0
        or args["N"] % heur_n_block_size(args) != 0
    )


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
        "NEED_MASK": heur_need_mask,
    },
)
@triton.jit
def amin_kernel(
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

    # Peel the first N-block and derive the running accumulator from real data
    # instead of a `tl.full(..., inf)` loop-carried state. On this XPU backend a
    # loop-carried accumulator initialized by tl.full forces the slow memory
    # path for the whole kernel (isolation: [64,64] BLOCK_M=8 one-iteration
    # 97.8us -> 14.8us; [4096,4096] 1157us -> 456us; [1024,65536] 2409us ->
    # 2034us, all correct). The peeled form reduces to the single-block reduce
    # when N <= BLOCK_N (loop range is empty).
    if NEED_MASK:
        # masked load + other=inf keeps the tail semantics identical to the
        # historical version.
        cols = tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=float("inf")).to(tl.float32)
        a = tl.where(mask, a, float("inf"))
        acc = tl.min(a, axis=1)[:, None]
        for off in range(BLOCK_N, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(inp + cols, mask, other=float("inf")).to(tl.float32)
            a = tl.where(mask, a, float("inf"))
            blk = tl.min(a, axis=1)[:, None]
            acc = tl.minimum(acc, blk)
        tl.store(out, acc, row_mask)
    else:
        # NEED_MASK == False: every tile is a full BLOCK_M x BLOCK_N block, so
        # no mask is needed anywhere (rows < M and cols < N are always true).
        cols = tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.min(a, axis=1)[:, None]
        for off in range(BLOCK_N, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            a = tl.load(inp + cols).to(tl.float32)
            blk = tl.min(a, axis=1)[:, None]
            acc = tl.minimum(acc, blk)
        tl.store(out, acc)


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMIN")
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
                inp, mid, M, block_size, buffer_size_limit=2048
            )
            amin_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid, buffer_size_limit=2048
            )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            amin_kernel[grid](inp, out, M, N, buffer_size_limit=2048)
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
                inp, mid, M, block_size, buffer_size_limit=2048
            )
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)
        inp.copy_(out.reshape(inp.shape) if keepdim else out)
        return inp
    else:
        result = amin(inp, dim=dim, keepdim=True)
        inp.copy_(result)
        return inp
