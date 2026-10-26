import logging
import builtins
from functools import reduce

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from ..heuristics_config_utils import HAS_SMT_RVISA, _sr_call, f16, f32, smt_rvisa_enabled
from ..heuristics_config_utils import tle as srv

import os

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

if os.environ.get("SPINE_TRITON_RPC_HOST"):
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        # fp16 add-reduce (tl.sum) returns garbage on this stack; accumulate
        # in fp32, exactly like the sum_dim kernels in this file already do
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    sum_val = tl.zeros((), dtype=cdtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=0).to(cdtype)
            sum_val += tl.sum(inp_val)

    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(mid + pid + lanes, sum_val, mask=lanes < 1)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        # fp16 add-reduce returns garbage on this stack; the host allocates
        # the mid as fp32 for fp16/bf16 inputs, and promoting here covers an
        # explicit dtype=fp16 request too
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(out + lanes, sum_val, mask=lanes < 1)


@libentry()
@triton.jit
def sum_kernel_barrier(
    inp,
    mid,
    out,
    bar,
    M,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        # fp16 add-reduce (tl.sum) returns garbage on this stack; accumulate
        # in fp32, exactly like the sum_dim kernels in this file already do
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    sum_val = tl.zeros((), dtype=cdtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=0).to(cdtype)
            sum_val += tl.sum(inp_val)

    tl.store(mid + pid, sum_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=0).to(cdtype)
        sum_val = tl.sum(mid_val)
        tl.store(out, sum_val)


def sum(inp, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM")
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = builtins.min(4096, triton.next_power_of_2(M))
    block_inner = 256
    num_blocks = triton.cdiv(M, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    # horizontal reduce over fewer than 32 lanes returns garbage (leftover
    # scalable register lanes); the masked mid load pads with identity
    block_mid = max(triton.next_power_of_2(mid_size), 32)

    # fp16 add-reduce is broken on this stack (see kernels); accumulate fp16
    # (and bf16) in a float32 mid, the final store casts back to `dtype`
    mid_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    mid = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                sum_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            sum_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
            sum_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_OUT")
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = builtins.min(4096, triton.next_power_of_2(M))
    block_inner = 256
    num_blocks = triton.cdiv(M, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    block_mid = triton.next_power_of_2(mid_size)

    # same fp32-mid accumulation as sum() above
    mid_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    mid = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                sum_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            sum_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
            sum_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def sum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def sum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0)
        tl.store(output_ptr + pid_m, out)
    else:
        sum = tl.zeros([TILE_N], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0)
        tl.store(output_ptr + pid_m, out)


@libentry()
@triton.heuristics(
    {
        # naive_reduction's tune config was static (BLOCK_M=2, BLOCK_N=512);
        # both are made adaptive: BLOCK_M keeps the grid within the 512
        # programs this stack's runtime executes normally, and BLOCK_N covers
        # a whole row so the N-loop collapses to one iteration -- a
        # loop-carried accumulator feeding the next iteration lowers to
        # per-iteration DDR scratch traffic (~100x wall time at N=2048)
        "BLOCK_M": lambda args: max(2, triton.next_power_of_2(triton.cdiv(args["M"], 512))),
        "BLOCK_N": lambda args: min(triton.next_power_of_2(args["N"]), 8192),
    }
)
@triton.jit
def sum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim is None:
        result = torch.sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1:
        dim = dim[0]
        N = inp.shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N
        shape[dim] = 1
        _out_provided = out is not None
        if _out_provided:
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_(shape[:dim] + shape[dim + 1 :])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                sum_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                if (
                    smt_rvisa_enabled()
                    and inp.dtype in (torch.float16, torch.float32)
                    and M > 0
                    and N > 0
                ):
                    scratch = torch.empty(M, dtype=torch.float32, device=inp.device)
                    block = builtins.max(1, triton.cdiv(M, 512))
                    with torch_device_fn.device(inp.device):
                        _sum_dim1_rvisa_host[(triton.cdiv(M, block),)](
                            inp,
                            scratch,
                            M,
                            N,
                            BLOCK=block,
                            F16=inp.dtype == torch.float16,
                        )
                    # out still carries the reduced dim as 1 (the squeeze happens
                    # below); reshape so copy_ does not try to broadcast
                    out.copy_(scratch.reshape(out.shape))
                else:
                    # one program per row; this stack's runtime dispatches large
                    # grids fine (per-row single-tile loads benchmarked faster
                    # than re-blocking through sum_dim_kernel)
                    grid = (M, 1, 1)
                    sum_dim_kernel_inner[grid](
                        out,
                        inp,
                        M,
                        N,
                    )
        if not keepdim and not _out_provided:
            out = out.squeeze(dim=dim)
        return out
    else:
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        _out_provided = out is not None
        if _out_provided:
            dim_set = set(dim)
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_([s for i, s in enumerate(shape) if i not in dim_set])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N)
        if not keepdim and not _out_provided:
            for d in sorted(dim, reverse=True):
                out = out.squeeze(dim=d)
        return out


if HAS_SMT_RVISA:

    @srv.raw_kernel
    def _sum_dim1_rvisa_f32(
        X: srv.mem(f32),
        out: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        # streaming-VL row reduce: whole rows accumulated in registers with
        # one horizontal vreduce at the end; also sidesteps the ptr.to_ptr
        # compile failure of sum_dim_kernel_inner on stacks whose spine-opt
        # lacks MemorySpaceAttrInterface. Ported from spine-triton
        # python/tests/raw/test_raw_sum.py with the mv svector row-block host
        # (BLOCK rows per program keeps the grid within the 512 programs the
        # runtime executes).
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            acc = srv.vzero(f32)
            for i in srv.range(0, Nfloor, nvl):
                vx = srv.vload(X, base + i, dtype=f32)
                acc = acc + vx
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tx = srv.vload(X, base + i, dtype=f32)
                acc = acc + tx
            srv.sstore(out, row, srv.vreduce_sum(acc))

    @srv.raw_kernel
    def _sum_dim1_rvisa_f16(
        X: srv.mem(f16),
        out: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            acc = srv.vzero(f32)
            for i in srv.range(0, Nfloor, nvl):
                vx = srv.vload(X, base + i)  # default dtype=f16
                acc = acc + srv.cast(vx, f32)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tx = srv.vload(X, base + i)
                acc = acc + srv.cast(tx, f32)
            srv.sstore(out, row, srv.vreduce_sum(acc))

    @triton.jit(do_not_specialize=["M", "N"])
    def _sum_dim1_rvisa_host(X, out, M, N, BLOCK: tl.constexpr, F16: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, M)
        if F16:
            _sr_call(_sum_dim1_rvisa_f16, outputs=[], inputs=[X, out, N, row_base, row_end])
        else:
            _sr_call(_sum_dim1_rvisa_f32, outputs=[], inputs=[X, out, N, row_base, row_end])


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT SUM_DIM")
    if inp.numel() == 0:
        if dtype is None:
            dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

        out_shape = list(inp.shape)
        if dim is None:
            if keepdim:
                out_shape = [1] * len(out_shape)
            else:
                out_shape = []
        elif isinstance(dim, (list, tuple)) and len(dim) == 0:
            if keepdim:
                out_shape = [1] * len(out_shape)
            else:
                out_shape = []
        else:
            dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims_to_reduce:
                    out_shape[d % inp.ndim] = 1
            else:
                sorted_dims_to_remove = sorted(
                    dims_to_reduce, key=lambda x: x % inp.ndim, reverse=True
                )
                for d in sorted_dims_to_remove:
                    index_to_remove = d % inp.ndim
                    out_shape.pop(index_to_remove)
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        out.zero_()
        return out
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_SPACEMIT SUM_DIM_OUT")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)