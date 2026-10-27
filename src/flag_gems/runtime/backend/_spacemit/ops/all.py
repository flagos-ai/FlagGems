import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

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


@triton.jit
def reduce_all(a, b):
    return a | b


@triton.jit
def reduce_count(a, b):
    return a + b


@libentry()
@triton.heuristics(
    {
        # naive_reduction's tune config was static (BLOCK_M=2, BLOCK_N=512);
        # both are made adaptive: BLOCK_M keeps the grid within the 512
        # programs this stack's runtime executes normally, and BLOCK_N covers
        # a whole row so the N-loop collapses to one iteration -- a
        # loop-carried accumulator feeding the next iteration lowers to
        # per-iteration DDR scratch traffic (~100x wall time at N=2048)
        "BLOCK_M": lambda args: max(
            2, triton.next_power_of_2(triton.cdiv(args["M"], 512))
        ),
        "BLOCK_N": lambda args: min(triton.next_power_of_2(args["N"]), 8192),
    }
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
    row_start = tl.program_id(0) * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)
    for mi in range(row_start, row_end, 1):
        has_zero_scalar = tl.zeros([], dtype=tl.int1)
        row_inp = inp + mi * N
        row_out = out + mi
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            mask = cols < N
            a = tl.load(row_inp + cols, mask, other=1.0)
            has_zero_block = a == 0
            has_zero_scalar = has_zero_scalar | tl.reduce(
                has_zero_block, axis=0, combine_fn=reduce_all
            )
        all_val = has_zero_scalar == 0
        tl.store(row_out, all_val.to(tl.int8))


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)
    count = tl.full((), value=0, dtype=tl.int32)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = min(n_start + BLOCK_SIZE, n_elements)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < n_elements
            inp_val = tl.load(inp + offset, mask=mask, other=1.0)
            nonzero = tl.where(mask & (inp_val != 0), 1, 0)
            count_val = tl.reduce(nonzero, axis=0, combine_fn=reduce_count)
            count += count_val

    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(mid + pid + lanes, count, mask=lanes < 1)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, n_elements, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=0)
    count = tl.reduce(mid_val, axis=0, combine_fn=reduce_count)
    all_val = count == n_elements
    # stores to bool/i8 pointers are broken on this spine-opt as well (they
    # reinterpret-cast through ptr.to_ptr); the host passes an int32 scalar
    # here and casts to bool after the launch
    lanes = tl.arange(0, 32)
    tl.store(out + lanes, all_val.to(tl.int32), mask=lanes < 1)


@libentry()
@triton.jit
def all_kernel_barrier(
    inp,
    mid,
    out,
    bar,
    n_elements,
    NUM_BLOCKS,
    MID_SIZE,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)
    count = tl.full((), value=0, dtype=tl.int32)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = min(n_start + BLOCK_SIZE, n_elements)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < n_elements
            inp_val = tl.load(inp + offset, mask=mask, other=1.0)
            nonzero = tl.where(mask & (inp_val != 0), 1, 0)
            count_val = tl.reduce(nonzero, axis=0, combine_fn=reduce_count)
            count += count_val

    tl.store(mid + pid, count)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < MID_SIZE
        mid_val = tl.load(mid + offset, mask=mask, other=0)
        count = tl.reduce(mid_val, axis=0, combine_fn=reduce_count)
        all_val = count == n_elements
        tl.store(out, all_val.to(tl.int32))


def all(inp):
    logger.debug("GEMS_SPACEMIT ALL")
    inp = inp.contiguous()
    n_elements = inp.numel()

    # Larger block_size than original sqrt-based (1024->4096) to reduce programs
    block_size = min(4096, triton.next_power_of_2(n_elements))
    block_inner = 256
    num_blocks = triton.cdiv(n_elements, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    # horizontal reduce over fewer than 32 lanes returns garbage (leftover
    # scalable register lanes); the masked mid load pads with identity
    block_mid = max(triton.next_power_of_2(mid_size), 32)

    mid = torch.empty((mid_size,), dtype=torch.int32, device=inp.device)
    # this stack cannot store to bool tensors from kernels (i8 pointer stores
    # reinterpret-cast through ptr.to_ptr); reduce into int32 and cast here
    out = torch.empty([], dtype=torch.int32, device=inp.device)

    with torch_device_fn.device(inp.device):
        if (
            alloc_mbarrier is not None
            and release_mbarrier is not None
            and mid_size <= 32767
        ):
            bar = alloc_mbarrier(mid_size)
            try:
                all_kernel_barrier[(mid_size,)](
                    inp,
                    mid,
                    out,
                    bar,
                    n_elements,
                    num_blocks,
                    mid_size,
                    block_size,
                    block_inner,
                    block_mid,
                )
            finally:
                release_mbarrier(bar)
        else:
            all_kernel_1[(mid_size,)](
                inp, mid, n_elements, num_blocks, block_size, block_inner
            )
            all_kernel_2[(1, 1)](mid, out, mid_size, n_elements, block_mid)
    # out.to(torch.bool) under full use_gems() registration dispatches to the
    # general copy kernel, whose 0-d path is the scalar-store/ptr.to_ptr
    # compile failure (the rerun3/4 test_all_perf flakes); pull the scalar to
    # the host and rebuild the bool tensor instead
    return torch.tensor(bool(out.item()), device=inp.device)


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT ALL_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        inp = inp.contiguous()
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N
        out = torch.empty(shape, dtype=torch.bool, device=inp.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            all_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT ALL_DIMS")
    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    inp = inp.contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N
    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
