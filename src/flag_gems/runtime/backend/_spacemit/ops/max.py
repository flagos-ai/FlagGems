import logging
import builtins
from collections import namedtuple

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

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
def max_kernel_1(
    inp,
    mid,
    M,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            max_val = tl.maximum(max_val, tl.max(inp_val).to(dtype))

    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(mid + pid + lanes, max_val, mask=lanes < 1)


@libentry()
@triton.jit
def max_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid + offset, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(out + lanes, max_val, mask=lanes < 1)


@libentry()
@triton.jit
def max_kernel_barrier(
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
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            max_val = tl.maximum(max_val, tl.max(inp_val).to(dtype))

    tl.store(mid + pid, max_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=min_value)
        final_max_val = tl.max(mid_val).to(dtype)
        tl.store(out, final_max_val)


@libentry()
@triton.heuristics(
    {
        # naive_reduction's tune config was static (BLOCK_M=2, BLOCK_N=512);
        # both are made adaptive: BLOCK_M keeps the grid within the 512
        # programs this stack's runtime executes normally, and BLOCK_N covers
        # a whole row so the N-loop collapses to one iteration -- a
        # loop-carried accumulator feeding the next iteration lowers to
        # per-iteration DDR scratch traffic (~100x wall time at N=2048)
        # builtins.max: this module's own max() shadows the builtin
        "BLOCK_M": lambda args: builtins.max(
            2, triton.next_power_of_2(triton.cdiv(args["M"], 512))
        ),
        "BLOCK_N": lambda args: min(triton.next_power_of_2(args["N"]), 8192),
    }
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    # fp16/bf16 max-with-indices lowers badly on this stack; compare in f32
    # (widening is lossless, so values and indices are unchanged)
    acc_type = tl.float32 if (dtype is tl.float16 or dtype is tl.bfloat16) else dtype
    load_pad = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=get_dtype_min(acc_type), dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_vals = tl.load(inp + offset, mask=mask, other=load_pad).to(acc_type)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    tl.store(out_value + m_offset, result_value.to(out_value.type.element_ty), mask=mask1)
    tl.store(out_index + m_offset, result_index, mask=mask1)


def max(inp):
    logger.debug("GEMS_SPACEMIT MAX")
    inp = inp.contiguous()
    M = inp.numel()
    block_size = builtins.min(4096, triton.next_power_of_2(M))
    block_inner = 256
    num_blocks = triton.cdiv(M, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    # horizontal reduce over fewer than 32 lanes returns garbage (leftover
    # scalable register lanes); the masked mid load pads with identity
    # (builtins.max: this module's own max() shadows the builtin)
    block_mid = builtins.max(triton.next_power_of_2(mid_size), 32)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                max_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            max_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
            max_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


if HAS_SMT_RVISA:

    # sentinel for "not the argmax lane" in the f32 index scratch; any value
    # larger than any real index works (real indices are < 2**24, exact in f32)
    _RVISA_INF_IDX = 1.0e30

    @srv.raw_kernel
    def _max_dim1_rvisa_f32(
        X: srv.mem(f32),
        vals: srv.mem(f32, out=True),
        idxs: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        # streaming-VL argmax: running select-based (value, index) pair per
        # lane, one horizontal vreduce_max + vreduce_min at the end; 5.0-9.2x
        # over max_kernel at f32 (K3 raw_vs_ops bench, 2026-10-23). Ported
        # from spine-triton python/tests/raw/test_raw_max_dim.py with the mv
        # svector row-block host (BLOCK rows per program keeps the grid within
        # the 512 programs the runtime executes).
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        lane = srv.viota()
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            best_val = srv.vload(X, base, dtype=f32)
            best_idx = lane
            for i in srv.range(0, Nfloor, nvl):
                vx = srv.vload(X, base + i, dtype=f32)
                idx = lane + srv.cast(i, f32)
                gt = vx > best_val
                best_val = srv.select(gt, vx, best_val)
                best_idx = srv.select(gt, idx, best_idx)
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tx = srv.vload(X, base + i, dtype=f32, fill=-1e38)
                tidx = lane + srv.cast(i, f32)
                gt2 = tx > best_val
                best_val = srv.select(gt2, tx, best_val)
                best_idx = srv.select(gt2, tidx, best_idx)
            gmax = srv.vreduce_max(best_val)
            is_max = best_val >= gmax
            big = srv.vzero(f32) + _RVISA_INF_IDX
            masked = srv.select(is_max, best_idx, big)
            argmax = srv.vreduce_min(masked)
            srv.sstore(vals, row, gmax)
            srv.sstore(idxs, row, argmax)

    @srv.raw_kernel
    def _max_dim1_rvisa_f16(
        X: srv.mem(f16),
        vals: srv.mem(f32, out=True),
        idxs: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        nvl = srv.vconfig(-1, 1)
        Nfloor = (N // nvl) * nvl
        lane = srv.viota()
        for row in srv.range(row_base, row_end, 1):
            base = row * N
            # widen to f32 for the compare (fp16 compare-with-index lowers
            # badly on this stack; the widening is lossless)
            best_val = srv.cast(srv.vload(X, base), f32)
            best_idx = lane
            for i in srv.range(0, Nfloor, nvl):
                vx = srv.cast(srv.vload(X, base + i), f32)
                idx = lane + srv.cast(i, f32)
                gt = vx > best_val
                best_val = srv.select(gt, vx, best_val)
                best_idx = srv.select(gt, idx, best_idx)
            # -65504 = lowest finite f16, so padded lanes can never win the max
            for i in srv.range(Nfloor, N, nvl):
                srv.vconfig(N - i, 1)
                tx = srv.cast(srv.vload(X, base + i, fill=-65504.0), f32)
                tidx = lane + srv.cast(i, f32)
                gt2 = tx > best_val
                best_val = srv.select(gt2, tx, best_val)
                best_idx = srv.select(gt2, tidx, best_idx)
            gmax = srv.vreduce_max(best_val)
            is_max = best_val >= gmax
            big = srv.vzero(f32) + _RVISA_INF_IDX
            masked = srv.select(is_max, best_idx, big)
            argmax = srv.vreduce_min(masked)
            srv.sstore(vals, row, gmax)
            srv.sstore(idxs, row, argmax)

    @triton.jit(do_not_specialize=["M", "N"])
    def _max_dim1_rvisa_host(X, vals, idxs, M, N, BLOCK: tl.constexpr, F16: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, M)
        if F16:
            _sr_call(_max_dim1_rvisa_f16, outputs=[], inputs=[X, vals, idxs, N, row_base, row_end])
        else:
            _sr_call(_max_dim1_rvisa_f32, outputs=[], inputs=[X, vals, idxs, N, row_base, row_end])


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    # the streaming kernel seeds its max with one full first tile, so it needs
    # N >= 64 (the widest VLMAX) to not read into the next row
    if (
        smt_rvisa_enabled()
        and inp.dtype in (torch.float16, torch.float32)
        and M > 0
        and N >= 64
    ):
        inp = inp.contiguous()
        vals = torch.empty(M, dtype=torch.float32, device=inp.device)
        idxs = torch.empty(M, dtype=torch.float32, device=inp.device)
        block = builtins.max(1, triton.cdiv(M, 512))
        with torch_device_fn.device(inp.device):
            _max_dim1_rvisa_host[(triton.cdiv(M, block),)](
                inp, vals, idxs, M, N, BLOCK=block, F16=inp.dtype == torch.float16
            )
        # outs still carry the reduced dim as 1 (the squeeze happens in the
        # caller); reshape so copy_ does not try to broadcast
        out_value.copy_(vals.reshape(out_value.shape))
        out_index.copy_(idxs.round().to(torch.int64).reshape(out_index.shape))
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            max_kernel[grid](inp, out_value, out_index, M, N)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out