import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from ..heuristics_config_utils import HAS_SMT_RVISA, _sr_call, f16, f32, smt_rvisa_enabled
from ..heuristics_config_utils import tle as srv


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # Vectorized load with block_ptr for better spine-triton codegen
    pid = tle.program_id(0)
    offset_start = (pid * BLOCK_SIZE).to(tl.int32)

    inp_ptr = tl.make_block_ptr(
        base=inp,
        shape=[M],
        strides=[1],
        offsets=[offset_start],
        block_shape=[BLOCK_SIZE],
        order=[0],
    )

    v = tl.load(inp_ptr, boundary_check=[0], padding_option="zero").to(tl.float32)
    sum_val = tl.sum(v, axis=0)
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(mid + pid + lanes, sum_val, mask=lanes < 1)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    mid_ptr = tl.make_block_ptr(
        base=mid,
        shape=[MID_SIZE],
        strides=[1],
        offsets=[0],
        block_shape=[BLOCK_MID],
        order=[0],
    )
    mid_val = tl.load(mid_ptr, boundary_check=[0], padding_option="zero").to(tl.float32)
    sum_val = tl.sum(mid_val, axis=0) / M
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(out + lanes, sum_val.to(out.dtype.element_ty), mask=lanes < 1)


def mean(inp, *, dtype=None):
    logging.debug("GEMS_SPACEMIT MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    # Use gems' standard two-stage design: many programs * BLOCK_SIZE each
    # Gems baseline strategy: block_size = sqrt(M), mid_size programs
    import math
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    # this stack's runtime silently drops programs beyond a 512-program grid
    # (uninitialised mid -> NaN); re-block so kernel_1 never exceeds 512
    if mid_size > 512:
        mid_size = 512
        block_size = triton.next_power_of_2(triton.cdiv(M, mid_size))
    # horizontal reduce over fewer than 32 lanes returns garbage (leftover
    # scalable register lanes); the masked mid load pads with identity
    block_mid = max(triton.next_power_of_2(mid_size), 32)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        # Stage 1: parallel reduction across mid_size programs
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        # Stage 2: final reduction in single program (fully on device)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.heuristics(
    {
        # mean_spacemit_v1's tune config was static (BLOCK_M=1, BLOCK_N=256);
        # both are made adaptive: BLOCK_N covers a whole row so the N-loop
        # collapses to one iteration (a loop-carried accumulator feeding the
        # next iteration lowers to per-iteration DDR scratch traffic), and
        # BLOCK_M stays at one row per program for all but huge M (this
        # stack's runtime dispatches multi-thousand-program grids fine, and
        # the per-row shape benchmarked faster than wide [BLOCK_M, BLOCK_N]
        # tiles)
        "BLOCK_M": lambda args: max(1, triton.next_power_of_2(triton.cdiv(args["M"], 4096))),
        "BLOCK_N": lambda args: min(triton.next_power_of_2(args["N"]), 8192),
    }
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_start = tl.program_id(0) * BLOCK_M
    row_offset = tl.arange(0, BLOCK_M)
    row_idx = row_start + row_offset
    row_mask = row_idx < M

    _mean_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[M, N],
        strides=[N, 1],
        offsets=[row_start, 0],
        block_shape=[BLOCK_M, BLOCK_N],
        order=[1, 0],
    )

    for off_n in range(0, num_pid_n):
        a = tl.load(x_ptr_desc, boundary_check=[0, 1]).to(tl.float32)
        _mean_acc += a
        x_ptr_desc = tl.advance(x_ptr_desc, [0, BLOCK_N])

    mean = tl.sum(_mean_acc, axis=1) / N

    mean_ptr_desc = tl.make_block_ptr(
        base=Mean,
        shape=[M],
        strides=[1],
        offsets=[row_start],
        block_shape=[BLOCK_M],
        order=[0],
    )
    tl.store(mean_ptr_desc, mean.to(Mean.dtype.element_ty), boundary_check=[0])


if HAS_SMT_RVISA:

    @srv.raw_kernel
    def _mean_dim1_rvisa_f32(
        X: srv.mem(f32),
        out: srv.mem(f32, out=True),
        N: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        # streaming-VL row reduce: accumulate whole rows in registers with one
        # horizontal vreduce at the end, instead of a [BLOCK_M, BLOCK_N] tile
        # accumulator; 3.6-7.0x over mean_dim_kernel at f32 (K3 raw_vs_ops
        # bench, 2026-10-23). Ported from spine-triton
        # python/tests/raw/test_raw_mean_dim.py with the mv svector row-block
        # host (BLOCK rows per program keeps the grid within the 512 programs
        # the runtime executes).
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
            srv.sstore(out, row, srv.vreduce_sum(acc) / N)

    @srv.raw_kernel
    def _mean_dim1_rvisa_f16(
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
            srv.sstore(out, row, srv.vreduce_sum(acc) / N)

    @triton.jit(do_not_specialize=["M", "N"])
    def _mean_dim1_rvisa_host(X, out, M, N, BLOCK: tl.constexpr, F16: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, M)
        if F16:
            _sr_call(_mean_dim1_rvisa_f16, outputs=[], inputs=[X, out, N, row_base, row_end])
        else:
            _sr_call(_mean_dim1_rvisa_f32, outputs=[], inputs=[X, out, N, row_base, row_end])

    def _mean_dim_rvisa(x, out, M, N):
        x = x.contiguous()
        scratch = torch.empty(M, dtype=torch.float32, device=x.device)
        block = max(1, triton.cdiv(M, 512))
        with torch_device_fn.device(x.device):
            _mean_dim1_rvisa_host[(triton.cdiv(M, block),)](
                x, scratch, M, N, BLOCK=block, F16=x.dtype == torch.float16
            )
        # out still carries the reduced dim as 1 (the squeeze happens in the
        # caller); reshape so copy_ does not try to broadcast
        out.copy_(scratch.reshape(out.shape))


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS_SPACEMIT MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    if (
        smt_rvisa_enabled()
        and x.dtype in (torch.float16, torch.float32)
        and M > 0
        and N > 0
    ):
        _mean_dim_rvisa(x, out, M, N)
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def avg_pool2d(x, kernel_size=None, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return mean_dim(x, dim=[2, 3], keepdim=True)
