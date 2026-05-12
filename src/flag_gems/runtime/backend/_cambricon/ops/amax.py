import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils.shape_utils import can_use_int32_index

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def amax_kernel_once(
    inp,
    out,
    M: tl.constexpr,
):
    offset = tl.arange(0, M)
    inp_val = tl.load(inp + offset)
    amax_val = tl.max(inp_val, 0)
    tl.store(out, amax_val)


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def amax_kernel_1(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = -float("inf")
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=-float("inf"))
        (amax_val,) = tl.max(inp_val, 0, return_indices=True)
        if amax_val > _tmp:
            _tmp = amax_val.to(tl.float32)
    tl.atomic_max(out, _tmp)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("amax_opt"), key=["N"])
@triton.jit
def amax_kernel_opt(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_NUM_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # Map the program id to the row of inp it should compute.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_n = pid_n.to(tl.int64)

    num_jobs = tl.num_programs(0)
    rows_per_job = (M + num_jobs - 1) // num_jobs
    row_begin = pid_m * rows_per_job
    row_end = min(row_begin + rows_per_job, M)

    BLOCK_N: tl.constexpr = (N + TILE_NUM_N - 1) // TILE_NUM_N

    for row_idx in range(row_begin, row_end):
        offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        inp_ptrs = inp + row_idx * N + offset_n
        mask = offset_n < N
        inps = tl.load(inp_ptrs, mask, other=-float("inf"))
        (max_val,) = tl.max(inps, 0, return_indices=True)
        new_out = out + row_idx
        tl.atomic_max(new_out, max_val)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("amax"), key=["M", "N"])
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    num_jobs = tl.num_programs(axis=0)
    start_m = pid * BLOCK_M
    step = num_jobs * BLOCK_M
    for off_m in range(start_m, M, step):
        rows = off_m + tl.arange(0, BLOCK_M)[:, None]
        new_inp = inp + rows * N
        new_out = out + rows
        row_mask = rows < M

        _all = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(new_inp + cols, mask, other=-float("inf"))
            _all = tl.maximum(a, _all)

        all = tl.max(_all, axis=1)[:, None]
        tl.store(new_out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS_CAMBRICON AMAX")
    if dim is None or len(dim) == 0:
        M = inp.numel()
        dtype = inp.dtype
        use_int64_index = not can_use_int32_index(inp)

        if M <= 65536:
            if not keepdim:
                out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = list(inp.shape)
                for i in range(0, inp.dim()):
                    shape[i] = 1
                out = torch.empty(shape, dtype=dtype, device=inp.device)
            with torch.cuda.device(inp.device):
                amax_kernel_once[(1, 1, 1)](inp, out, M)
            return out
        else:
            outdtype = torch.float32
            if not keepdim:
                out = torch.full(
                    [], torch.finfo(outdtype).min, dtype=outdtype, device=inp.device
                )
            else:
                shape = list(inp.shape)
                for i in range(0, inp.dim()):
                    shape[i] = 1
                out = torch.full(
                    shape, torch.finfo(outdtype).min, dtype=outdtype, device=inp.device
                )
            grid = lambda meta: (
                min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
            )
            with torch_device_fn.device(inp.device):
                amax_kernel_1[grid](inp, out, M, INT64_INDEX=use_int64_index)
            return out.to(dtype)
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        use_int64_index = not can_use_int32_index(inp)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        with torch_device_fn.device(inp.device):
            if N > 1048576:
                out = torch.empty(shape, dtype=dtype, device=inp.device)
                grid = lambda meta: (
                    min(triton.cdiv(M, meta["BLOCK_M"]), TOTAL_CORE_NUM),
                )
                amax_kernel[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
            else:
                out = torch.full(
                    shape,
                    torch.finfo(torch.float32).min,
                    dtype=torch.float32,
                    device=inp.device,
                )
                grid = lambda meta: (
                    min(triton.cdiv(TOTAL_CORE_NUM, meta["TILE_NUM_N"]), M),
                    meta["TILE_NUM_N"],
                )
                amax_kernel_opt[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out.to(dtype)
