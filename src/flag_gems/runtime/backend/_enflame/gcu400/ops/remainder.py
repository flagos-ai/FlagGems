import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
MAX_GRID = 48


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_tt_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        y = tl.load(y_ptr + off, mask=mask)
        r = x % y
        c1 = r != 0
        c2 = (x < 0) ^ (y < 0)
        out = tl.where(c1 & c2, r + y, r)
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_ts_kernel(x_ptr, y_scalar, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        r = x % y_scalar
        c1 = r != 0
        c2 = (x < 0) ^ (y_scalar < 0)
        out = tl.where(c1 & c2, r + y_scalar, r)
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_st_kernel(x_scalar, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        y = tl.load(y_ptr + off, mask=mask)
        r = x_scalar % y
        c1 = r != 0
        c2 = (x_scalar < 0) ^ (y < 0)
        out = tl.where(c1 & c2, r + y, r)
        tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N, dtype):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    if elem_size <= 2:
        return 4096
    return 65536


def _launch_tt(inp_a, inp_b, out, N, dtype):
    BLOCK = _choose_block(N, dtype)
    grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
    with torch_device_fn.device(inp_a.device):
        remainder_tt_kernel[(grid_size,)](
            inp_a,
            inp_b,
            out,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )


def remainder(A, B):
    logger.debug("GEMS REMAINDER GCU400")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        inp_a = A if A.is_contiguous() else A.contiguous()
        inp_b = B if B.is_contiguous() else B.contiguous()
        if inp_a.shape != inp_b.shape:
            shape = torch.broadcast_shapes(inp_a.shape, inp_b.shape)
            inp_a = inp_a.expand(shape).contiguous()
            inp_b = inp_b.expand(shape).contiguous()
        N = inp_a.numel()
        out = torch.empty_like(inp_a)
        _launch_tt(inp_a, inp_b, out, N, inp_a.dtype)
        return out
    elif isinstance(A, torch.Tensor):
        inp = A if A.is_contiguous() else A.contiguous()
        N = inp.numel()
        out = torch.empty_like(inp)
        BLOCK = _choose_block(N, inp.dtype)
        grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
        with torch_device_fn.device(inp.device):
            remainder_ts_kernel[(grid_size,)](
                inp,
                B,
                out,
                N,
                BLOCK=BLOCK,
                num_warps=4,
            )
        return out
    elif isinstance(B, torch.Tensor):
        inp = B if B.is_contiguous() else B.contiguous()
        N = inp.numel()
        out = torch.empty_like(inp)
        BLOCK = _choose_block(N, inp.dtype)
        grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
        with torch_device_fn.device(inp.device):
            remainder_st_kernel[(grid_size,)](
                A,
                inp,
                out,
                N,
                BLOCK=BLOCK,
                num_warps=4,
            )
        return out
    else:
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS REMAINDER_ GCU400")
    inp_a = A if A.is_contiguous() else A.contiguous()
    N = inp_a.numel()
    if isinstance(B, torch.Tensor):
        inp_b = B if B.is_contiguous() else B.contiguous()
        if inp_a.shape != inp_b.shape:
            shape = torch.broadcast_shapes(inp_a.shape, inp_b.shape)
            inp_b = inp_b.expand(shape).contiguous()
        BLOCK = _choose_block(N, inp_a.dtype)
        grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
        with torch_device_fn.device(inp_a.device):
            remainder_tt_kernel[(grid_size,)](
                inp_a,
                inp_b,
                A,
                N,
                BLOCK=BLOCK,
                num_warps=4,
            )
    else:
        BLOCK = _choose_block(N, inp_a.dtype)
        grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
        with torch_device_fn.device(inp_a.device):
            remainder_ts_kernel[(grid_size,)](
                inp_a,
                B,
                A,
                N,
                BLOCK=BLOCK,
                num_warps=4,
            )
    return A
