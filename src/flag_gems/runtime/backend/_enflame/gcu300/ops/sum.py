# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["M"])
def sum_kernel_1(
    inp,
    mid,
    M,
    GRID_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    acc = tl.zeros((), dtype=cdtype)
    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    in_block = tl.make_block_ptr(
        base=inp,
        shape=(M,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in tl.range(pid, num_blocks, GRID_DIM):
        inp_val = tl.load(in_block, boundary_check=(0,), padding_option="zero").to(
            cdtype
        )
        acc += tl.sum(inp_val)
        in_block = tl.advance(in_block, (GRID_DIM * BLOCK_SIZE,))
    tl.store(mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["M"])
def sum_kernel_1_dual(
    inp,
    mid,
    M,
    GRID_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    # Dual accumulators: break serial dependency chain between iterations.
    # acc0 and acc1 are independent, allowing load_1 to overlap with
    # reduction of load_0 within the same SIP's warp scheduler.
    acc0 = tl.zeros((), dtype=cdtype)
    acc1 = tl.zeros((), dtype=cdtype)
    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    in_block_0 = tl.make_block_ptr(
        base=inp,
        shape=(M,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    in_block_1 = tl.make_block_ptr(
        base=inp,
        shape=(M,),
        strides=(1,),
        offsets=((pid + GRID_DIM) * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    stride = GRID_DIM * 2 * BLOCK_SIZE
    for _ in tl.range(pid, num_blocks, GRID_DIM * 2):
        inp_val_0 = tl.load(in_block_0, boundary_check=(0,), padding_option="zero").to(
            cdtype
        )
        inp_val_1 = tl.load(in_block_1, boundary_check=(0,), padding_option="zero").to(
            cdtype
        )
        acc0 += tl.sum(inp_val_0)
        acc1 += tl.sum(inp_val_1)
        in_block_0 = tl.advance(in_block_0, (stride,))
        in_block_1 = tl.advance(in_block_1, (stride,))
    tl.store(mid + pid, acc0 + acc1)


@libentry()
@triton.jit(do_not_specialize=["M"])
def sum_atomic_kernel(
    inp,
    out,
    M,
    GRID_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    acc = tl.zeros((), dtype=cdtype)
    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    in_block = tl.make_block_ptr(
        base=inp,
        shape=(M,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in tl.range(pid, num_blocks, GRID_DIM):
        inp_val = tl.load(in_block, boundary_check=(0,), padding_option="zero").to(
            cdtype
        )
        acc += tl.sum(inp_val)
        in_block = tl.advance(in_block, (GRID_DIM * BLOCK_SIZE,))
    tl.atomic_add(out, acc, sem="relaxed")


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) | tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    # grid limit
    if BLOCK_M < 64:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) | tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


def sum(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    GRID_DIM = 6
    BLOCK_SIZE = 32768
    DUAL_THRESHOLD = 4 * 1024 * 1024  # 4M elements
    kernel = sum_kernel_1_dual if M > DUAL_THRESHOLD else sum_kernel_1
    mid = torch.empty((GRID_DIM,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        kernel[(GRID_DIM,)](
            inp,
            mid,
            M,
            GRID_DIM=GRID_DIM,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        sum_kernel_2[(1,)](mid, out, GRID_DIM, triton.next_power_of_2(GRID_DIM))
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_OUT")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32
    if M == 0:
        out.resize_([])
        out.zero_()
        return out

    GRID_DIM = 6
    BLOCK_SIZE = 32768
    DUAL_THRESHOLD = 4 * 1024 * 1024  # 4M elements
    kernel = sum_kernel_1_dual if M > DUAL_THRESHOLD else sum_kernel_1
    mid = torch.empty((GRID_DIM,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        kernel[(GRID_DIM,)](
            inp,
            mid,
            M,
            GRID_DIM=GRID_DIM,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        sum_kernel_2[(1,)](mid, out, GRID_DIM, triton.next_power_of_2(GRID_DIM))
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM_DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int32

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is not None:
            dims = dim if isinstance(dim, (list, tuple)) else [dim]
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
            out_shape = [1] * inp.ndim if keepdim else []
        else:
            if keepdim:
                for d in dims:
                    out_shape[d % inp.ndim] = 1
            else:
                for d in sorted([d % inp.ndim for d in dims], reverse=True):
                    out_shape.pop(d)
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        out.zero_()
        return out

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

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
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_DIM_OUT")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int32

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out.resize_(shape)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out.squeeze_(dim=dim)
    return out
