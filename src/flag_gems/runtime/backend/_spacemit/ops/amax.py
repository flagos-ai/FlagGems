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

import builtins
import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

from ..heuristics_config_utils import get_num_ctas

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

if os.environ.get("SPINE_TRITON_RPC_HOST"):
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def amax_kernel_1(
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
    amax_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            amax_val = tl.maximum(amax_val, tl.max(inp_val).to(dtype))

    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(mid + pid + lanes, amax_val, mask=lanes < 1)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid + offset, mask=mask, other=min_value)
    amax_val = tl.max(mid_val)
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt
    # cannot parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(out + lanes, amax_val, mask=lanes < 1)


@libentry()
@triton.jit
def amax_kernel_barrier(
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
    amax_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            amax_val = tl.maximum(amax_val, tl.max(inp_val).to(dtype))

    tl.store(mid + pid, amax_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=min_value)
        final_amax_val = tl.max(mid_val).to(dtype)
        tl.store(out, final_amax_val)


@libentry()
@triton.heuristics(
    {
        # one row per program with a full-row 1-D load, mirroring
        # sum_dim_kernel_inner; the previous [BLOCK_M, BLOCK_N] tile reduced
        # with tl.max(axis=1) lowered through a linalg.transpose whose
        # thread-local scratch fails the memref.view space check for fp32
        # ("different memory spaces specified" from spine-opt) -- fp16 only
        # dodged it because the widening cast broke the scratch-to-reduce
        # dataflow. BLOCK_N is floored at 32: horizontal reduces over fewer
        # lanes return garbage on this stack, and masked lanes pad with the
        # dtype minimum (the max identity)
        "BLOCK_N": lambda args: max(32, min(triton.next_power_of_2(args["N"]), 8192)),
    }
)
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    # fp16/bf16: compare in f32 (widening is lossless), but widen the reduced
    # scalar, not the loaded vector -- the vector-wide cast costs ~3x at fp16
    # (2048x2048: 4.42 -> 1.39 ms measured); the 1-D horizontal max is safe in
    # the native dtype (amax_kernel_1 relies on it)
    acc_type = tl.float32 if (dtype is tl.float16 or dtype is tl.bfloat16) else dtype
    load_pad = get_dtype_min(dtype)
    element_ty = out.type.element_ty

    pid = tle.program_id(0)
    amax_val = tl.full((), value=get_dtype_min(acc_type), dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        inp_val = tl.load(inp + pid * N + cols, mask=mask, other=load_pad)
        amax_val = tl.maximum(amax_val, tl.max(inp_val).to(acc_type))
    # bare scalar store lowers to ptr.to_ptr, which this spine-opt cannot
    # parse; write through a lane-0 masked vector store instead
    lanes = tl.arange(0, 32)
    tl.store(out + pid + lanes, amax_val.to(element_ty), mask=lanes < 1)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT AMAX")
    if dim is None or len(dim) == 0:
        inp = inp.contiguous()
        M = inp.numel()
        block_size = builtins.min(4096, triton.next_power_of_2(M))
        block_inner = 256
        num_blocks = triton.cdiv(M, block_size)
        mid_size = min(get_num_ctas(), num_blocks)
        # horizontal reduce over fewer than 32 lanes returns garbage (leftover
        # scalable register lanes); the masked mid load pads with identity
        block_mid = max(triton.next_power_of_2(mid_size), 32)
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
            if (
                alloc_mbarrier is not None
                and release_mbarrier is not None
                and mid_size <= 32767
            ):
                bar = alloc_mbarrier(mid_size)
                try:
                    amax_kernel_barrier[(mid_size,)](
                        inp,
                        mid,
                        out,
                        bar,
                        M,
                        num_blocks,
                        block_size,
                        block_inner,
                        block_mid,
                    )
                finally:
                    release_mbarrier(bar)
            else:
                amax_kernel_1[(mid_size,)](
                    inp, mid, M, num_blocks, block_size, block_inner
                )
                amax_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
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

        # one program per row (sum_dim_kernel_inner's dispatch); this stack's
        # runtime handles multi-thousand-program grids
        grid = (M, 1, 1)
        with torch_device_fn.device(inp.device):
            amax_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
