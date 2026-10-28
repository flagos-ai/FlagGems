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
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# grid from the tuned config's NUM_CTAS (same META mechanism as BLOCK_SIZE)
num_ctas_grid = lambda META: (META["NUM_CTAS"],)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lt_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def lt_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        b = tl.load(b_blk, boundary_check=(0,))
        out = tl.where(a < b, 1, 0).to(Out_ptr.type.element_ty)
        tl.store(out_blk, out, boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lt_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def lt_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        out = tl.where(a < scalar, 1, 0).to(Out_ptr.type.element_ty)
        tl.store(out_blk, out, boundary_check=(0,))


def lt(A, B):
    logger.debug("GEMS_SPACEMIT LT")
    A, B = A.contiguous(), B.contiguous()
    out = torch.empty(A.shape, dtype=torch.uint8, device=A.device)
    n = A.numel()
    lt_kernel_tt[num_ctas_grid](A, B, out, n)
    return out.view(torch.bool)


def lt_scalar(A, B):
    logger.debug("GEMS_SPACEMIT LT_SCALAR")
    A = A.contiguous()
    out = torch.empty(A.shape, dtype=torch.uint8, device=A.device)
    n = A.numel()
    lt_kernel_ts[num_ctas_grid](A, out, float(B), n)
    return out.view(torch.bool)
