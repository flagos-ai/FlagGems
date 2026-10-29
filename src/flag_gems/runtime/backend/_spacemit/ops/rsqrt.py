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
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# grid from the tuned config's NUM_CTAS (same META mechanism as BLOCK_SIZE)
num_ctas_grid = lambda META: (META["NUM_CTAS"],)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("rsqrt"),
    key=["n_elements"],
)
@triton.jit
def rsqrt_kernel(
    X_ptr,
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

        x_blk = tl.make_block_ptr(
            base=X_ptr,
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

        x = tl.load(x_blk, boundary_check=(0,))
        out = tl.rsqrt(x.to(tl.float32))
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


def rsqrt(A):
    logger.debug("GEMS_SPACEMIT RSQRT")
    A = A.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    with torch_device_fn.device(A.device):
        rsqrt_kernel[num_ctas_grid](A, out, n_elements)
    return out


def rsqrt_(A):
    logger.debug("GEMS_SPACEMIT RSQRT_")
    A_contig = A.contiguous()
    n_elements = A_contig.numel()
    with torch_device_fn.device(A.device):
        rsqrt_kernel[num_ctas_grid](A_contig, A_contig, n_elements)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A
