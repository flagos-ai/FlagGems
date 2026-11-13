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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
NUM_SIPS = 24


# ---------- Optimized contiguous kernel ----------
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def floor_kernel(
    in_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        in_blk = tl.make_block_ptr(
            base=in_ptr,
            shape=(N_total,),
            strides=(1,),
            offsets=(block_id * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=out_ptr,
            shape=(N_total,),
            strides=(1,),
            offsets=(block_id * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(in_blk, boundary_check=(0,))
        out = x
        if IS_FP32:
            out = tl.floor(x.to(tl.float32))
        elif IS_FP16:
            out = tl.cast(tl.floor(tl.cast(x, tl.float32)), tl.float16)
        elif IS_BF16:
            out = tl.cast(tl.floor(tl.cast(x, tl.float32)), tl.bfloat16)
        tl.store(out_blk, out, boundary_check=(0,))


# ---------- Generic fallback (non-contiguous) ----------
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_func(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def _floor_contiguous(inp, out):
    """Optimized path for contiguous tensors using Fixed Grid + block_ptr DMA."""
    N = inp.numel()
    dtype = inp.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS)
    with torch_device_fn.device(inp.device):
        floor_kernel[(grid,)](
            inp,
            out,
            N,
            BLOCK=BLOCK,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            num_warps=1,
            num_stages=3,
        )


SMALL_THRESHOLD = 65536  # below this, pointwise_dynamic has lower launch overhead


def floor(A):
    logger.debug("GEMS_ENFLAME FLOOR")
    if A.is_contiguous() and A.numel() > SMALL_THRESHOLD:
        out = torch.empty_like(A)
        _floor_contiguous(A, out)
        return out
    return floor_func(A)


def floor_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME FLOOR_OUT")
    if out is None:
        return floor(A)
    if (
        A.is_contiguous()
        and out.is_contiguous()
        and A.shape == out.shape
        and A.numel() > SMALL_THRESHOLD
    ):
        _floor_contiguous(A, out)
        return out
    floor_func(A, out0=out)
    return out


def floor_(A):
    logger.debug("GEMS_ENFLAME FLOOR_")
    if A.is_contiguous() and A.numel() > SMALL_THRESHOLD:
        _floor_contiguous(A, A)
        return A
    floor_func(A, out0=A)
    return A
