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


def _is_float64_scalar(*args):
    return any(
        isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.ndim == 0
        for a in args
    )


# ---------- Optimized contiguous kernel ----------
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def ne_scalar_kernel(
    in_ptr,
    out_ptr,
    scalar_val,
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
        if IS_FP32:
            result = (x != scalar_val).to(tl.int8)
        elif IS_FP16:
            result = (x.to(tl.float32) != scalar_val).to(tl.int8)
        elif IS_BF16:
            result = (x.to(tl.float32) != scalar_val).to(tl.int8)
        tl.store(out_blk, result, boundary_check=(0,))


SMALL_THRESHOLD = 65536


# ---------- Generic fallback (non-contiguous / non-float) ----------
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_ENFLAME NE")
    if _is_float64_scalar(A, B):
        dev = A.device if isinstance(A, torch.Tensor) else B.device
        A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        return torch.ne(A_cpu, B_cpu).to(dev)
    return ne_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne_scalar(A, B):
    logger.debug("GEMS_ENFLAME NE_SCALAR")
    if _is_float64_scalar(A):
        return torch.ne(A.cpu(), B).to(A.device)
    if A.is_contiguous() and A.numel() > SMALL_THRESHOLD and A.is_floating_point():
        out = torch.empty_like(A, dtype=torch.bool)
        N = A.numel()
        dtype = A.dtype
        IS_FP32 = dtype == torch.float32
        IS_FP16 = dtype == torch.float16
        IS_BF16 = dtype == torch.bfloat16
        BLOCK = 8192
        grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS)
        with torch_device_fn.device(A.device):
            ne_scalar_kernel[(grid,)](
                A,
                out,
                float(B),
                N,
                BLOCK=BLOCK,
                IS_FP32=IS_FP32,
                IS_FP16=IS_FP16,
                IS_BF16=IS_BF16,
                num_warps=1,
                num_stages=3,
            )
        return out
    return ne_func_scalar(A, B)


# not_equal_scalar is an alias of ne_scalar (torch.not_equal dispatches to aten.not_equal.Scalar)
not_equal_scalar = ne_scalar
