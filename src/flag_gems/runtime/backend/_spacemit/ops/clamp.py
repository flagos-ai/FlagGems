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
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel_tensor(
    A_ptr,
    Min_ptr,
    Max_ptr,
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
        min_blk = tl.make_block_ptr(
            base=Min_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        max_blk = tl.make_block_ptr(
            base=Max_ptr,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        mini = tl.load(min_blk, boundary_check=(0,))
        maxi = tl.load(max_blk, boundary_check=(0,))
        out = tl.minimum(maxi, tl.maximum(mini, a))
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel_min_tensor(
    A_ptr,
    Min_ptr,
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
        min_blk = tl.make_block_ptr(
            base=Min_ptr,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        mini = tl.load(min_blk, boundary_check=(0,))
        out = tl.maximum(mini, a)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel_max_tensor(
    A_ptr,
    Max_ptr,
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
        max_blk = tl.make_block_ptr(
            base=Max_ptr,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        maxi = tl.load(max_blk, boundary_check=(0,))
        out = tl.minimum(maxi, a)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel(
    A_ptr,
    Out_ptr,
    mini,
    maxi,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        out = tl.minimum(maxi, tl.maximum(mini, a))
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel_min(
    A_ptr,
    Out_ptr,
    mini,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        out = tl.maximum(mini, a)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def clamp_kernel_max(
    A_ptr,
    Out_ptr,
    maxi,
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        out = tl.minimum(maxi, a)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


def _broadcast_tensor_arg(arg, shape):
    return torch.broadcast_to(arg, shape).contiguous()


def _empty_like(A):
    return torch.empty_like(A)


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_SPACEMIT CLAMP TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    A = A.contiguous()
    out = _empty_like(A)
    n = A.numel()
    if mini is None:
        maxi = _broadcast_tensor_arg(maxi, A.shape)
        clamp_kernel_max_tensor[num_ctas_grid](A, maxi, out, n)
    elif maxi is None:
        mini = _broadcast_tensor_arg(mini, A.shape)
        clamp_kernel_min_tensor[num_ctas_grid](A, mini, out, n)
    else:
        mini = _broadcast_tensor_arg(mini, A.shape)
        maxi = _broadcast_tensor_arg(maxi, A.shape)
        clamp_kernel_tensor[num_ctas_grid](A, mini, maxi, out, n)
    return out


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_SPACEMIT CLAMP_ TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    A = A.contiguous()
    n = A.numel()
    if mini is None:
        maxi = _broadcast_tensor_arg(maxi, A.shape)
        clamp_kernel_max_tensor[num_ctas_grid](A, maxi, A, n)
    elif maxi is None:
        mini = _broadcast_tensor_arg(mini, A.shape)
        clamp_kernel_min_tensor[num_ctas_grid](A, mini, A, n)
    else:
        mini = _broadcast_tensor_arg(mini, A.shape)
        maxi = _broadcast_tensor_arg(maxi, A.shape)
        clamp_kernel_tensor[num_ctas_grid](A, mini, maxi, A, n)
    return A


def clamp_min(A, mini):
    logger.debug("GEMS_SPACEMIT CLAMP MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    A = A.contiguous()
    out = _empty_like(A)
    n = A.numel()
    clamp_kernel_min[num_ctas_grid](A, out, float(mini), n)
    return out


def clamp_min_(A, mini):
    logger.debug("GEMS_SPACEMIT CLAMP_ MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    A = A.contiguous()
    n = A.numel()
    clamp_kernel_min[num_ctas_grid](A, A, float(mini), n)
    return A


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_SPACEMIT CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    A = A.contiguous()
    out = _empty_like(A)
    n = A.numel()
    if mini is None:
        clamp_kernel_max[num_ctas_grid](A, out, float(maxi), n)
    elif maxi is None:
        clamp_kernel_min[num_ctas_grid](A, out, float(mini), n)
    else:
        clamp_kernel[num_ctas_grid](A, out, float(mini), float(maxi), n)
    return out


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_SPACEMIT CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    A = A.contiguous()
    n = A.numel()
    if mini is None:
        clamp_kernel_max[num_ctas_grid](A, A, float(maxi), n)
    elif maxi is None:
        clamp_kernel_min[num_ctas_grid](A, A, float(mini), n)
    else:
        clamp_kernel[num_ctas_grid](A, A, float(mini), float(maxi), n)
    return A
