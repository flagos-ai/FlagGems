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
import math
from collections import namedtuple

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems import runtime

# from ..runtime import torch_device_fn
# from ..utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# @libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


# @libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=float("inf"))
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 8}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 2}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=4),
    ],
    key=["M"],  # re-tune when tensor size changes
)
# @libentry()
@triton.jit
def min_kernel_3(inp, out, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(inp + offsets, mask=mask)
    min_val = tl.min(x, axis=None)
    tl.atomic_min(out, min_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


# @libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("min"),
    key=[
        "M",
        "N",
    ],
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    min_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("inf"))
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=float("inf"))
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmin = tl.argmin(inp_vals, 1)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def min(inp):
    logger.debug("GEMS_ARM MIN")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)
    # Use two-stage reduction for broader dtype support on Triton CPU.
    min_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ARM MIN_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp_np = inp.detach().cpu().numpy()
    out_index_np = np.argmin(inp_np, axis=dim)
    gather_index = np.expand_dims(out_index_np, axis=dim)
    out_value_np = np.take_along_axis(inp_np, gather_index, axis=dim)
    out_index = torch.from_numpy(out_index_np.astype(np.int64, copy=False)).to(
        inp.device
    )
    out_value = torch.from_numpy(out_value_np).to(inp.device)
    if keepdim:
        out_index = out_index.unsqueeze(dim)
    else:
        out_value = out_value.squeeze(dim)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
