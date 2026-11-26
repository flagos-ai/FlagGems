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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

# CANN rejects a launch whose flattened grid (coreDim) exceeds this value.
_ASCEND_MAX_CORE_DIM = 65535


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_SIZE)
    # Grid-stride loop: the launch grid is capped at the coreDim limit and each
    # program walks the remaining tiles, so huge reductions no longer fail.
    for block in range(tl.program_id(0), tl.cdiv(M, BLOCK_SIZE), tl.num_programs(0)):
        off = block * BLOCK_SIZE + lanes
        mask = off < M
        inp_val = tl.load(inp + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(mid + block, tl.sum(inp_val, axis=0))


def _mean_dtype(inp, dtype):
    """Match ATen: mean is only defined for floating point / complex dtypes."""
    if dtype is None:
        dtype = inp.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError(
                "mean(): could not infer output dtype. Input dtype must be either "
                f"a floating point or complex dtype. Got: {dtype}"
            )
    elif not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "mean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    return dtype


def mean(inp, *, dtype=None):
    logger.debug("GEMS_ASCEND MEAN")
    dtype = _mean_dtype(inp, dtype)
    M = inp.numel()
    out = torch.empty([], dtype=dtype, device=inp.device)
    if M == 0:
        # torch.mean over zero elements is NaN, and launching a zero-tile
        # reduction would abort the device (coreDim == 0).
        out.fill_(float("nan"))
        return out

    inp = inp.contiguous()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = min(block_size, 2048)
    num_ctas = triton.cdiv(M, block_size)
    grid_size = min(num_ctas, _ASCEND_MAX_CORE_DIM)
    mid = torch.empty((num_ctas,), dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(grid_size,)](inp, mid, M, block_size)
        out = mid.sum() / M
    return out.to(dtype)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    lanes_m = tl.arange(0, BLOCK_M)[:, None]
    # Grid-stride loop over row blocks (coreDim limit; see mean_kernel_1).
    for block in range(tl.program_id(0), tl.cdiv(M, BLOCK_M), tl.num_programs(0)):
        pid = block * BLOCK_M + lanes_m
        x_ptr = X + pid * N
        mean_ptr = Mean + pid
        row_mask = pid < M

        # Compute mean
        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(x_ptr + cols, mask, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=1) / N
        mean = mean[:, None]
        tl.store(mean_ptr, mean, row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND MEAN_DIM")

    dtype = _mean_dtype(x, dtype)
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        # Full reduction: keepdim=True must yield shape [1] * ndim.
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    if isinstance(dim, int):
        dim = [dim]
    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1

    out = torch.empty(shape, dtype=dtype, device=x.device)
    if N == 0:
        # Mean over an empty axis is NaN (matches ATen); do not divide by zero.
        out.fill_(float("nan"))
        return out if keepdim else out.squeeze(dim)

    M = x.numel() // N
    if M == 0:
        # Empty output: launching a zero-block grid aborts on Ascend.
        return out if keepdim else out.squeeze(dim)

    x = dim_compress(x, dim)
    grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), _ASCEND_MAX_CORE_DIM),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out
