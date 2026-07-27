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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logcumsumexp_block_kernel(
    inp,
    out,
    state_max,
    state_sum,
    block_start,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FIRST_BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    m = pid // K
    k = pid % K
    n_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = n_offsets < N
    base = m * N * K + k
    values = tl.load(
        inp + base + n_offsets * K, mask=mask, other=float("-inf")
    ).to(tl.float32)
    block_max = tl.max(values, axis=0)

    if FIRST_BLOCK:
        new_max = block_max
        scaled_prefix = 0.0
    else:
        previous_max = tl.load(state_max + pid)
        previous_sum = tl.load(state_sum + pid)
        new_max = tl.maximum(previous_max, block_max)
        scaled_prefix = previous_sum * tl.exp(previous_max - new_max)

    exp_values = tl.exp(values - new_max)
    block_prefix = scaled_prefix + tl.cumsum(exp_values, axis=0)
    result = new_max + tl.log(block_prefix)
    tl.store(out + base + n_offsets * K, result, mask=mask)
    tl.store(state_max + pid, new_max)
    tl.store(state_sum + pid, scaled_prefix + tl.sum(exp_values, axis=0))


def _result_dtype(inp, dtype):
    if dtype is not None:
        return dtype
    if inp.dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return torch.float32
    return inp.dtype


def logcumsumexp(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN LOGCUMSUMEXP")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    result_dtype = _result_dtype(inp, dtype)
    if inp.numel() == 0:
        return torch.empty_like(inp, dtype=result_dtype)

    shape = inp.shape
    M = 1
    for size in shape[:dim]:
        M *= size
    N = shape[dim]
    K = inp.numel() // M // N

    inp = inp.contiguous()
    out = torch.empty_like(inp, dtype=result_dtype)
    state_max = torch.empty((M * K,), dtype=torch.float32, device=inp.device)
    state_sum = torch.empty((M * K,), dtype=torch.float32, device=inp.device)
    block_size = triton.next_power_of_2(min(N, 1024))
    num_blocks = triton.cdiv(N, block_size)
    with torch_device_fn.device(inp.device):
        for block_idx in range(num_blocks):
            logcumsumexp_block_kernel[(M * K, 1, 1)](
                inp,
                out,
                state_max,
                state_sum,
                block_idx * block_size,
                N,
                K,
                block_size,
                FIRST_BLOCK=block_idx == 0,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return out.view(shape)


def logcumsumexp_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN LOGCUMSUMEXP_OUT")
    result_dtype = _result_dtype(inp, dtype)
    if out.dtype != result_dtype:
        raise RuntimeError(
            f"logcumsumexp.out: expected out dtype {result_dtype}, got {out.dtype}"
        )
    result = logcumsumexp(inp, dim, dtype=dtype)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    out.copy_(result)
    return out
