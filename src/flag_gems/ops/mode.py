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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mode_kernel(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M

    # Load first element
    n0 = tl.arange(0, 1)
    offset0 = m_offset[:, None] * N + n0[None, :]
    mask0 = mask_m[:, None] & (n0[None, :] < N)
    val0 = tl.load(sorted_inp + offset0, mask=mask0)
    idx0 = tl.load(sorted_indices + offset0, mask=mask0)

    # Squeeze the n-dimension (size 1)
    cur_value = val0.reshape([BLOCK_M])
    cur_index = idx0.reshape([BLOCK_M])
    cur_count = tl.full([BLOCK_M], 1, dtype=tl.int64)
    best_value = cur_value
    best_index = cur_index
    best_count = tl.full([BLOCK_M], 1, dtype=tl.int64)

    # Scan remaining elements one by one
    for i in range(1, N):
        ni = tl.full([1], i, dtype=tl.int32)
        offset_i = m_offset[:, None] * N + ni[None, :]
        mask_i = mask_m[:, None] & (ni[None, :] < N)
        val_i = tl.load(sorted_inp + offset_i, mask=mask_i).reshape([BLOCK_M])
        idx_i = tl.load(sorted_indices + offset_i, mask=mask_i).reshape([BLOCK_M])

        same = val_i == cur_value
        cur_count = tl.where(same, cur_count + 1, tl.full([BLOCK_M], 1, dtype=tl.int64))
        cur_value = tl.where(same, cur_value, val_i)
        cur_index = idx_i  # Always track the latest index in the run

        better = cur_count > best_count
        best_count = tl.where(better, cur_count, best_count)
        best_value = tl.where(better, cur_value, best_value)
        best_index = tl.where(better, cur_index, best_index)

    tl.store(out_value + m_offset, best_value, mask=mask_m)
    tl.store(out_index + m_offset, best_index, mask=mask_m)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N

    from flag_gems.ops.sort import sort as gems_sort

    sorted_inp, sorted_indices = gems_sort(inp, dim=dim)

    # Move dim to last for 2D processing
    sorted_inp = torch.movedim(sorted_inp, dim, -1).contiguous()
    sorted_indices = torch.movedim(sorted_indices, dim, -1).contiguous()

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    out_value = torch.empty(M, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(M, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        mode_kernel[grid](sorted_flat, indices_flat, out_value, out_index, M, N)

    out_shape = list(shape)
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Mode_out = namedtuple("mode", ["values", "indices"])
    return Mode_out(values=out_value, indices=out_index)
