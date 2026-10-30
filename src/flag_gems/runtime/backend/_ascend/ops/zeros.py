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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)

# NPU 要求 grid 每个轴的取值落在 [1, 65535]。BLOCK_SIZE=20480 时，N 超过
# 65535*20480（int8 约 1.25 GiB）就必然越界，KernelLaunch 报
# "value 65536 for parameter coreDim is invalid"，进程随后 segfault。
# vLLM 这类框架会给 KV cache 一次性分配几十 GB 的 int8 张量，直接命中；
# 超过一片容量就按片多次启动，每片的 grid 都落在上限内。
_MAX_GRID = 65535
_BLOCK_SIZE = 20480
_BLOCK_SIZE_SUB = 1024
_CHUNK = _MAX_GRID * _BLOCK_SIZE


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = ext.program_id(axis=0)

    for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        sub_offset = (
            pid * BLOCK_SIZE + sub_block_start_idx + tl.arange(0, BLOCK_SIZE_SUB)
        )
        mask = sub_offset < n_elements
        tl.store(output_ptr + sub_offset, 0.0, mask=mask)


def _launch(out, n_elements):
    grid_fn = lambda meta: (max(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1),)
    zeros_kernel[grid_fn](
        out, n_elements, BLOCK_SIZE=_BLOCK_SIZE, BLOCK_SIZE_SUB=_BLOCK_SIZE_SUB
    )


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # 空张量没有元素要写：N == 0 时 cdiv(N, BLOCK_SIZE) 为 0，grid 退化成 0
    # 会被驱动直接拒绝，这里不启动 kernel。
    if N == 0:
        return out
    with torch_device_fn.device(device):
        if N <= _CHUNK:
            _launch(out, N)
        else:
            # kernel 按扁平偏移写，分片要求连续布局；torch.empty 一定是连续的。
            flat = out.view(-1)
            for start in range(0, N, _CHUNK):
                part = flat[start : start + _CHUNK]
                _launch(part, part.numel())
    return out
