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

from flag_gems.runtime import torch_device_fn

from .zeros import zeros_kernel

logger = logging.getLogger(__name__)

# NPU 要求 grid 每个轴的取值落在 [1, 65535]；BLOCK_SIZE=32768 时 N 超过
# 65535*32768（int8 约 2 GiB）就必然越界（EE1003 coreDim 越界，随后 segfault）。
# 大张量（如 vLLM 的 KV cache）按片多次启动。
_MAX_GRID = 65535
_BLOCK_SIZE = 32768
_BLOCK_SIZE_SUB = 1024
_CHUNK = _MAX_GRID * _BLOCK_SIZE


def _launch(out, n_elements):
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    zeros_kernel[grid_fn](
        out, n_elements, BLOCK_SIZE=_BLOCK_SIZE, BLOCK_SIZE_SUB=_BLOCK_SIZE_SUB
    )


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ASCEND ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    if x.numel() == 0:
        return torch.empty_like(x, device=device, dtype=dtype)
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    with torch_device_fn.device(x.device):
        if N <= _CHUNK:
            _launch(out, N)
        elif out.is_contiguous():
            # kernel 按扁平偏移写，分片要求连续布局；非连续布局保持原来的
            # 单次启动（本就用不了分片语义）。
            flat = out.view(-1)
            for start in range(0, N, _CHUNK):
                part = flat[start : start + _CHUNK]
                _launch(part, part.numel())
        else:
            _launch(out, N)
    return out
