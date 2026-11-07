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

# grid 的每个轴都有上限：CUDA 是 2**31-1，NPU 是 65535。这里按更严的 65535
# 取上限，BLOCK_SIZE=1024 时 N 超过 65535*1024（64 MB 的 int8）就分片多次启动，
# 避免 coreDim 越界（EE1003，随后 segfault）。
_MAX_GRID = 65535
_BLOCK_SIZE = 1024
_CHUNK = _MAX_GRID * _BLOCK_SIZE


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def _launch(out, n_elements):
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    zeros_kernel[grid_fn](out, n_elements, BLOCK_SIZE=_BLOCK_SIZE)


def _fill_zeros(out, n_elements):
    """按 _CHUNK 分片清零；kernel 按扁平偏移写，要求 out 连续。"""
    if n_elements <= _CHUNK:
        _launch(out, n_elements)
    elif out.is_contiguous():
        flat = out.view(-1)
        for start in range(0, n_elements, _CHUNK):
            part = flat[start : start + _CHUNK]
            _launch(part, part.numel())
    else:
        _launch(out, n_elements)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # 空张量没有元素要写；N == 0 时 grid 会退化成 0，被驱动直接拒绝。
    if N == 0:
        return out
    with torch_device_fn.device(device):
        _fill_zeros(out, N)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ZERO_")
    N = x.numel()
    if N == 0:
        return x
    with torch_device_fn.device(x.device):
        _fill_zeros(x, N)
    return x
