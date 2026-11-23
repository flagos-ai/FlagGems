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

from flag_gems.runtime import torch_device_fn

from .zeros import _zeros_grid, zeros_kernel

logger = logging.getLogger(__name__)


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ASCEND ZEROS_LIKE")
    options = {
        "device": x.device if device is None else device,
        "dtype": x.dtype if dtype is None else dtype,
    }
    if layout is not None:
        options["layout"] = layout
    if pin_memory is not None:
        options["pin_memory"] = pin_memory
    if memory_format is not None:
        options["memory_format"] = memory_format
    # A CPU factory request must be initialized on the CPU.
    if torch.device(options["device"]).type == "cpu":
        return torch.zeros_like(x, **options)
    if memory_format is not None and layout in (None, torch.strided):
        # torch_npu empty_like currently ignores channels_last. Resolve the
        # requested strides with a metadata allocation, then allocate on NPU.
        template = torch.empty_like(x, device="meta", memory_format=memory_format)
        out = torch.empty_strided(
            template.shape,
            template.stride(),
            **{key: value for key, value in options.items() if key != "memory_format"},
        )
    else:
        out = torch.empty_like(x, **options)
    N = x.numel()
    if N == 0:
        return out
    grid_fn = lambda meta: _zeros_grid(N, meta["BLOCK_SIZE"])
    with torch_device_fn.device(out.device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=32768, BLOCK_SIZE_SUB=1024)
    return out
