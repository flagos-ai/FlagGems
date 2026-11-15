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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fill_diagonal_kernel(
    inp_ptr,
    fill_value,
    main_size,
    wrap_size,
    wrap_offset,
    diagonal_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < main_size + wrap_size
    storage_offsets = tl.where(
        offsets < main_size,
        offsets * diagonal_stride,
        wrap_offset + (offsets - main_size) * diagonal_stride,
    )
    tl.store(inp_ptr + storage_offsets, fill_value, mask=mask)


def _fill_strided(inp, fill_value, main_size, wrap_size, wrap_offset, diagonal_stride):
    count = main_size + wrap_size
    if count == 0:
        return

    with torch_device_fn.device(inp.device):
        fill_diagonal_kernel[(triton.cdiv(count, 1024),)](
            inp,
            fill_value,
            main_size,
            wrap_size,
            wrap_offset,
            diagonal_stride,
            BLOCK_SIZE=1024,
        )


def fill_diagonal_(self, fill_value, wrap=False):
    logger.debug("GEMS FILL_DIAGONAL_")

    n_dims = self.ndim
    if n_dims < 2:
        raise RuntimeError("dimensions must larger than 1")

    height, width = self.shape[:2]
    if n_dims > 2 and any(dim != height for dim in self.shape[1:]):
        raise RuntimeError("all dimensions of input must be of equal length")

    diagonal_stride = sum(self.stride())
    main_size = min(height, width)
    wrap_size = 0
    wrap_offset = 0

    # Match ATen's two-segment wrap behavior. The logical wrap step determines
    # how many values are written, while physical offsets follow tensor strides.
    if wrap and n_dims == 2 and height > width + 1:
        logical_step = width + 1
        wrap_size = (self.numel() + logical_step - 1) // logical_step - main_size
        wrap_offset = self.stride(0) * logical_step
        last_storage_offset = (
            self.storage_offset() + wrap_offset + (wrap_size - 1) * diagonal_stride
        )
        storage_size = self.untyped_storage().nbytes() // self.element_size()
        if wrap_size > 0 and last_storage_offset >= storage_size:
            raise RuntimeError(
                "setStorage: wrapped diagonal is out of bounds for storage"
            )

    _fill_strided(
        self,
        fill_value,
        main_size,
        wrap_size,
        wrap_offset,
        diagonal_stride,
    )

    return self
