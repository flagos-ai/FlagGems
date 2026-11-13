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
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

# Tiles folded into one program on the contiguous path (see launcher).
_TILES_PER_PROGRAM = 4

# CANN 9.0 enables automatic multi-buffering. Larger tiles make this kernel
# exceed the 192 KiB unified-buffer capacity on Ascend 910B.
_ASCEND_UB_SAFE_TILE_SIZE = 512

_LOG_SIGMOID_BACKWARD_CONFIG = CodeGenConfig(
    max_tile_size=_ASCEND_UB_SAFE_TILE_SIZE,
    max_grid_size=(65535, 1, 1),
    max_num_warps_per_cta=32,
    prefer_block_pointer=False,
    prefer_1d_tile=True,
)

# Reusing the forward buffer removes the exponential and permits a wider tile.
_LOG_SIGMOID_BACKWARD_BUFFER_CONFIG = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(65535, 1, 1),
    max_num_warps_per_cta=32,
    prefer_block_pointer=False,
    prefer_1d_tile=True,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_LOG_SIGMOID_BACKWARD_CONFIG,
)
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    derivative = tl.where(self_fp32 < 0.0, 1.0 / (1.0 + z), z / (1.0 + z))
    return grad_output * derivative


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_LOG_SIGMOID_BACKWARD_BUFFER_CONFIG,
)
@triton.jit
def log_sigmoid_backward_buffer_kernel(grad_output, self, buffer):
    z = buffer
    derivative = tl.where(self < 0.0, 1.0, z) / (1.0 + z)
    return grad_output * derivative


@libentry()
@triton.jit
def log_sigmoid_backward_contiguous_kernel(
    grad_output,
    self,
    grad_input,
    n_elements,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = ext.program_id(0)
    program_count = ext.num_programs(0)
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    for tile in tl.range(0, TILES_PER_PROGRAM):
        offsets = (program_id + tile * program_count) * BLOCK_SIZE + lane_offsets
        mask = offsets < n_elements
        grad = tl.load(grad_output + offsets, mask=mask)
        inp = tl.load(self + offsets, mask=mask).to(tl.float32)
        result = grad * tl.sigmoid(-inp)
        tl.store(grad_input + offsets, result, mask=mask)


def _can_use_contiguous_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (grad_input is None or grad_input.is_contiguous())
    )


def _can_use_buffer(self, buffer):
    return (
        buffer.shape == self.shape
        and buffer.dtype == self.dtype
        and buffer.is_contiguous()
    )


def _device_guard(tensor):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        return nullcontext()
    return torch_device_fn.device(tensor.device)


def _launch_contiguous_kernel(grad_output, self, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input

    # 8192 lanes/program is the pointwise plateau on 910B (256-bit vector unit;
    # two live inputs + the result fit the UB tile budget at 8192 fp32 lanes).
    # The 512-lane tile launched 16x more, 16x smaller transfers.
    block_size = 8192
    tile_count = triton.cdiv(n_elements, block_size)
    # One program per tile pays the per-program MTE setup once per 8192
    # elements. Folding four tiles into each program is worth another 7-19%
    # (33.5M fp16: 206us at one tile/program -> 174us at four). Deeper walks
    # stop paying once the grid no longer fills the vector cores, hence the
    # cap and the launch-limit clamp.
    grid_size = min(triton.cdiv(tile_count, _TILES_PER_PROGRAM), 65535)
    tiles_per_program = triton.cdiv(tile_count, grid_size)
    with _device_guard(self):
        log_sigmoid_backward_contiguous_kernel[(grid_size,)](
            grad_output,
            self,
            grad_input,
            n_elements,
            TILES_PER_PROGRAM=tiles_per_program,
            BLOCK_SIZE=block_size,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_ASCEND LOG_SIGMOID_BACKWARD")

    # Recomputing sigmoid(-x) beats reading the saved exp(-|x|) buffer: the
    # kernel is bandwidth bound, so dropping one of the three input streams
    # (25% of the traffic) pays for the vector-unit exp and then some. Measured
    # at 33.5M elements: 295us with the buffer load -> 206us without (fp16),
    # 464us -> 357us (fp32). The buffer is exactly exp(-|x|) of the same
    # tensor, so the result is unchanged up to one rounding.
    if _can_use_contiguous_kernel(grad_output, self):
        grad_input = torch.empty_like(self)
        return _launch_contiguous_kernel(grad_output, self, grad_input)
    if _can_use_buffer(self, buffer):
        return log_sigmoid_backward_buffer_kernel(grad_output, self, buffer)
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_ASCEND LOG_SIGMOID_BACKWARD_OUT")

    del buffer
    if _can_use_contiguous_kernel(grad_output, self, grad_input):
        return _launch_contiguous_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
