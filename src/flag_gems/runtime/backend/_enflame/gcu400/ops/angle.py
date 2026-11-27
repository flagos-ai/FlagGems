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

# Absolute import: pointwise_dynamic re-emits this module's imports into the
# generated kernel cache module, where a relative import cannot be resolved.
from flag_gems.runtime.backend._enflame.gcu400.ops.arctan2 import _atan2_vec
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Persistent launch borrowed from the sibling atan2 kernel: a handful of wide
# tiles per resident program hides the polynomial latency, and the block loop
# keeps grid.x far below the 65535 hardware limit.
_NUM_PROGRAMS = 48
_COMPLEX_BLOCK = 65536
_MIN_COMPLEX_BLOCK = 2048
_COMPLEX_WARPS = 4
_SMALL_TILE_LIMIT = 16384


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def angle_float_and_int(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


@libentry()
@triton.jit
def _angle_complex_kernel(
    real_ptr, imag_ptr, out_ptr, N, BLOCK: tl.constexpr, HALF: tl.constexpr
):
    """atan2(imag, real) over planar (de-interleaved) real/imag tiles.

    The complex input is split into real/imag planes by the host.  Reading the
    interleaved buffer directly (real at 2k, imag at 2k + 1) makes the backend
    emit width-1 gathers, which run at ~47 GB/s on GCU400 instead of the
    ~1.3 TB/s a contiguous stream reaches; the extra de-interleave copies are
    far cheaper than that penalty.

    ``_atan2_vec`` is reused from the atan2 op because libdevice atan2 lowers to
    a per-128-lane extern call here (~2 G element/s), while its branch-free
    minimax evaluation runs at memory speed.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        real = tl.load(real_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        imag = tl.load(imag_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        angle = _atan2_vec(real, imag, HALF)
        tl.store(out_ptr + offsets, angle, mask=mask)


def _choose_block(n_elements: int):
    # Narrow tiles waste the vector units on small problems; wide tiles keep the
    # polynomial hidden for everything that does not fit in cache.
    if n_elements <= _SMALL_TILE_LIMIT:
        return max(128, triton.next_power_of_2(max(n_elements, 1))), _COMPLEX_WARPS
    # Fill all _NUM_PROGRAMS resident programs: a fixed 64K tile leaves mid-sized
    # inputs (e.g. 1M elements -> 16 programs) with most SIPs idle, which costs
    # more than the lost per-tile width because the atan2 polynomial, not the
    # memory system, is the limiter.
    per_program = max(triton.cdiv(n_elements, _NUM_PROGRAMS), 1)
    block = 1 << (
        per_program.bit_length() - 1
    )  # largest power of two not exceeding per_program
    return min(max(block, _MIN_COMPLEX_BLOCK), _COMPLEX_BLOCK), _COMPLEX_WARPS


def _angle_complex(input_tensor: torch.Tensor) -> torch.Tensor:
    # angle() of a complex tensor returns the matching real dtype.
    out = torch.empty(
        input_tensor.shape,
        dtype=input_tensor.real.dtype,
        device=input_tensor.device,
    )
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return out

    # reshape(-1) yields a contiguous 1-D complex view (0-dim inputs included) and
    # only copies when the caller passed a non-unit last stride.
    flat_input = input_tensor.reshape(-1)
    real = flat_input.real.contiguous()
    imag = flat_input.imag.contiguous()

    block, warps = _choose_block(n_elements)
    grid = min(triton.cdiv(n_elements, block), _NUM_PROGRAMS)
    _angle_complex_kernel[(grid,)](
        real,
        imag,
        out,
        n_elements,
        BLOCK=block,
        # complex32 (4-byte) feeds 16-bit lanes, which favour the short Horner chain.
        HALF=input_tensor.element_size() == 4,
        num_warps=warps,
    )
    return out


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        return _angle_complex(input_tensor)
    else:
        return angle_float_and_int(input_tensor)
