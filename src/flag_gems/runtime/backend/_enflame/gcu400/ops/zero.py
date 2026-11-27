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

from .fill import fill_scalar_func

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024
NUM_WARPS = 1
GRID_SIZE = 24

_INT32_MAX = torch.iinfo(torch.int32).max
# Pure-write stream calibration on GCU400 (perf_playbook "pure-write stream write wall and grid calibration"):
# the 24 schedulable SIPs already saturate the DRAM write path (grid 24 == 48,
# grid 32 is pathological), and 32Ki-element tiles are the mid-band sweet spot
# that also ties the write wall at stream sizes.
_FLAT_MAX_GRID = 24
_FLAT_MIN_BLOCK = 2048
_FLAT_MAX_BLOCK = 32768
_FLAT_NUM_WARPS = 2
# Element-wise Triton stores of 64-bit dtypes lower unreliably on this backend
# (the runtime replaces Long/Double with Int, corrupting every other element),
# so int64/uint64 are zeroed through an int32 words view instead: zeros are
# all-zero bytes, so zeroing every 32-bit word is value-exact for any 64-bit
# ABI interpretation. bool element stores narrow to the i1 lane, so bool goes
# through an int8 view.
_FLAT_WORDS_DTYPES = (torch.int64, torch.uint64)
_FLAT_DIRECT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
    torch.uint16,
    torch.uint32,
)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def zero_kernel(
    out_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets, 0.0, mask=mask)


@libentry()
@triton.jit
def zero_flat_kernel(
    out_ptr,
    n_elements: tl.int32,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # Uniform zero fill: write the constant in flat memory order with a
    # grid-stride loop capped at the 24 schedulable SIPs. EXACT drops the tail
    # mask when the size divides the block, keeping the hot loop a bare store
    # stream (a uniform fill covers any dense layout in flat order).
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        z = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, z)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, z, mask=mask)


def _zero_flat_block(n_units):
    if n_units <= 8192:
        # Small sizes sit on the launch floor: one CTA with an exact tile.
        return max(2, triton.next_power_of_2(n_units))
    # 32Ki-unit tiles are the mid-band sweet spot (ramp-friendly: -1.4%~-4% vs
    # 64/128Ki caps at 4-64MB) and tie the DRAM write wall at stream sizes, so
    # the smaller tile also keeps DSM headroom per program.
    return min(
        _FLAT_MAX_BLOCK, max(_FLAT_MIN_BLOCK, triton.next_power_of_2(-(-n_units // 48)))
    )


def _launch_zero_flat(store_target, n_units):
    block = _zero_flat_block(n_units)
    grid = (min(_FLAT_MAX_GRID, -(-n_units // block)),)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level flake windows on shared nodes); retrying
    # the launch recompiles and succeeds (atan2_.py / sinh.py / rad2deg
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(store_target.device):
                zero_flat_kernel[grid](
                    store_target,
                    n_units,
                    BLOCK=block,
                    EXACT=(n_units % block == 0),
                    num_warps=_FLAT_NUM_WARPS,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "zero flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _try_flat_words(tensor: torch.Tensor, view_dtype: torch.dtype):
    """Reinterpret `tensor` as `view_dtype` words; None when not flat-zeroable."""
    try:
        words = tensor.view(view_dtype)
    except RuntimeError:
        return None
    if not (
        words.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(words)
    ):
        return None
    return words


def _zero_flat(tensor: torch.Tensor) -> bool:
    """Zero `tensor` through the flat store stream; return False if ineligible."""
    n_elements = tensor.numel()
    if n_elements == 0:
        return True
    if tensor.dtype == torch.bool:
        words = _try_flat_words(tensor, torch.int8)
        if words is None or n_elements > _INT32_MAX:
            return False
        _launch_zero_flat(words, n_elements)
        return True
    if tensor.dtype in _FLAT_WORDS_DTYPES:
        n_units = 2 * n_elements
        words = _try_flat_words(tensor, torch.int32)
        if words is None or n_units > _INT32_MAX or words.numel() != n_units:
            return False
        _launch_zero_flat(words, n_units)
        return True
    if tensor.dtype in _FLAT_DIRECT_DTYPES:
        if n_elements > _INT32_MAX:
            return False
        if not (
            tensor.is_contiguous()
            or torch.ops.aten.is_non_overlapping_and_dense(tensor)
        ):
            return False
        _launch_zero_flat(tensor, n_elements)
        return True
    return False


def _launch_zero_kernel(tensor: torch.Tensor) -> torch.Tensor:
    n_elements = tensor.numel()
    if n_elements == 0:
        return tensor
    if _zero_flat(tensor):
        return tensor
    if tensor.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(tensor):
        if tensor.dtype in _FLAT_WORDS_DTYPES:
            # 64-bit element stores lower unreliably (P17) and no int32 words
            # view was possible: reuse the FlagGems pointwise fill instead.
            fill_scalar_func(tensor, 0, out0=tensor)
            return tensor
        # Exotic dtypes (fp8/complex/...) keep the legacy masked store, with
        # the same semantics as the previous implementation.
        grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), GRID_SIZE),)
        zero_kernel[grid](
            tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
        return tensor
    # Overlapping/expanded layouts need stride-aware writes: reuse the FlagGems
    # pointwise fill kernel instead of failing the old contiguity assert.
    fill_scalar_func(tensor, 0, out0=tensor)
    return tensor


def zero(*args, **kwargs):
    logger.debug("GEMS_ENFLAME ZERO")
    target = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        target = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        target = kwargs["self"]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        target = kwargs["input"]
    elif "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        target = kwargs["out"]
    else:
        raise ValueError(
            "zero expects a Tensor as the first argument or in kwargs as 'self', 'input', or 'out'"
        )
    return _launch_zero_kernel(target)


def zero_out(*args, **kwargs):
    logger.debug("GEMS_ENFLAME ZERO_OUT")
    out = None
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        out = kwargs["out"]
    elif len(args) >= 1 and isinstance(args[0], torch.Tensor):
        out = args[0]
    else:
        raise ValueError(
            "zero_out expects an output Tensor as the first positional argument or 'out' kwarg"
        )
    return _launch_zero_kernel(out)


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ZERO_")
    return _launch_zero_kernel(x)
