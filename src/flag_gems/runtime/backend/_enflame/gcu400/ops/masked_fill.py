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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Upper limit of resident programs per SIP: GCU400 has 24 SIPs, and ×2 leaves the grid-stride
# loop some load-balancing headroom without inflating launch/dispatch overhead with too many programs.
NUM_SIPS = 24
MAX_PROGRAMS = NUM_SIPS * 2
# Tile cap for flat 1D access: at 32768 elements (measured on [4096,4096] fp32), memory bandwidth
# already peaks at 2655 GB/s; 8192 reaches only 1527 GB/s, and 65536 actually drops back to 2505 GB/s.
MAX_BLOCK = 32768
MIN_BLOCK = 1024
# No device guard is needed in a single-card environment (each device-context switch costs ~3us, one of the main
# overheads for small tensors).
_NEED_DEVICE_GUARD = runtime_device.device_count > 1


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def masked_fill_flat_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    value,
    BLOCK: tl.constexpr,
):
    """Fast path for same-shape, both-way contiguous masked_fill: flatten to 1D, grid-stride walk.

    All three streams (inp / mask / out) are contiguous, so no per-dim stride or broadcast index.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + offsets
        in_range = off < n_elements
        x = tl.load(inp_ptr + off, mask=in_range)
        mask_flag = tl.load(mask_ptr + off, mask=in_range)
        filled = tl.where(mask_flag, value.to(x.dtype), x)
        tl.store(out_ptr + off, filled, mask=in_range)


def _pick_block(n_elements):
    """Tile by element count: small tensors shrink it to fill the machine, large ones take the max."""
    elements_per_program = -(-n_elements // MAX_PROGRAMS)
    block = 1 << max(elements_per_program - 1, 0).bit_length()
    return max(MIN_BLOCK, min(MAX_BLOCK, block))


def _launch_flat(inp, mask, out, value):
    n_elements = inp.numel()
    block = _pick_block(n_elements)
    grid = min((n_elements + block - 1) // block, MAX_PROGRAMS)
    if _NEED_DEVICE_GUARD:
        with torch_device_fn.device(inp.device):
            masked_fill_flat_kernel[(grid,)](
                inp, mask, out, n_elements, value, BLOCK=block, num_warps=1
            )
    else:
        masked_fill_flat_kernel[(grid,)](
            inp, mask, out, n_elements, value, BLOCK=block, num_warps=1
        )


def _can_use_flat_path(inp, mask):
    """Fast-path condition: mask matches inp in shape (no broadcast) and both are contiguous."""
    return (
        mask.dtype == torch.bool
        and mask.shape == inp.shape
        and inp.numel() > 0
        and inp.is_contiguous()
        and mask.is_contiguous()
    )


def _as_python_value(value):
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    return value


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    inp = tl.where(expand_mask == 1, value, inp)
    return inp


def masked_fill(inp, mask, value):
    # Check the fast path first: for small tensors the host-side cost is the whole cost, and extra checks directly
    # increase latency.
    value_type = type(value)
    if value_type is not int and value_type is not float and value_type is not bool:
        value = _as_python_value(value)
    if _can_use_flat_path(inp, mask):
        out = torch.empty_like(inp)
        _launch_flat(inp, mask, out, value)
        return out

    logger.debug("GEMS_ENFLAME MASKED_FILL")
    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"
    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value)


def masked_fill_(inp, mask, value):
    value_type = type(value)
    if value_type is not int and value_type is not float and value_type is not bool:
        value = _as_python_value(value)
    if _can_use_flat_path(inp, mask):
        _launch_flat(inp, mask, inp, value)
        return inp

    logger.debug("GEMS_ENFLAME MASKED_FILL_")
    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"
    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value, out0=inp)
