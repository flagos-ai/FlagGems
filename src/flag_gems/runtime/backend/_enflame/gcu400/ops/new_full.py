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

from flag_gems.ops.full import check_dtype, full_func, full_func_scalar
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_INT32_MAX = torch.iinfo(torch.int32).max
# Element-wise Triton stores of 64-bit dtypes lower unreliably on this backend
# (the runtime replaces Long/Double with Int, corrupting every other element),
# so fp64/uint64/complex stay on the generic pointwise kernel. int64 is filled
# through an int32 view instead (see _new_full_flat).
_FLAT_UNSAFE_DTYPES = (
    torch.uint64,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
_FLAT_MAX_GRID = 48
_FLAT_MIN_BLOCK = 2048


@libentry()
@triton.jit
def new_full_flat_kernel(
    out_ptr,
    n_elements,
    fill_value,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The fill value is a plain Python scalar: skip the generic block-pointer
    # wrapper (num_ctas = min(12, num_tiles)) and write the broadcast constant
    # directly in flat memory order with a 48-CTA grid-stride loop.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, val)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, val, mask=mask)


def _new_full_flat_block(n_units, unit_itemsize):
    if n_units <= 8192:
        # Small sizes sit on the launch floor: one CTA with an exact tile.
        return max(2, triton.next_power_of_2(n_units))
    cap = 131072 if unit_itemsize <= 2 else 65536
    # Keep ~48 CTAs busy for mid sizes (a single CTA cannot saturate the
    # write bandwidth on this device) and cap the tile for large streams.
    return min(cap, max(_FLAT_MIN_BLOCK, triton.next_power_of_2(-(-n_units // 48))))


def _launch_new_full_flat(store_target, n_units, value, unit_itemsize):
    block = _new_full_flat_block(n_units, unit_itemsize)
    grid = (min(_FLAT_MAX_GRID, -(-n_units // block)),)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level flake windows on shared nodes); retrying
    # the launch recompiles and succeeds.  3 attempts keep deterministic
    # breakage loud while riding out the flake (atan2_.py / sinh.py / rad2deg
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(store_target.device):
                new_full_flat_kernel[grid](
                    store_target,
                    n_units,
                    value,
                    BLOCK=block,
                    EXACT=(n_units % block == 0),
                    num_warps=2,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "new_full flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _new_full_flat(out, fill_value):
    n_elements = out.numel()
    if out.dtype == torch.int64:
        # Triton int64 element stores corrupt every other element on this
        # backend. The aten layer stores int64 values as one int32 word per
        # element (the trailing half of the allocation stays unread garbage),
        # so mirror that layout: write the first numel words of an int32 view.
        value = int(fill_value)
        if not (-0x80000000 <= value <= 0x7FFFFFFF):
            # Out of the representable range of the packed word; keep the
            # generic-kernel behaviour for such values.
            return full_func_scalar(out, fill_value, out0=out)
        words = out.reshape(-1).view(torch.int32)
        _launch_new_full_flat(words, n_elements, value, 4)
        return out
    store_target = out
    value = fill_value
    if out.dtype == torch.bool:
        # Store through an int8 view: bool element stores lower to the narrow
        # i1 lane on this backend.
        store_target = out.reshape(-1).view(torch.int8)
        value = 1 if fill_value else 0
    _launch_new_full_flat(store_target, n_elements, value, out.element_size())
    return out


def new_full(
    self,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_ENFLAME NEW_FULL")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value, out0=out)
    n_elements = out.numel()
    if (
        0 < n_elements <= _INT32_MAX
        and dtype not in _FLAT_UNSAFE_DTYPES
        and (layout is None or layout == torch.strided)
    ):
        return _new_full_flat(out, fill_value)
    return full_func_scalar(out, fill_value, out0=out)
