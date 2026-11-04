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

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linspace_kernel(
    out_ptr,
    out_stride0,
    start,
    mid,
    end,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps
    fw_mask = idx < mid
    fw_values = start + (step_size * idx)
    bd_values = end - step_size * (steps - idx - 1)

    out_val = tl.where(fw_mask, fw_values, bd_values)
    tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)


def _linspace_empty(start, end, *, dtype, layout, device, pin_memory):
    # Native tensor-endpoint overloads validate rank before extracting Scalars,
    # even for an empty result. Keep this validation ahead of allocation.
    for endpoint in (start, end):
        if isinstance(endpoint, torch.Tensor):
            if endpoint.ndim != 0:
                raise RuntimeError(
                    "linspace only supports 0-dimensional start and end tensors"
                )
        elif not isinstance(endpoint, (int, float, complex)):
            raise TypeError(
                "linspace start and end must be numbers or 0-dimensional tensors"
            )
    start = start.item() if isinstance(start, torch.Tensor) else start
    end = end.item() if isinstance(end, torch.Tensor) else end
    for endpoint in (start, end):
        if isinstance(endpoint, int) and not -(1 << 63) <= endpoint < (1 << 64):
            raise OverflowError("linspace scalar endpoint is out of range")

    if isinstance(start, complex) or isinstance(end, complex):
        if dtype is None:
            # Match the native default complex dtype, including default FP16.
            default_dtype = torch.get_default_dtype()
            if default_dtype == torch.float16:
                dtype = torch.complex32
            elif default_dtype == torch.float64:
                dtype = torch.complex128
            else:
                dtype = torch.complex64
        elif not isinstance(dtype, torch.dtype):
            raise TypeError("linspace dtype must be a torch.dtype")
        elif not dtype.is_complex:
            raise RuntimeError(
                "torch.linspace(): complex endpoints require a complex dtype"
            )

    out = torch.empty(
        (0,), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if out.layout != torch.strided:
        raise NotImplementedError("linspace only supports strided layout")
    return out


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS LINSPACE")
    # Limit this repair to ordinary Python integer zero. Other step types keep
    # their existing validation instead of accepting 0.0, False or index-like
    # objects merely because they compare equal to zero.
    if isinstance(steps, int) and not isinstance(steps, bool) and steps == 0:
        return _linspace_empty(
            start, end, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
        )
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 1:
        return torch.fill(out, start)
    else:
        if isinstance(start, torch.Tensor):
            start = start.item()
        if isinstance(end, torch.Tensor):
            end = end.item()
        mid = steps // 2
        step_size = (float(end) - float(start)) / (steps - 1)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        linspace_kernel[grid](
            out, out.stride(0), start, mid, end, step_size, steps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
