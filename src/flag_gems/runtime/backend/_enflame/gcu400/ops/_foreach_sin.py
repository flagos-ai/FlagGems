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
"""GCU400-specific ``_foreach_sin`` (out-of-place): single-tensor flat arm + delegated executor.

Multi-tensor lists delegate to this directory's ``_foreach_unary.py`` lean single-launch route:
the older version ran a 1-D flat kernel per tensor at ~10us of host ``@libentry`` dispatch each,
pinning candidate latency at a shape-independent ~240-260us under the official bench core (256^2
only 0.31-0.34x, 9-cell mean 0.679x, re-anchored on the new stack 2026-11-18; the 0.28x in QA
weekly 1114 is a stale old-stack value). The lean route scales BLOCK by powers of two to bring
grid.y back under 255 and finishes all tensors in one launch (host ~8us, 2026-11-16 24-variant
campaign; the same table's in-place ``sin_`` row measured 3.93x); the fp32 promotion semantics of
``sin_func`` in the table are bit-identical to the historical kernel's ``.to(tl.float32)`` branch,
so there is no new math risk.

A single-tensor list keeps the dedicated flat arm: the lean route's BLOCK (16-bit 65536/131072)
only yields 1-8 CTAs at len=1, measured 11.6-20.9us on the list_length axis, worse than the old
flat kernel (BLOCK 2048 x grid<=48) at 4.7-12.2us (2026-11-18, 256^2/512^2/1024^2 x fp16/fp32
all tiers); both arms are one launch each, so the single-kernel contract holds.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype promotion and error semantics (int64 in-place raising RuntimeError,
empty lists / non-Tensor elements aligned with ATen) are unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# Best measured configuration for the single-tensor flat arm (same as the historical per-tensor implementation):
# BLOCK 2048 / grid limit 48 / num_warps 4, fastest across all tiers for small and medium single-tensor shapes.
_FLAT_BLOCK = 2048
_FLAT_FIXED_GRID = 48
_FLAT_NUM_WARPS = 4
_FLAT_NUM_STAGES = 1

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_sin_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """1-D fixed-grid stride computing sin per element, bit-identical to the executor body after fp32 promotion."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.sin(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _run_single_flat(tensor):
    out_tensor = torch.empty_like(tensor)
    num_elems = tensor.numel()
    grid = (min(triton.cdiv(num_elems, _FLAT_BLOCK), _FLAT_FIXED_GRID),)
    with torch_device_fn.device(tensor.device):
        _foreach_sin_flat_kernel[grid](
            tensor,
            out_tensor,
            num_elems,
            _FLAT_BLOCK,
            _FLAT_FIXED_GRID,
            num_warps=_FLAT_NUM_WARPS,
            num_stages=_FLAT_NUM_STAGES,
        )
    return out_tensor


def _foreach_sin(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIN")
    if len(self) == 1:
        tensor = self[0]
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype in _FLOAT_DTYPES
            and not tensor.dtype.is_complex
            and tensor.is_contiguous()
            and tensor.numel() > 0
        ):
            return [_run_single_flat(tensor)]
    return apply_unary("sin", self, inplace=False)
