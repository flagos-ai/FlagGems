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
"""GCU400-specific ``_foreach_sign`` / ``_foreach_sign_``: len==1 flat arm + delegated executor.

The historical implementation launched one 1-D flat kernel per tensor: a 16-tensor list cost 16
``@libentry`` host dispatches (~15us each), pinning candidate latency at 150-247us under the
official bench core 9 cells regardless of shape/dtype (2026-11-18 re-measure: out-of-place mean
0.307x, in-place mean 0.424x), with the 1024^2 tier at ~10x the 1R1W memory floor. Multi-tensor
lists now go through this directory's ``_foreach_unary.py`` lean single-launch route (the shared
bulk path costs ~55us host, lean ~8us; the table's ``_sign_func`` body ``(x > 0) - (x < 0)`` is
bit-identical to the historical kernel's fp32-promoted three-way where semantics: NaN/+-0 -> 0,
+-inf -> +-1), and the single-kernel contract holds.

After the change, official bench core (kernel mode, three fresh rounds): out-of-place 9-cell mean
0.307x -> 3.552/3.743x and in-place 0.424x -> 4.175/4.150x, with absolute latency
240/151us -> 8-63us that now scales with shape (256^2 sits on the abs family floor at 8-11us,
and 1024^2 fp16 at 48.1us beats abs at 62.3us); the len>=2 and len==1 arms are one launch each,
so the single-kernel contract holds.

A single-tensor list keeps the dedicated flat arm: the lean route scales BLOCK by dtype (16-bit
131072 / 32-bit 65536), so at len==1 it only yields 1-8 CTAs, and on the list_length axis it
measured worse than the flat kernel (BLOCK 2048 x grid<=48) at 4.7-12.2us (same table as the
``_foreach_sin`` precedent, 2026-11-18).

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back from the local
executor to the shared grouped path; dtype promotion and error semantics (empty lists /
non-Tensor elements aligned with ATen) are unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Best measured configuration for the single-tensor flat arm (same setting as _foreach_sin on 2026-11-18:
# 256^2-1024^2 x fp16/fp32 len=1, all tiers 4.7-12.2us).
FLAT_BLOCK = 2048
FLAT_FIXED_GRID = 48
FLAT_NUM_WARPS = 4
FLAT_NUM_STAGES = 1
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_sign_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """1-D fixed-grid stride computing sign per element (dedicated to the len==1 arm).

    in_ptr and out_ptr differ only on the out-of-place variant; after fp32 promotion the three-way
    where is bit-identical to the executor table's ``_sign_func`` (both NaN comparisons false -> 0).
    """
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.where(value > 0, 1.0, tl.where(value < 0, -1.0, 0.0))
        tl.store(out_ptr + offs, result, mask=mask)


def _single_tensor_ok(tensor):
    """Entry condition for the len==1 flat arm: a non-empty contiguous float tensor.

    All other forms (non-Tensor elements, integer/bool/complex, non-contiguous, empty tensors) go
    to ``apply_unary``, which unifies exception translation and grouped fallback with ATen semantics.
    """
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype in _FLOAT_DTYPES
        and not tensor.dtype.is_complex
        and tensor.is_contiguous()
        and tensor.numel() > 0
    )


def _run_single_flat(tensor, inplace):
    """One flat launch for a single tensor: in-place writes back, out-of-place allocates a new output."""
    out_tensor = tensor if inplace else torch.empty_like(tensor)
    num_elems = tensor.numel()
    grid = (min(triton.cdiv(num_elems, FLAT_BLOCK), FLAT_FIXED_GRID),)
    with torch_device_fn.device(tensor.device):
        _foreach_sign_flat_kernel[grid](
            tensor,
            out_tensor,
            num_elems,
            FLAT_BLOCK,
            FLAT_FIXED_GRID,
            num_warps=FLAT_NUM_WARPS,
            num_stages=FLAT_NUM_STAGES,
        )
    return None if inplace else [out_tensor]


def _foreach_sign(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIGN")
    if len(self) == 1 and _single_tensor_ok(self[0]):
        return _run_single_flat(self[0], inplace=False)
    return apply_unary("sign", self, inplace=False)


def _foreach_sign_(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIGN_")
    if len(self) == 1 and _single_tensor_ok(self[0]):
        return _run_single_flat(self[0], inplace=True)
    return apply_unary("sign", self, inplace=True)
