# Copyright 2026, The FlagOS Contributors.
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

from .fix import (
    _FAST_BLOCK,
    _FAST_WARPS,
    fix_fast_kernel,
    fix_func,
    fix_masked_kernel,
)

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def _fix_inplace(A):
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32)
        and A.is_contiguous()
        and A.dim() > 0
        and numel > 1024
    ):
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if numel % block == 0:
            # x_ptr == y_ptr: every lane reads then writes its own element, so
            # the in-place store is hazard-free.
            fix_fast_kernel[(numel // block,)](
                A, A, BLOCK=block, num_warps=_FAST_WARPS
            )
        else:
            fix_masked_kernel[(triton.cdiv(numel, block),)](
                A, A, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return A
    # bf16 / fp64 / non-contiguous / tiny tensors: generic pointwise path.
    fix_func(A, out0=A)
    return A


def fix_(A):
    logger.debug("GEMS_KUNLUNXIN FIX_")
    if not A.is_floating_point():
        # Integer tensors are a no-op and complex keeps the pre-existing
        # generic behaviour; both are out of scope of tests/test_fix_.py
        # (FLOAT_DTYPES only).
        from flag_gems.ops.fix_ import fix_ as _generic_fix_

        return _generic_fix_(A)
    if A.numel() == 0:
        return A
    return _fix_inplace(A)
