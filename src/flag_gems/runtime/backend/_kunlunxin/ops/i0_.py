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

# NOTE(2026-10-14): Kunlunxin (XPU) implementation of i0_ (in-place Bessel i0).
#
# The generic (flag_gems.ops.i0_) kernel was 2.5-2.9x slower than the
# already-validated i0 kernel at identical shapes (pre-profiling: i0_kernel_
# 9.1ms vs i0 3.1ms at (1024,65536) f16) because it emits 3 scalar fdiv per
# lane, uses a masked BLOCK_SIZE=1024 tile (8x more programs), and builds the
# launch grid as `grid = lambda meta:` — a fresh lambda object each call, and
# on the xpubin backend the launch grid participates in the compile-cache key
# (triton/runtime/jit.py: "For the XPU backend, the launch grid participates
# in compilation"), so every launch misses the kernel cache and recompiles
# (the bf16 (64,64) benchmark cell measured 118.6ms; see
# special_logit/hypot/special_log1p precedent).
#
# This mirrors the accepted i0.py (READ-ONLY, validated PASS):
#   1. In-place call of the shared _i0_poly / i0_kernel / i0_kernel_nomask
#      from .i0 (single source of truth): 3 fdiv -> 1 fdiv + 1 sqrt.
#   2. BLOCK_SIZE 1024 -> 8192 (1.8-2.5x on the i0 node), except for small
#      tensors (n < 65536) where the 1024-lane masked tile measures 1.3-2x
#      FASTER than a single 8192-lane program (6.3us vs 3.7us at n=16384,
#      6.4us vs 4.95us at n=32768; crossover at n=65536, profiled via
#      [XPURT_PROF]) -- keeps the tiny benchmark cells from regressing.
#   3. Mask-free kernel when n % 8192 == 0.
#   4. Constant-tuple grid.
#
# In-place safety: each program handles the disjoint range
# [pid*BLOCK, (pid+1)*BLOCK) for both the load and the store, and in program
# order the load completes before the store, so passing x for both x_ptr and
# out_ptr is safe (no cross-program aliasing, no read-after-write within a
# program).
import logging

import torch
import triton

import flag_gems

from .i0 import _BLOCK, _i0_poly, i0_kernel, i0_kernel_nomask  # noqa: F401

logger = logging.getLogger(__name__)

_SMALL_BLOCK = 1024
_SMALL_LIMIT = 65536


def i0_(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN I0_")
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        # Try common keyword names
        for k in ("input", "self", "x"):
            if k in kwargs:
                x = kwargs[k]
                break
    if x is None:
        raise ValueError(
            "i0_ expects a tensor as the first positional argument or in keyword 'input'/'self'/'x'."
        )

    if x.device.type != flag_gems.device:
        raise AssertionError(f"Input tensor must be on a {flag_gems.device} device.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise AssertionError(
            "Unsupported dtype for i0_. Supported: float16, bfloat16, float32, float64."
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x

    if n_elements < _SMALL_LIMIT:
        # Small tensors: 1024-lane masked tile (see NOTE above).
        grid = (triton.cdiv(n_elements, _SMALL_BLOCK),)
        i0_kernel[grid](x, x, n_elements, BLOCK_SIZE=_SMALL_BLOCK)
    elif n_elements % _BLOCK == 0:
        grid = (n_elements // _BLOCK,)
        i0_kernel_nomask[grid](x, x, n_elements, BLOCK_SIZE=_BLOCK)
    else:
        grid = (triton.cdiv(n_elements, _BLOCK),)
        i0_kernel[grid](x, x, n_elements, BLOCK_SIZE=_BLOCK)
    return x
