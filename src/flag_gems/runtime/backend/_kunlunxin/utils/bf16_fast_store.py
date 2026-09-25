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
"""Launch-scoped control of the ``TRITONXPU_BF16_FAST`` store lowering.

Why this exists
---------------
The XPU backend has two lowerings for an ``f32 -> bf16`` store
(``TritonXPUToLLVM/LoadStoreOpToLLVM.cpp``):

* **default** -- per-lane ``vand``/``vadd`` rounding chain, then two masked
  ``SCATTER_MH`` ops.  Costs ~+150us at n = 16.7M (a ~1.35x slowdown of the
  whole op versus the equivalent f32 store).
* **fast** (``TRITONXPU_BF16_FAST=1``) -- the vendor ``vstore2_lm`` device
  call: one hardware-rounded 16->bf16 pack + store per two 512-bit vectors,
  ~70us faster at 16.7M.  Same lowering the vendor's own sglang kernels use
  (see ``third_party/xpu/test/sglang/qwen3_next/test_l2norm_fwd_kernel.py``).

The two lowerings are **not** bit-identical, contrary to what the comment this
helper replaces used to claim.  They differ in the RNE tie bias: the fast path
biases on mantissa bit 16 (``0x7FFF + ((x >> 16) & 1)``, textbook round-half-to-
even), while the default path biases on bit 0.  They therefore disagree only
when the discarded low half is exactly ``0x8000`` -- but that is enough to move
a bf16 result by one ULP, which is observable against a tight tolerance (this
is how ``adaptive_avg_pool2d_backward`` at ``atol=1e-4`` flipped from pass to
fail once the flag was switched on globally).

Why it is scoped, not global
----------------------------
This flag is read once, when the kernel is compiled, and it is listed in
``CACHE_INVALIDATING_ENV_VARS`` (``include/triton/Tools/Sys/GetEnv.hpp``), so it
is part of the JIT cache key.  A module-level ``os.environ.setdefault`` in one
op therefore (a) changes the bf16 numerics of *every other op* in the process
and (b) cannot be undone by a caller who did not set it.  Entering this scope
around a single launch keeps the benefit where it has been measured, and leaves
every other op on the default lowering.

The scope is only entered for bf16 outputs: the flag changes nothing else, and
entering it unconditionally would fork the cache key (and so force a second
compilation) for fp16/fp32 kernels whose generated code is identical.
"""

import contextlib
import os

import torch

_ENV_KEY = "TRITONXPU_BF16_FAST"


@contextlib.contextmanager
def bf16_fast_store(out_dtype):
    """Select the fast f32->bf16 store lowering for the enclosed launch.

    No-op unless ``out_dtype`` is ``torch.bfloat16``.  Any pre-existing value of
    the variable is restored on exit, so a caller that set it explicitly (or set
    it to ``"0"`` to opt out) keeps that setting.
    """
    if out_dtype is not torch.bfloat16:
        yield
        return
    saved = os.environ.get(_ENV_KEY)
    os.environ[_ENV_KEY] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_ENV_KEY, None)
        else:
            os.environ[_ENV_KEY] = saved
