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
"""Kunlunxin XPU3 overrides for ``aten::special_xlogy``.

``torch.special.xlogy(x, y)`` is the same operator as ``torch.xlogy(x, y)``
(ATen defines ``special_xlogy`` as the composite alias of ``xlogy``); the
documented semantics are therefore identical:

    y is NaN      -> NaN          (regardless of x)
    x == 0        -> 0            (even when y == 0 / y < 0 / y == +-inf)
    otherwise     -> x * log(y)

The generic implementation (``flag_gems/ops/special_xlogy.py``) builds a
``pointwise_dynamic`` kernel around ``tl_extra_shim.log`` -- a per-lane
libdevice soft extern -- and has no ``CodeGenConfig``.  On XPU3 that lowers to
the pathological path described in ``KERNEL_OPT_EXPERIENCE.md`` section 3.1
(social externs are 3-40x slower than the core ``tl.log``):

  * pre-fix profiling: ``special_xlogy_func_kernel_rank_1`` = 35 calls /
    1181.3 ms, **33.75 ms/call, 98.3 % of total device time**
    (``harness/perf_ir/prof_special_xlogy_pre.log``);
  * the whole benchmark matrix therefore scored 0.003-0.36x.

The vendor ``ops/xlogy.py`` already carries the tuned, exact raw kernels for
this exact math (core ``tl.log`` + the two ``tl.where`` guards, `_pick_block`
tiling, unmasked fast path).  Per the knowledge-base recipe "reuse an existing
fast path instead of rewriting it" (KERNEL_OPT_EXPERIENCE.md 3.6-36 / 3.8-8)
this module simply routes the ``special_xlogy`` entry points at those kernels;
no new kernel and no aten/native/CPU fallback is involved.  Delegating also
fixes the generic implementation's promotion bug: with ``DEFAULT`` promotion
integer inputs stayed integer (measured int64 ``[2, 6, 0]`` instead of ATen's
float32 ``[2.7726, 6.2383, 0.0]``), while ``xlogy`` uses ``INT_TO_FLOAT`` like
ATen.
"""

from .xlogy import xlogy as _xlogy
from .xlogy import xlogy_ as _xlogy_


def special_xlogy(self, other):
    """``aten::special_xlogy`` -- exact xlogy semantics, vendor raw kernels."""
    return _xlogy(self, other)


def special_xlogy_(self, other):
    """``aten::xlogy_``-style in-place variant of ``special_xlogy``."""
    return _xlogy_(self, other)
