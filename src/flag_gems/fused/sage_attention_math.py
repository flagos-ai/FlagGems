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

"""Explicit FP32 reduction candidates; no vendor extensions."""

import triton
import triton.language as tl


@triton.jit
def probability_sum(p, BM: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr):
    if MODE == 1:
        even, odd = tl.split(p.reshape((BM, BN // 4, 2, 2)))
        p0, p2 = tl.split(even)
        p1, p3 = tl.split(odd)
        # Four adjacent elements are accumulated in sequence, then the 32
        # partials are reduced, matching the vectorized reference strategy.
        total = tl.sum(((p0 + p1) + p2) + p3, 1)
    else:
        total = tl.sum(p, 1)
    return total
