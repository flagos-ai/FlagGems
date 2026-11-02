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

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _gt_inplace_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    result = tl.where(a > b, 1.0, 0.0)
    tl.store(a_ptr + offsets, result, mask=mask)


def gt_tensor_(A, B):
    n_elements = A.numel()
    if n_elements == 0:
        return A
    grid = (triton.cdiv(n_elements, 1024),)
    _gt_inplace_kernel[grid](A, B, n_elements, BLOCK_SIZE=1024)
    return A
