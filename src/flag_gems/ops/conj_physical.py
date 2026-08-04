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

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conj_physical"),
    key=["n_elements"],
)
@triton.jit
def conj_physical_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    base = offsets * 2
    real = tl.load(in_ptr + base, mask=mask)
    imag = tl.load(in_ptr + base + 1, mask=mask)

    tl.store(out_ptr + base, real, mask=mask)
    tl.store(out_ptr + base + 1, -imag, mask=mask)


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS Conj_Physical")
    if not input.is_complex():
        return input

    # If input has conj bit set, resolve it first so view_as_real won't crash.
    if input.is_conj():
        input = input.resolve_conj()

    n_elements = input.numel()
    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(src)
    in_real_ptr = torch.view_as_real(src)
    out_real_ptr = torch.view_as_real(output)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    conj_physical_kernel[grid](in_real_ptr, out_real_ptr, n_elements)

    return output
