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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK = 8192
MIN_BLOCK = 512
NUM_CLUSTERS = 12


def _pick_block(n_elements):
    # Bucket the tile to a power of two in [MIN_BLOCK, MAX_BLOCK] so the kernel
    # compiles at most ~5 times total (no per-shape recompilation / IR
    # explosion) while still splitting small/medium tensors across the 12 XPU
    # clusters instead of running on 1-2 programs.
    target = triton.next_power_of_2(max(1, triton.cdiv(n_elements, NUM_CLUSTERS)))
    return max(MIN_BLOCK, min(MAX_BLOCK, target))


@libentry()
@triton.jit(do_not_specialize=["n_elements", "beta", "threshold"])
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    z = x * beta
    soft_z = tl.where(z > threshold, z, tl.log(1 + tl.exp(z)))
    out = soft_z / beta
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS")
    x = self.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    block_size = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(x.device):
        softplus_kernel[grid](
            x, out, n_elements, beta, threshold, BLOCK_SIZE=block_size
        )
    return out
