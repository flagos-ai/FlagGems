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

logger = logging.getLogger(__name__)

# GCU400: 4 DIE × 6 SIP, resident CTA count aligned with the topology
NUM_SIPS = 24
MAX_GRID = NUM_SIPS * 2
BLOCK_LARGE = 65536
BLOCK_MID = 8192
# 2026-11-23 sweep on the official protocol (GEO over the 15 core cells, 2.080x at HEAD):
# 32-bit tile 32768/131072, grid 96, 16-bit tile 32768, 16-bit slab, 4 warps for 16-bit,
# and mid tier 4096/16384 all landed in 2.017x-2.072x, i.e. neutral to slightly worse, and a
# libdevice acosh instead of the rsqrt form measured 0.301x. The geometry below stays.


@libentry()
@triton.jit
def arccosh_kernel(
    x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr, SLAB: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    if SLAB:
        per_pid = (num_blocks + num_pids - 1) // num_pids
        begin = pid * per_pid
        end = tl.minimum(begin + per_pid, num_blocks)
        step = 1
    else:
        begin = pid
        end = num_blocks
        step = num_pids
    for block_id in tl.range(begin, end, step):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        # sqrt(x^2-1) = (x^2-1) * rsqrt(max(x^2-1, 1e-30)): eliminates the software sqrt
        q = x * x - 1.0
        y = tl.log(x + q * tl.rsqrt(tl.maximum(q, 1e-30)))
        if EVEN:
            tl.store(out_ptr + off, y)
        else:
            tl.store(out_ptr + off, y, mask=mask)


def _choose_config(num_elements, dtype):
    """Returns (BLOCK, num_warps, slab): tile picked by element count and dtype, see the module docstring."""
    if num_elements <= 32768:
        return max(1024, triton.next_power_of_2(num_elements)), 4, False
    if num_elements < 1 << 22:
        return BLOCK_MID, 4, False
    # fp32 uses 4 warps + contiguous segmentation (contiguous segmentation evens out HBM channel hot spots on a 2GB
    # buffer);
    # 16-bit halves the traffic, so 2 warps + grid-stride is better (see module docstring for measurements).
    if dtype == torch.float32:
        return BLOCK_LARGE, 4, True
    return BLOCK_LARGE, 2, False


def _launch(inp, out):
    num_elements = inp.numel()
    if num_elements == 0:
        return out
    block_size, num_warps, slab = _choose_config(num_elements, inp.dtype)
    grid_size = min(triton.cdiv(num_elements, block_size), MAX_GRID)
    with torch_device_fn.device(inp.device):
        arccosh_kernel[(grid_size,)](
            inp,
            out,
            num_elements,
            BLOCK=block_size,
            EVEN=(num_elements % block_size == 0),
            SLAB=slab,
            num_warps=num_warps,
        )
    return out


def _arccosh(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def arccosh_(A):
    logger.debug("GEMS_ENFLAME ARCCOSH_")
    if not A.is_contiguous():
        A.copy_(_arccosh(A))
        return A
    return _launch(A, A)
