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

NUM_SIPS = 24
FLAT_GRID = NUM_SIPS * 2

# Size ladder (L600 / GCU400, do_bench w1000/r100 median, benchmark core shapes):
# - n <= SMALL_N: BLOCK 512 / nw 2. At (64, 64) = 4096 elems this measures
#   ~3.6 us vs 4.7 us for BLOCK 8192 / nw 4 (single half-masked CTA); best of
#   {512,1024,2048,4096,8192} x {1,2,4} on all three dtypes.
# - 16-bit and n >= HUGE_16BIT_N: BLOCK 131072 / nw 2 edges out B8192/nw4 by
#   ~1.2% at 512Mi (2120 vs 2145 us; 256KB DSM tile, arcsin-proven ceiling).
# - otherwise: BLOCK 8192 / nw 4, grid = min(cdiv, 48). Full sweep of
#   BLOCK {8192..131072} x nw {1,2,4,8,16} and grid {48..65535} confirms this
#   is the optimum for everything else: wider blocks reach a lower copy floor
#   (fp16 512Mi: 673 us@B65536/nw2) but give it back through higher
#   exposed-ALU cost at mid sizes (16.7M fp16: B131072 is 12% worse).
SMALL_N = 16384
HUGE_16BIT_N = 1 << 28


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def arctanh_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    # atanh(x) = 0.5 * log((1 + x) / (1 - x)). The division is eliminated with
    # the hardware rsqrt identity (1-x)^-1 == rsqrt(1-x)^2: on GCU400 tl.rsqrt
    # is a ~free vector primitive while a real fp32 division is not
    # (family-measured: fp32 512Mi 2789 -> 2419 us, native parity 0.977x ->
    # 1.127x). Domain semantics propagate naturally with no branches:
    #   |x| > 1 -> rsqrt(negative) = NaN -> log(NaN) = NaN   (matches torch)
    #   x == +1 -> rsqrt(0) = +inf -> (2)*inf*inf = +inf -> log = +inf
    #   x == -1 -> (1+x) = 0 -> log(0) = -inf
    # The final 0.5 is folded into the OUTPUT dtype domain (cast first, then
    # multiply): multiplying by 0.5 in the fp32 domain before the 16-bit cast
    # triggers a compiler degradation (+700 us@512Mi fp16, family-measured).
    #
    # GCU400/L600 measured cost model (512Mi elems, official bench kernel mode,
    # do_bench w1000/r100 median): kernel time = copy(BLOCK) + ALU(BLOCK), i.e.
    # DMA and ALU never overlap on this backend (num_stages/warp_specialize/
    # flatten on tl.range are all no-ops; grid > 48 only adds CTA overhead).
    # Before the rsqrt swap: fp32 copy 1444 + div 241 + log 1105 = 2789 us
    # (native 2726 = same structure); fp16 copy 898 + div 769 + log 532 = 2199
    # (native 3094). Alternative formulations all measured slower: two-log
    # 4932/3410 us, extern atanh 11674/7389, log1p extern 16758/12640,
    # log2-based 2969/3008, reciprocal-mul 3859/3240, fast-div == plain div,
    # odd minimax poly deg-5/10/11 2590-6758 us (a Horner step costs ~230 us per
    # op-slot at 512Mi, so deg-10 exceeds the whole log+div chain).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        r = tl.rsqrt(1.0 - x)
        val = tl.log((1.0 + x) * r * r)
        y = val.to(x_ptr.dtype.element_ty) * 0.5
        tl.store(x_ptr + off, y, mask=mask)


def arctanh_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME ARCTANH_")
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if not isinstance(x, torch.Tensor):
        raise TypeError("arctanh_ expects a single Tensor argument")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_floating_point():
        raise TypeError("arctanh_ only supports floating point tensors")

    N_total = x.numel()
    if N_total == 0:
        return x

    if N_total <= SMALL_N:
        BLOCK, num_warps = 512, 2
    elif x.element_size() == 2 and N_total >= HUGE_16BIT_N:
        # 16-bit huge: B131072/nw2 edges out B8192/nw4 by ~1.2% at 512Mi
        # (2118 vs 2144 us); at 16.7M the wide block is 12% worse, hence the
        # separate arm. DSM: 131072 x 2B = 256KB tile (arcsin-proven ceiling).
        BLOCK, num_warps = 131072, 2
    else:
        BLOCK, num_warps = 8192, 4
    grid_size = min((N_total + BLOCK - 1) // BLOCK, FLAT_GRID)
    with torch_device_fn.device(x.device):
        arctanh_kernel_[(grid_size,)](x, N_total, BLOCK=BLOCK, num_warps=num_warps)
    return x
