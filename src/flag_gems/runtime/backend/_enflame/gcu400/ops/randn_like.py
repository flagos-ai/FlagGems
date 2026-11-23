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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import UNROLL, randn_kernel

logger = logging.getLogger(__name__)


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ENFLAME RANDN_LIKE")
    if device is None:
        device = x.device.index
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    # Dispatch ladder (GCU400, official do_bench kernel mode, median,
    # warmup 1000ms / rep 100ms, spreads <0.1%):
    #   N<=8192 latency band: B128/nw1 (2.7us @4096 vs 3.10 B1024/nw4).
    #   N<=128K transition:   B1024/nw4 (96K: 3.7us; B128 there costs 6.4us).
    #   Beyond 128K:          B4096/nw2 for every dtype, and fp32 joins the
    #   REDUCED pairing (one 32-bit hash per output pair, 16-bit uniforms)
    #   that the 16-bit lanes already use: 262K 5.6->4.7us, 1M 12.1->10.4,
    #   4M 31.5->25.7, 16.7M 103.4->87.8, 1Gi 5722->5326 (2D grid form).
    #   fp32 keeps full 24-bit uniforms in the two small bands, where the
    #   hash budget is not the bottleneck and the finer uniforms are free.
    #   REDUCED-fp32 quality battery (2^26 samples, fp64, 3 calls): marginal
    #   mean/std/skew clean; excess kurt +0.003 from the u==0 clamp tail
    #   spike, identical to the shipped 16-bit lanes; within-pair
    #   cross-moment +6..9e-4 from fmix32 high/low-half correlation, same
    #   magnitude as the shipped fp16/bf16 lanes (8x below the vetoed
    #   B16384 marginal skew); 512-bin chi2 0.93-1.04 (24-bit path: 1.75,
    #   aten native GCU: 18.3); WIDE 2^31+12345 fp32 smoke finite.
    #   B16384 stays out of every lane: at that tile the paired hash inputs
    #   differ by XOR 0x4000 exactly, and fmix32 leaves that pair correlated
    #   enough to bias the cos-lane marginal (skew +0.0068 across seeds,
    #   2^26 samples, fp64-accumulated; every other BLOCK tested stays
    #   within +/-0.002).  Larger REDUCED tiles also measured slower
    #   (1Gi fp32: B8192+ 8.2-8.5ms vs 5.33ms at B4096).
    if N <= 8192:
        BLOCK, num_warps = 128, 1
    elif N <= 131072:
        BLOCK, num_warps = 1024, 4
    else:
        BLOCK, num_warps = 4096, 2
    # Grid form: 1D while the program count fits grid.x (<=65535 on this
    # backend), otherwise split the program axis in 2D so smaller tiles stay
    # launchable for huge N (e.g. 1Gi B4096 needs 65536 programs, any dtype).
    total = triton.cdiv(N, BLOCK * UNROLL)
    if total <= 65535:
        grid = (total,)
        grid_x, flat = 0, True
    else:
        grid_y = triton.cdiv(total, 32768)
        grid_x = triton.cdiv(total, grid_y)
        grid = (grid_x, grid_y)
        flat = False
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(x.device):
        # fp32 goes REDUCED only past the transition band (see ladder note);
        # the two small bands keep the full 24-bit uniforms.
        reduced = dtype != torch.float32 or N > 131072
        randn_kernel[grid](
            out,
            N,
            grid_x,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            REDUCED=reduced,
            MASKED=N % (BLOCK * UNROLL) != 0,
            FLAT=flat,
            WIDE=N > 2**31 - BLOCK * UNROLL,
            num_warps=num_warps,
        )
    return out
