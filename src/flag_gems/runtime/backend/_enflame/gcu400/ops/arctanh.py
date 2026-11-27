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

"""GCU400-specific arctanh / arctanh_out: persistent grid-stride loop that fixes the grid.x overflow."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
FLAT_GRID = NUM_SIPS * 2

# Size ladder (L600 / GCU400, do_bench w1000/r100 median, benchmark core
# shapes). The math body is byte-identical to the in-place twin
# `_enflame/gcu400/ops/arctanh_.py` (2026-11-16 full RSI sweep), so the ladder
# transfers directly; the out-of-place variant was re-confirmed at the two
# large buckets where read-x/write-y traffic could shift the optimum.
#   n <= SMALL_N: BLOCK 512 / nw 2 — chunk-latency amortization at tiny sizes
#   ((64,64) = 4096 elems: 3.4-3.7 us vs 4.7 us for one wide masked CTA).
#   16-bit and n >= HUGE_16BIT_N: BLOCK 131072 / nw 2 — edges out B8192/nw4 by
#   ~1.2% at 512Mi (256KB DSM tile, arcsin-proven ceiling); 12% worse at 16.7M,
#   hence the separate arm.
#   otherwise: BLOCK 8192 / nw 4 — full family sweep optimum.
SMALL_N = 16384
HUGE_16BIT_N = 1 << 28


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def arctanh_kernel(x_ptr, y_ptr, N_total, BLOCK: tl.constexpr):
    # atanh(x) = 0.5 * log((1 + x) / (1 - x)). The division is eliminated with
    # the hardware rsqrt identity (1-x)^-1 == rsqrt(1-x)^2: on GCU400 tl.rsqrt
    # is a ~free vector primitive while a real fp32 division is not
    # (family-measured on the arctanh_ twin: fp32 512Mi 2789 -> 2419 us,
    # native parity 0.977x -> 1.127x). Domain semantics propagate naturally
    # with no branches:
    #   |x| > 1 -> rsqrt(negative) = NaN -> log(NaN) = NaN   (matches torch)
    #   x == +1 -> rsqrt(0) = +inf -> (2)*inf*inf = +inf -> log = +inf
    #   x == -1 -> (1+x) = 0 -> log(0) = -inf
    # The final 0.5 is folded into the OUTPUT dtype domain (cast first, then
    # multiply): multiplying by 0.5 in the fp32 domain before the 16-bit cast
    # triggers a compiler degradation (+700 us@512Mi fp16, family-measured).
    #
    # Persistent grid-stride: grid.x is capped at FLAT_GRID on the host, so
    # the 512Mi core shapes (cdiv = 524288 at BLOCK=1024) can never exceed the
    # 65535 hardware limit that aborted the previous one-block-per-program
    # layout (bench core first-touch, P19/flaggems_arctanh).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        r = tl.rsqrt(1.0 - x)
        val = tl.log((1.0 + x) * r * r)
        y = val.to(y_ptr.dtype.element_ty) * 0.5
        tl.store(y_ptr + off, y, mask=mask)


def _launch(x_contig, y_contig, N_total, dtype):
    if N_total <= SMALL_N:
        BLOCK, num_warps = 512, 2
    elif dtype.itemsize == 2 and N_total >= HUGE_16BIT_N:
        # 16-bit huge: B131072/nw2 edges out B8192/nw4 by ~1.2% at 512Mi;
        # at 16.7M the wide block is 12% worse, hence the separate arm.
        # DSM: 131072 x 2B = 256KB tile (arcsin-proven ceiling).
        BLOCK, num_warps = 131072, 2
    else:
        BLOCK, num_warps = 8192, 4
    grid_size = min((N_total + BLOCK - 1) // BLOCK, FLAT_GRID)
    with torch_device_fn.device(x_contig.device):
        arctanh_kernel[(grid_size,)](
            x_contig, y_contig, N_total, BLOCK=BLOCK, num_warps=num_warps
        )


def arctanh(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ARCTANH")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    N_total = x_contig.numel()
    if N_total == 0:
        return out.view_as(x)
    _launch(x_contig, out, N_total, x_contig.dtype)
    return out.view_as(x)


def arctanh_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME ARCTANH_OUT")
    assert x.shape == out.shape, "Input and output shapes must match"

    # Compute in x's own precision; the kernel casts to out.dtype at the store
    # (Triton auto-converts), avoiding an extra dtype-copy round trip.
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32
    x_contig = x.to(result_dtype).contiguous()

    N_total = x_contig.numel()
    if N_total == 0:
        return out
    if out.is_contiguous():
        _launch(x_contig, out, N_total, x_contig.dtype)
        return out
    out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
    _launch(x_contig, out_contig, N_total, x_contig.dtype)
    out.copy_(out_contig)
    return out
