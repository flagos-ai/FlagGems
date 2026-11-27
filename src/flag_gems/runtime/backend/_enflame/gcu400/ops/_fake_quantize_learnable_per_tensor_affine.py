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
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# out = clamp(rint(x / scale), qmin - zp, qmax - zp) * scale
#
#   (identical to the aten reference: (clamp(rint(x/scale)+zp, qmin, qmax)-zp)*s;
#   zp is an integer inside [qmin, qmax] after the reference rounds+clamps it,
#   so translating the clamp window is bit-exact for every representable input
#   and saves the add+sub of zp from the hot loop.)
#
# * scale / zero_point are single-element device tensors read inside the
#   kernel, so there is no host synchronisation per call.
# * The division mirrors the aten reference bit-for-bit: the reference computes
#   ``inv_scale = 1.0f / scale`` (a correctly-rounded fp32 reciprocal) and then
#   ``x * inv_scale``, so the kernel uses ``rcp_rn`` followed by a multiply.
#   A plain ``x / scale`` flips round-half-to-even at exact half-way quotients.
# * rint in the hot loop is the fp32 round-to-nearest-even trick
#   ``(v + 12582912.0) - 12582912.0`` (12582912 = 1.5 * 2**23), NOT the gcu
#   libdevice nearbyint extern: that extern lowers to a ~10x slower path and
#   capped this kernel at 210-620 GB/s while the trick reaches ~2.8-3.2 TB/s
#   (measured 2026-11-18, Triton GCU 4.1.6; see add.py for the same flat
#   pointwise recipe).  The product is pre-clamped to the translated window
#   [lo - 0.25, hi + 0.25] first, which (a) keeps |v| < 2**23 where the trick
#   is exactly round-to-nearest-even for every quantisation range that fp32
#   can represent, and (b) puts a min/max between the multiply and the magic
#   add so the backend cannot contract them into an FMA -- the enflame
#   compiler fuses mul+add regardless of the enable_fp_fusion launch option
#   (that option is accepted but never consumed), and an FMA would round the
#   product only once, flipping exact-tie inputs (x=11.85, scale=0.1: fp32
#   product 118.5 -> ties-even 118, FMA exact-product -> 119).  The scalar
#   zero_point keeps the libdevice nearbyint (its cost is one op per program,
#   not per element).
# * The kernel runs a fixed grid-stride loop over the flat range
#   (grid = min(cdiv(n, BLOCK), 48), stride = tl.num_programs(0)); engaging all
#   48 SPs is what saturates HBM (32-way grids lose a third of the bandwidth,
#   see add.py).  The band-tuned (BLOCK, num_warps) ladder was measured with
#   the core benchmark protocol (do_bench kernel median, warmup 1000ms /
#   rep 100ms) on 2026-11-18 (final body, benchmark record run):
#     fp16 1Gi: B131072/nw2 1.63ms (~2.6TB/s);  16M: B65536/nw2 34.0us;
#     bf16 1Gi: B131072/nw2 1.58ms;  fp32 1Gi: B65536/nw2 2.94ms (~3.1TB/s);
#     fp32 16M: B32768/nw2 50.5us;  2M band: B8192/nw4;  4K: B4096/nw2 ~4.7us
#   (launch floor).  The backend compile-cache key drops num_warps, so a given
#   (dtype, BLOCK) always maps to a single num_warps value across bands.
#   Re-measured 2026-11-20 on the current board (Triton 3.6.0, TOPS 4.1.5) with
#   the same protocol: the fp32 bands were leaving the vector unit idle at
#   num_warps=2, so they moved to nw4 (fp32 1Gi 3.36 -> 2.99ms, fp32 16M
#   68.9 -> 59.6us).  The two-byte bands showed no reproducible win under nw4
#   (the 1D and 4D cases of the same element count disagree by 7%, which tracks
#   the 2GB allocation's address rather than the launch config), so they keep
#   the 2026-11-18 numbers.
# * Per-program DSM is BLOCK * itemsize bytes (compiler-accounted); every
#   ladder entry stays inside the 448 KB double-program budget
#   (917504 // 2), so two programs co-reside per SP.
# * The hot loop walks whole tiles only, so it carries no bounds mask; a single
#   partial tail tile is handled once by program 0.
# * All arithmetic happens in fp32; the result is stored back in the input
#   dtype.
# * The launch is retried twice on failure: the external llvm converter
#   intermittently fails on cold compiles (converterResult=-2 family, P02);
#   recompiling succeeds (sinh.py / atan2_.py / add.py precedent).

_NUM_SP = 48


def _pick_config(n, itemsize):
    """(BLOCK, num_warps) ladder, measured on gcu400.  The grid is
    min(cdiv(n, BLOCK), 48) so big shapes run a grid-stride loop on all 48
    SPs while smaller shapes keep one tile per program.

    The per-element vector body (clamp + round + rescale) dominates the two-byte
    dtypes, so the ladder is tuned for instruction throughput rather than for
    the raw tile footprint: two warps per program leave the vector unit idle on
    the large fp32 bands and four warps measure 11-14% faster there (2026-11-20,
    core benchmark protocol on this board)."""
    if n <= 4096:
        block = 1 << max(10, (n - 1).bit_length())
        return min(block, 4096), 2
    if itemsize == 2:
        if n <= (1 << 23):
            return 8192, 4
        if n <= (1 << 25):
            return 65536, 2
        return 131072, 2
    if n <= (1 << 22):
        return 8192, 4
    if n <= (1 << 26):
        return 32768, 4
    return 65536, 4


@triton.jit
def _fake_quantize_learnable_per_tensor_affine_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    zp_ptr,
    n_elements,
    qmin_val,
    qmax_val,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    s = tl.load(scale_ptr).to(tl.float32)
    inv_scale = tl_extra_shim.rcp_rn(s)
    zpf = tl.load(zp_ptr).to(tl.float32)
    # aten rounds the zero point to nearest-even, then clamps it; translating
    # the clamp window by the (integer) zero point is bit-exact.
    zpf = tl_extra_shim.nearbyint(zpf)
    zpf = tl.minimum(tl.maximum(zpf, qmin_val), qmax_val)
    lo = qmin_val - zpf
    hi = qmax_val - zpf
    # Pre-clamp the scaled value to [lo - 0.25, hi + 0.25] before rounding:
    # every representable input outside the window rounds into it and then to
    # a level inside [lo, hi], so the post-rint clamp disappears.  0.25 is
    # never a rounding tie and the window keeps |r| < 2**23, where the
    # 1.5*2**23 trick below is exactly round-to-nearest-even.  The min/max
    # between the multiply and the magic add also makes an FMA contraction of
    # the product impossible (the enflame backend fuses mul+add regardless
    # of enable_fp_fusion; a fused mul+add would round once and flip
    # exact-tie inputs like x=11.85, scale=0.1 -> 118.5 -> 118, fma -> 119).
    lo_in = lo - 0.25
    hi_in = hi + 0.25

    offs = tl.arange(0, BLOCK)
    n_full = n_elements // BLOCK
    for tile in range(pid, n_full, num_pids):
        offsets = tile * BLOCK + offs
        x = tl.load(x_ptr + offsets).to(tl.float32)
        r = tl.minimum(tl.maximum(x * inv_scale, lo_in), hi_in)
        r = (r + 12582912.0) - 12582912.0
        y = r * s
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty))

    if (pid == 0) & (n_full * BLOCK != n_elements):
        offsets = n_full * BLOCK + offs
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.minimum(tl.maximum(x * inv_scale, lo_in), hi_in)
        r = (r + 12582912.0) - 12582912.0
        y = r * s
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _fake_quantize_learnable_per_tensor_affine(
    self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE")

    x = self
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor([float(scale)], dtype=torch.float32, device=x.device)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.as_tensor(
            [float(zero_point)], dtype=torch.float32, device=x.device
        )
    qmin = (
        int(quant_min)
        if not isinstance(quant_min, torch.Tensor)
        else int(quant_min.item())
    )
    qmax = (
        int(quant_max)
        if not isinstance(quant_max, torch.Tensor)
        else int(quant_max.item())
    )

    block, num_warps = _pick_config(n, x.element_size())
    grid = min((n + block - 1) // block, _NUM_SP)
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                _fake_quantize_learnable_per_tensor_affine_kernel[(grid,)](
                    x,
                    out,
                    scale,
                    zero_point,
                    n,
                    float(qmin),
                    float(qmax),
                    BLOCK=block,
                    num_warps=num_warps,
                    num_stages=1,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "fake_quantize_learnable_per_tensor_affine launch failed "
                "(attempt %d): %s",
                attempt + 1,
                exc,
            )
    return out
