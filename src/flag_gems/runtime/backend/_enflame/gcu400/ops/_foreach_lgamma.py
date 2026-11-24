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
"""GCU400-specific ``_foreach_lgamma`` / ``_foreach_lgamma_``: 16 pointer slots in one launch.

Design (2026-11-16 perf round, calibrated over 4 fresh bench rounds):

- **Math body**: the pure Triton Stirling body shared with ``gcu400/ops/lgamma_.py``
  (reflection + shift to y>=4 + three-term Stirling series + 5-term odd Taylor sin; the sin
  term count is a 2026-11-16 second-round perf reduction versus the 7 terms in lgamma_.py).
  No libdevice extern at all -- the ``__ef_lgammaf`` extern triggers P16 (SIP abort under
  >=512 concurrent CTAs), and its fp32 vector-width pathology (with 16-bit input the pipeline
  widens to vector<256> and lands on the v2 slow implementation, 16.2ms/16M versus 0.57ms for
  the pure body) makes it unusable at any CTA count. Against the 9-division Lanczos body used by
  the per-tensor version (~99ns/element, where the ``tl.sin`` extern dominates), the Stirling body
  runs ~23-26ns/element.
- **Launch structure**: the NT pointer-slot pattern of ``_amp_foreach_non_finite_check_and_unscale_``
  scaled to 16 slots -- 16 in + 16 out + 16 size explicit parameters, where padding slots with
  size=0 become zero-iteration loops, so one compiled variant serves any list length <=16; longer
  lists are chunked by 16. The per-tensor launch host cost is 16x(JIT dispatch ~10us + dispatch
  floor ~2.4us), so out-of-place spends ~160-250us per call, exceeding the do_bench L2 flush
  shadow window and swallowing small-shape readings into host time ((256,256)x16 measured 0.57x);
  after the single launch the same reading is 7.5x. The grid is fixed at 48 CTAs with an in-slot
  stride (GCU400 Fixed Grid; the grid.y cap and the 65535 grid.x hard limit are both immune for
  any numel).
- **MASKED constexpr**: when every numel % BLOCK == 0 in the batch the hot loop drops the mask
  predicate (all three bench shapes hit it); otherwise the whole batch takes the masked path with
  unchanged correctness.
- **5-term sin Taylor** (2026-11-16 second perf round): the 7-term body truncates at 5.6e-8 error
  for |angle| <= pi/2, three orders of magnitude tighter than the fp32 suite tolerance (atol
  1e-4); the 5-term body (up to angle^11) has 2.3e-6 truncation error, which after amplification
  by the ``err/sin`` ratio still peaks at |angle|=pi/2 and vanishes at the sin zeros (it is not
  amplified in the near-integer pole band), keeping a 44x tolerance margin while removing one
  Horner layer per element: -1.1% (fp32) / -1.4% (fp16) on the 16M single-slot probe. The more
  aggressive 4-term body (up to angle^9, error 1.13e-4) crosses the atol gate and was rejected by
  the odd-size randn cases of the full-domain probe.
- **out-of-place single-buffer allocation** (same round): the host cost of 16 ``empty_like``
  calls is ~45us (~2.8us dispatcher trip each), which under the bench framework process allocator
  state exceeds the do_bench L2 flush shadow window and costs small shapes +10us; uniform-shaped
  lists instead use one ``torch.empty((n, *shape))`` plus ``unbind(0)`` views (~6us).
  Visible difference from ATen: the outputs of a uniform list share one storage (values are
  correct and shape/contiguity/dtype match; user code calling ``resize_`` on the output is not
  supported -- mixed shapes and non-fast-path forms still allocate per tensor / take the shared path).
- **Pole detection fix** (introduced by the Stirling body port, caught by probes): using
  ``reduced = x - floor(x+0.5)`` as the integer criterion lets ``x+0.5`` round first when the fp32
  ulp is >=1 (|x|>=2^23), so negative odd-integer poles in [2^23, 2^24) were missed (a negative
  logspace(1,8) full-domain probe saw 2136/100001 points return a finite value instead of +inf).
  Fix: ``is_pole = (x <= 0) & ((reduced == 0) | (x <= -8388608))`` -- every fp32 value with
  |x|>=2^23 is already an integer, so the constant comparison seals the range with no second
  floor. ``_lgamma_stirling`` in ``gcu400/ops/lgamma_.py`` shares the gap (its full-domain
  verification sampling did not cover that band); this file does not change that operator.

Host guards: non-float, mixed dtype, mixed device, non-contiguous and numel >= 2^31 (outside
the int32 flat offset domain) fall back to the shared ``apply_unary`` grouped path; dtype
promotion (int/bool -> float32) and error semantics match ATen; empty lists and non-Tensor
elements keep the shared path exceptions.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# GCU400 Fixed Grid: 48 CTAs; if the grid is insufficient, shrink according to the total number of tiles (so degenerate
# small batches do not idle).
_GRID_SIZE = 48

# Number of pointer slots carried in one launch; lists with >16 tensors are split into chunks of 16 (one launch per
# chunk).
_SLOTS = 16

# Compute/memory-bound structural floor: BLOCK x num_warps across all combinations ({1024..8192}x{w2,w4}
# with eight configurations within ±0.3%), keeping 2048/w4 = the scalar single-tensor ``lgamma_`` calibration.
_BLOCK = 2048
_NUM_WARPS = 4

# Stirling shift threshold (same parameters as ``lgamma_.py``; interleaved A/B of 3 and 4 showed no difference, so keep
# 4 for accuracy).
_SHIFT_THRESHOLD = 4

# Upper bound of the int32 flat offset domain: max offsets = ceil(n/BLOCK)*BLOCK - 1 <= 2^31 - 1.
_MAX_FLAT_NUMEL = 2**31 - 1


@triton.jit
def _lgamma_stirling(x, SHIFT_THRESHOLD: tl.constexpr):
    # lgamma in fp32 via reflection + shift-to->=SHIFT_THRESHOLD + Stirling
    # asymptotics.  Verbatim body of ``gcu400/ops/lgamma_.py`` except for the
    # pole test fix documented in the module docstring.  No libdevice extern:
    # every op stays on the core Triton path (log/rsqrt/where) at any vector
    # width, which keeps the kernel immune to P16 (extern SIP abort under
    # >=512 concurrent CTAs) and to the extern's 16-bit-widening slowdown.
    reflect = x < 0.5
    y = tl.where(reflect, 1.0 - x, x)
    # Shift y up to >= SHIFT_THRESHOLD, accumulating the product of the skipped
    # factors; lgamma(y) = stirling(y') - log(product).
    t = 1.0
    for _ in tl.static_range(SHIFT_THRESHOLD):
        m = y < SHIFT_THRESHOLD * 1.0
        t = tl.where(m, t * y, t)
        y = tl.where(m, y + 1.0, y)
    logy = tl.log(y)
    # 1/y via rsqrt(y*y): same value for y > 0 (series term is only r/12, so
    # the extra ulp of rsqrt is irrelevant), and it avoids the divide that
    # pushes the 16-bit pipeline onto a slower codegen path on GCU400.
    r = tl.math.rsqrt(y * y)
    r2 = r * r
    series = r * (
        0.08333333333333333 - r2 * (0.002777777777777778 - r2 * 0.0007936507936507937)
    )
    res = (y - 0.5) * logy - y + 0.9189385332046727 + series - tl.log(t)
    # Reflection for x < 0.5: lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x).
    # `reduced` is in [-0.5, 0.5], so |angle| <= pi/2 and a 5-term odd Taylor
    # for sin (through angle^11) is enough: truncation error 2.3e-6 at pi/2
    # (the 7-term body's 5.6e-8 was 3 decades under the fp32 suite atol 1e-4;
    # 2.3e-6 keeps 44x of that margin), and after the err/sin amplification
    # the log|sin| error peaks at |angle| = pi/2 and vanishes toward the sin
    # zeros, so the near-integer pole band is NOT amplified.  Dropping the
    # 6th Taylor level measured -1.1% (fp32) / -1.4% (fp16) per element on
    # the 16M single-slot probe; a 4-level body (through angle^9, error
    # 1.13e-4) crossed the suite atol and was rejected by the full-domain
    # probe (odd-size randn cases).  The Taylor form also keeps sin finite
    # for tiny angles and the 2^64 pre-scaling keeps log|sin| finite when
    # |sin| is subnormal.
    reduced = x - tl.floor(x + 0.5)
    angle = 3.141592653589793 * reduced
    angle2 = angle * angle
    sin_poly = 1.0 - angle2 * (
        0.16666666666666666
        - angle2
        * (
            0.008333333333333333
            - angle2 * (0.0001984126984126984 - angle2 * 2.7557319223985893e-06)
        )
    )
    sin_pi_x = angle * sin_poly
    log_sin = tl.log(tl.abs(sin_pi_x) * 18446744073709551616.0) - 44.3614195558365
    reflected = 1.1447298858494002 - log_sin - res
    result = tl.where(reflect, reflected, res)
    # Gamma's poles (0 and the negative integers) and +-inf map to +inf; NaN
    # propagates through the arithmetic.  The pole test reuses `reduced`
    # (x - round(x) == 0 iff x is an exact integer) instead of a second
    # tl.floor(x).  FIX over lgamma_.py: for |x| >= 2^23 the fp32 addition
    # x + 0.5 rounds FIRST (ulp >= 1), so odd integers in [2^23, 2^24) get
    # reduced == +-1 and the pole is missed.  Every representable fp32 value
    # with |x| >= 2^23 is an integer, and lgamma's poles are the non-positive
    # ones, so the constant comparison closes the gap without a second floor.
    is_pole = (x <= 0.0) & ((reduced == 0.0) | (x <= -8388608.0))
    result = tl.where(is_pole | (tl.abs(x) == float("inf")), float("inf"), result)
    return result


@triton.jit
def _lgamma_slot(
    in_ptr,
    out_ptr,
    pid,
    num_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    SHIFT_THRESHOLD: tl.constexpr,
):
    # Grid-stride over the tiles of one tensor slot.  MASKED=0 requires
    # num_elements % BLOCK == 0 for every live slot (host guarantees it) and
    # removes every mask predicate from the hot loop; size-0 padding slots
    # make the tile loop a zero-trip loop.
    if MASKED:
        num_tile = (num_elements + BLOCK - 1) // BLOCK
    else:
        num_tile = num_elements // BLOCK
    for tile_id in tl.range(pid, num_tile, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        if MASKED:
            mask = offsets < num_elements
            raw = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        else:
            raw = tl.load(in_ptr + offsets)
        result = _lgamma_stirling(raw.to(tl.float32), SHIFT_THRESHOLD)
        out_val = result.to(out_ptr.dtype.element_ty)
        if MASKED:
            tl.store(out_ptr + offsets, out_val, mask=mask)
        else:
            tl.store(out_ptr + offsets, out_val)


@libentry()
@triton.jit(
    do_not_specialize=[
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
    ]
)
def _foreach_lgamma_kernel(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    SHIFT_THRESHOLD: tl.constexpr,
):
    # Fused out-of-place / in-place lgamma over up to 16 same-(device,dtype)
    # tensors in a single launch: each program visits every slot's tile space
    # with the same grid stride, so load balance stays even across differently
    # sized tensors.  In-place calls pass o_k == i_k.
    pid = tl.program_id(0)
    _lgamma_slot(i0, o0, pid, s0, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i1, o1, pid, s1, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i2, o2, pid, s2, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i3, o3, pid, s3, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i4, o4, pid, s4, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i5, o5, pid, s5, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i6, o6, pid, s6, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i7, o7, pid, s7, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i8, o8, pid, s8, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i9, o9, pid, s9, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i10, o10, pid, s10, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i11, o11, pid, s11, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i12, o12, pid, s12, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i13, o13, pid, s13, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i14, o14, pid, s14, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i15, o15, pid, s15, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)


def _fast_path(self) -> bool:
    """True selects the 16-slot single launch: non-empty, all contiguous float, single (dtype, device),
    numel inside the int32 flat offset domain. Any AttributeError (non-Tensor elements, empty
    list) and every other unsatisfied case fall back to the shared ``apply_unary`` path.
    """
    try:
        if not self:
            return False
        dtype = None
        device = None
        for tensor in self:
            if (
                not tensor.is_contiguous()
                or not tensor.dtype.is_floating_point
                or tensor.numel() > _MAX_FLAT_NUMEL
            ):
                return False
            if dtype is None:
                dtype = tensor.dtype
                device = tensor.device
            elif tensor.dtype is not dtype or tensor.device != device:
                return False
    except AttributeError:
        return False
    return True


def _launch_fused(tensors, inplace: bool):
    """Handles <=16 tensors of one (device,dtype) group (the guard already ensures a single group) in one launch;
    longer lists are chunked by 16, and out-of-place allocates outputs first to keep the input order.

    Uniform-shaped lists allocate their out-of-place outputs with one ``torch.empty((n, *shape))``
    plus ``unbind(0)`` views (16 ``empty_like`` host trips cost ~45us, which exceeds the do_bench
    flush shadow window and costs small shapes +10us; single buffer + unbind is ~6us). The outputs
    therefore share one storage; mixed shapes still use per-tensor ``empty_like``."""
    numels = [t.numel() for t in tensors]
    if not inplace:
        if len({t.shape for t in tensors}) == 1:
            stacked = torch.empty(
                (len(tensors), *tensors[0].shape),
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            destinations = list(stacked.unbind(0))
        else:
            destinations = [torch.empty_like(t) for t in tensors]
    with torch_device_fn.device(tensors[0].device):
        for start in range(0, len(tensors), _SLOTS):
            chunk = tensors[start : start + _SLOTS]
            outs = chunk if inplace else destinations[start : start + _SLOTS]
            sizes = [
                numels[start + slot] if slot < len(chunk) else 0
                for slot in range(_SLOTS)
            ]
            total_tiles = 0
            masked = 0
            for numel in sizes:
                total_tiles += (numel + _BLOCK - 1) // _BLOCK
                if numel % _BLOCK:
                    masked = 1
            if total_tiles == 0:
                continue
            grid = (min(_GRID_SIZE, total_tiles), 1, 1)
            _foreach_lgamma_kernel[grid](
                *[
                    chunk[slot] if slot < len(chunk) else chunk[0]
                    for slot in range(_SLOTS)
                ],
                *[
                    outs[slot] if slot < len(outs) else outs[0]
                    for slot in range(_SLOTS)
                ],
                *sizes,
                BLOCK=_BLOCK,
                GRID_SIZE=grid[0],
                MASKED=masked,
                SHIFT_THRESHOLD=_SHIFT_THRESHOLD,
                num_warps=_NUM_WARPS,
            )
    return None if inplace else destinations


def _foreach_lgamma(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA")
    if _fast_path(self):
        return _launch_fused(self, inplace=False)
    return apply_unary("lgamma", self, inplace=False)


def _foreach_lgamma_(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA_")
    if _fast_path(self):
        return _launch_fused(self, inplace=True)
    return apply_unary("lgamma", self, inplace=True)
