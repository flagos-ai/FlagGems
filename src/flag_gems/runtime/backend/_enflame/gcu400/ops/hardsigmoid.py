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

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def hardsigmoid_kernel(
    x_ptr, out_ptr, N_total, BLOCK_SIZE: tl.constexpr, MASKED: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        # GCU400 retune (2026-11-21): keeping the tail predicate on every
        # load/store costs 1.15-1.45x because the predicated path drops the wide
        # vector access. When BLOCK_SIZE divides N_total exactly the predicate is
        # provably all-true, so it is compiled out (MASKED=False from the host).
        # 16.7M fp16 in-place: 83.7us -> 47.5us; 16.7M fp32: 131.4us -> 67.0us.
        if MASKED:
            x = tl.load(x_ptr + off, mask=mask)
        else:
            x = tl.load(x_ptr + off)
        xf = x.to(tl.float32)
        y = tl.fma(xf, 1.0 / 6.0, 0.5)
        # aten.hardsigmoid propagates NaN through clamp; all clamp forms below
        # keep NaN, unlike tl.minimum/tl.maximum which drop it.
        #
        # GCU400 retune (2026-11-17): the f32-domain where-clamp was the only
        # math-chain cost above the copy floor (fma+cvt alone run at the
        # floor: 670us vs copy 668us on 512Mi). Round to the output dtype
        # FIRST and clamp there -- bit-exact vs clamp-then-round since
        # round-to-nearest is monotonic and 0/1 are exactly representable
        # (only corner: a signed-zero difference of |diff|=0).
        # - fp16: 16-bit compares/selects run packed; clamping in the fp16
        #   domain closes the whole gap. 512Mi fp16: 767.6us -> 676.6us
        #   (native 675.9us); 16.7M fp16: 33.0us -> 28.8us (native 30.4us).
        # - bf16: compares promote to f32 (no packed bf16 ALU; bf16-domain
        #   selects regressed 512Mi to 907us), so saturate the uint16 bit
        #   pattern instead: positive patterns are monotonic, negatives have
        #   bit 15 set, NaN has magnitude > 0x7F80. Restore NaN after the
        #   negative->zero select so -NaN survives (aten clamp keeps it).
        #   512Mi bf16: 767.6us -> 702.5us (native 676.1us); 16.7M bf16:
        #   33.0us -> 30.5us (native 30.5us). Verified bit-identical to CPU
        #   aten on randn + +/-NaN/inf/-0/boundary specials.
        # - fp32: unchanged (identity round).
        if x.dtype == tl.float16:
            y = y.to(tl.float16)
            y = tl.where(y < 0, 0, tl.where(y > 1, 1, y))
        elif x.dtype == tl.bfloat16:
            bits = y.to(tl.bfloat16).to(tl.uint16, bitcast=True)
            sat = tl.minimum(bits, tl.full((), 0x3F80, tl.uint16))
            sat = tl.where(
                bits >= tl.full((), 0x8000, tl.uint16),
                tl.full((), 0, tl.uint16),
                sat,
            )
            mag = bits & tl.full((), 0x7FFF, tl.uint16)
            y = tl.where(mag > tl.full((), 0x7F80, tl.uint16), bits, sat).to(
                tl.bfloat16, bitcast=True
            )
        else:
            y = tl.where(y < 0.0, 0.0, tl.where(y > 1.0, 1.0, y))
            y = y.to(x.dtype)
        if MASKED:
            tl.store(out_ptr + off, y, mask=mask)
        else:
            tl.store(out_ptr + off, y)


# GCU400 retune (2026-11-16, interleaved do_bench(median) A/B, 3 rounds;
# dtype-specialized saturating body, see hardsigmoid_kernel): the f32 math
# chain is the limiter for 16-bit sizes, so multi-SIP tiles with few warps
# hide it (32K/nw4 at the 16.7M band, 128K/nw2 at the 512Mi band); fp32 big
# sits on the memory floor with B65536/nw8. Configs verified for out-of-place
# and in-place; bf16 mirrors fp16 within noise. 2026-11-20 retune of the
# 16.7M/4096 bands is documented inline in _choose_launch.
def _choose_launch(n_total, itemsize):
    if n_total <= 2048:
        return max(triton.next_power_of_2(n_total), 64), 4
    if n_total <= 8192:
        # 2026-11-20 benchmark A/B at n=4096: nw=2 3.54us vs nw=4 3.59-3.95us
        # (noise band), 1024/nw8 4.07us; nw=2 keeps the 2-CTA tile.
        return 2048, 2
    if n_total <= 524288:
        return 8192, 8
    if n_total <= (1 << 21):
        return 65536, 4
    # 2026-11-21 retune with the predicate compiled out: 64K/nw8 is the best
    # single recipe for both the 16.7M band and the 512Mi band, for every dtype
    # (in-place A/B below, native in brackets).
    # 16.7M: fp16 47.5us [52.2], bf16 55.1us [51.8], fp32 67.0us [67.6]
    # 512Mi : bf16 1564us [1292] (was 1631us with 128K/nw2 + mask)
    return 65536, 8


def _launch_hardsigmoid(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.element_size())
    # A tile that divides n_total exactly lets the kernel drop the tail
    # predicate; otherwise keep the masked path (and its wider grid cap, which
    # is what the masked tile measured best with).
    masked = n_total % block != 0
    grid_cap = NUM_SIPS * 2 if masked else NUM_SIPS
    grid_size = min(triton.cdiv(n_total, block), grid_cap)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # neg.py / square.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                hardsigmoid_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK_SIZE=block,
                    MASKED=masked,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "hardsigmoid flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(t):
    return (
        t.is_contiguous() and t.dtype in _FLAT_DTYPES and t.numel() < _INT32_SAFE_LIMIT
    )


def hardsigmoid(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME HARDSIGMOID")
    if not _flat_path_ok(x):
        from flag_gems.ops.hardsigmoid import hardsigmoid as _public

        return _public(x)
    out = torch.empty_like(x)
    _launch_hardsigmoid(x, out)
    return out


def hardsigmoid_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME HARDSIGMOID_OUT")
    assert x.numel() == out.numel()
    if not (_flat_path_ok(x) and _flat_path_ok(out)):
        from flag_gems.ops.hardsigmoid import hardsigmoid_out as _public

        return _public(x, out)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    _launch_hardsigmoid(x, out)
    return out
