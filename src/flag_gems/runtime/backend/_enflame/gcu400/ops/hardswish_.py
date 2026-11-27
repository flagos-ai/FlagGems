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


# GCU400 retune of the in-place hardswish (2026-11-16, interleaved
# do_bench(median) A/B probes on L600, official bench kernel mode):
# the old fixed config (BLOCK=32768, nw2, masked, ieee `/6` body) left the
# 16-bit streams far off the copy floor because the per-element f32 math
# chain (add+max+min+mul+div) was exposed instead of hidden behind wider
# tiles, and put the whole 4096-element core shape on a single CTA
# (grid = cdiv(4096, 32768) = 1). The flat body below keeps the aten
# promote-to-f32 semantics (hardswish(x) = x * min(max(x+3, 0), 6) / 6) but
# swaps the ieee division for a multiply by 1/6 and drops the tail mask
# whenever n divides the tile (EVEN constexpr arm); NaN/-inf inputs keep
# aten semantics because the final multiply propagates them
# (NaN * 0 = NaN, -inf * 0 = NaN), verified value-by-value against
# torch.nn.functional.hardswish on CPU.
#
# Measured (in-place, median, vs old config):
#   16-bit 512M: 941 -> 675us (-28%, 2.96TB/s, copy floor 668)
#   16-bit 16.7M: 36.2 -> 28.4us (-22%)
#   16-bit 4096: 5.73 -> 3.16us (-45%, launch floor 3.11)
#   fp32 512M: 1339 -> 1331us (copy floor 1330)
#   fp32 16.7M: 47.6 -> 46.7us
#   fp32 4096: 6.40 -> 3.27us (-49%)
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def hardswish_kernel_(x_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
            y = x * inner * (1.0 / 6.0)
            tl.store(x_ptr + off, y)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
            y = x * inner * (1.0 / 6.0)
            tl.store(x_ptr + off, y, mask=mask)


def _choose_launch(n_total, itemsize):
    # Boundary probes (fp16/bf16/fp32, 3 interleaved rounds each):
    # - tiny (<=8192) rides the ~3.1-3.2us launch floor with a 2048-wide tile
    #   so several CTAs share the work (old config put 4096 on one CTA);
    # - 65536->8192 and 131072->65536 boundaries verified at n=65536/131072
    #   (8192: 4.07 vs 4.69us, 32768: 4.76 vs 5.74us);
    # - 32768 -> 65536 tile switch measured at 2^21 (32768 wins by 7%) and
    #   2^23 (65536 wins by 12%); boundary placed at 2^22;
    # - 16-bit 65536 -> 131072 switch measured at 2^25 (65536 wins by 2%)
    #   and 2^26 (131072 wins by 0.8%); boundary placed at 2^25. fp32 never
    #   wants 131072 (172.5 vs 170.6us at 2^26; DSM line) and keeps 65536.
    # - num_warps: mid band 16-bit w4==w8 (28.4/28.4us); big 16-bit w4 wins
    #   at B131072 (675.0 vs 683.3 w2); fp32 >=2^22 runs w8 (46.7/1331.5us).
    if n_total <= 8192:
        return min(max(triton.next_power_of_2(n_total), 64), 2048), 4
    if n_total <= 65536:
        return 8192, 8
    if n_total <= (1 << 22):
        return 32768, 8
    if itemsize == 4:
        return 65536, 8
    if n_total > (1 << 25):
        return 131072, 4
    return 65536, 4


def _launch_hardswish_(inp):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.element_size())
    even = n_total % block == 0
    if not even:
        # The tail mask costs nothing at nw2 (685 vs 683us at 512M) but is
        # heavily exposed at nw8 (36.5 vs 28.4us at 16.7M); odd sizes ride
        # the masked arm on nw2.
        num_warps = 2
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (abs.py / sinh.py
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                hardswish_kernel_[(grid_size,)](
                    inp,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "hardswish_ flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def hardswish_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME HARDSWISH_")
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("hardswish_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardswish_: expected a Tensor")
    if not x.is_floating_point():
        raise TypeError("hardswish_: expected a floating point tensor")

    if x.dtype not in _FLAT_DTYPES or x.numel() >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.hardswish_ import hardswish_ as _public

        return _public(x)

    if not x.is_contiguous():
        # Keep the historical round-trip for strided inputs (sinh_.py
        # pattern): the generic KernelGen fallback would crash on the
        # unbounded cdiv(n, 1024) grid for large tensors, while the flat
        # kernel above is grid-capped and safe at any size.
        x_work = x.contiguous()
        _launch_hardswish_(x_work)
        x.copy_(x_work)
        return x

    _launch_hardswish_(x)
    return x
