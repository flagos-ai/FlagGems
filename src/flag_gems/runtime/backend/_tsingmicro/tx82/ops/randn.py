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

"""
torch.randn via TX81 hardware PRNG (tle.dsa.tsingmicro.randn → __RandGen / xorshift128+).

Avoids Philox int-vector IR that falls back to RISC-V scalar loops on TX81.

P0: host only derives two int64 scalars (no CPU randint / seed H2D); the kernel
expands them to the 16 xorshift streams with pid/lane mixing.

Small-N: DSA launch has a fixed host/runtime floor; for numel below
``FLAGGEMS_RANDN_CPU_FALLBACK_MAX`` (default 16384) use CPU randn + H2D.
"""

import logging
import os

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.runtime import device, get_heuristic_config, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)

_U64 = (1 << 64) - 1

# Below this numel, CPU generate + H2D beats on-device DSA (launch floor now
# ~0.12ms after txModuleLoad path; crossover ~16K–64K). Override with
# FLAGGEMS_RANDN_CPU_FALLBACK_MAX.
_CPU_FALLBACK_MAX = int(os.environ.get("FLAGGEMS_RANDN_CPU_FALLBACK_MAX", "16384"))


def _splitmix64(x: int) -> int:
    """SplitMix64 — cheap host mix to turn Philox state into xorshift seeds."""
    z = (x + 0x9E3779B97F4A7C15) & _U64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _U64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _U64
    return z ^ (z >> 31)


def _to_i64(u: int) -> int:
    u &= _U64
    if u >= (1 << 63):
        u -= 1 << 64
    return u


def _philox_to_dsa_seed_scalars(philox_seed, philox_offset):
    """
    Map Philox (seed, offset) → two signed int64 scalars for the kernel.

    No tensor alloc / H2D. Kernel expands these into 16 streams.
    """
    s = int(philox_seed) & _U64
    o = int(philox_offset) & _U64
    # Two independent mixes; OR 1 avoids the all-zero xorshift fixed point.
    seed0 = _splitmix64(s ^ ((o << 1) & _U64)) | 1
    seed1 = _splitmix64((s + 0xA0761D6478BD642F) ^ (o * 3)) | 1
    return _to_i64(seed0), _to_i64(seed1)


def _should_cpu_fallback(
    n: int, device, cpu_fallback_max: int = _CPU_FALLBACK_MAX
) -> bool:
    """Small-N CPU hybrid decision; ``cpu_fallback_max`` lets callers like
    ``normal`` use their own measured crossover (see normal.py)."""
    if n > cpu_fallback_max or n <= 0:
        return False
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return dev.type != "cpu"


def _native_cpu_randn(size, dtype, device):
    """CPU randn + H2D without re-entering FlagGems aten overrides."""
    cpu = torch.ops.aten.randn.default(
        size,
        dtype=dtype,
        layout=torch.strided,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    return cpu.to(device=device, dtype=dtype)


@libentry()
@triton.heuristics(get_heuristic_config("randn"))
@triton.jit
def randn_kernel(
    out_ptr,
    seed0,  # scalar i64
    seed1,  # scalar i64
    N,
    BLOCK: tl.constexpr,
):
    """
    One program produces ``BLOCK`` Normal(0, 1) samples via ``tle.dsa.tsingmicro.randn``.
    ``BLOCK`` must be a multiple of 32.

    ``seed0`` / ``seed1`` are host-side scalars; each program expands them to
    16 xorshift streams mixed with ``pid`` / lane (hardware seed writeback is
    currently a no-op, so independence comes from this mix).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    pid64 = pid.to(tl.int64)
    lane = tl.arange(0, 16).to(tl.int64)
    # Signed-safe multipliers (unsigned golden-ratio constants overflow int64).
    s0 = seed0 + pid64 * (-7046029254386353131) + lane
    s1 = seed1 + pid64 * (-4412409329100832499) + (lane << 1)
    s0 = s0 | 1
    s1 = s1 | 1

    vals, _s0, _s1 = tle.dsa.tsingmicro.randn(s0, s1, BLOCK)
    tl.store(out_ptr + offs, vals, mask=mask)


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_TSINGMICRO RANDN (tle.dsa.tsingmicro.randn)")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    N = volume(size)
    if N == 0:
        return torch.empty(size, device=device, dtype=dtype)

    # Advance Philox state for FlagGems RNG accounting (both paths).
    philox_seed, philox_offset = philox_backend_seed_offset(max(N, 1))

    # float64: DSA randn is float32-only; keep CPU for correctness.
    if dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO RANDN f64 → CPU")
        return _native_cpu_randn(size, dtype, device)

    # Small tensors: DSA launch + Box-Muller floor (~0.5ms) loses to CPU+H2D.
    if _should_cpu_fallback(N, device):
        logger.debug(
            "GEMS_TSINGMICRO RANDN cpu fallback N=%s (max=%s)", N, _CPU_FALLBACK_MAX
        )
        return _native_cpu_randn(size, dtype, device)

    out = torch.empty(size, device=device, dtype=dtype)
    seed0, seed1 = _philox_to_dsa_seed_scalars(philox_seed, philox_offset)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out.view(-1), seed0, seed1, N)
    return out
