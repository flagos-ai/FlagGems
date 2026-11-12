"""
Tensor.exponential_ via TX81 hardware PRNG (tle.dsa.tsingmicro.rand → __RandGen).

Exp(lambd) samples: X = -log(U) / lambd, U ~ Uniform(0, 1).

Avoids Philox int-vector IR that falls back to RISC-V scalar loops on TX81.
Mirrors the ``randn`` DSA seed / launch pattern.
"""

import logging
import os

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.runtime import get_heuristic_config, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import _philox_to_dsa_seed_scalars

logger = logging.getLogger(__name__)

# Below this numel, CPU fill + H2D beats on-device DSA. Measured on txda:
# DSA is already faster at 4096 elements, so the fallback floor is 4096.
# Override with FLAGGEMS_EXPONENTIAL_CPU_FALLBACK_MAX. (Independent of the
# randn fallback threshold, which is not yet re-validated at 4096.)
_CPU_FALLBACK_MAX = int(os.environ.get("FLAGGEMS_EXPONENTIAL_CPU_FALLBACK_MAX", "4096"))


def _native_cpu_exponential_(x, lambd: float):
    """CPU exponential_ + H2D without re-entering FlagGems aten overrides.

    Manual construction: X = -log(U) / lambd, U ~ Uniform(0, 1).  Each step
    (uniform_, clamp_min_, log_, neg_, div_) is an SIMD-vectorized CPU pass,
    which is ~5x faster than aten::exponential_ (a scalar rejection loop).
    The ops run on a CPU tensor, so FlagGems' PrivateUse1 (txda) overrides
    cannot recurse into this path.
    """
    cpu = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
    cpu.uniform_(0.0, 1.0)
    cpu.clamp_min_(torch.finfo(x.dtype).tiny)
    cpu.log_().neg_().div_(lambd)

    x.copy_(cpu, non_blocking=True)
    return x


@libentry()
@triton.heuristics(get_heuristic_config("exponential_"))
@triton.jit
def exponential_kernel(
    out_ptr,
    seed0,  # scalar i64
    seed1,  # scalar i64
    N,
    inv_lambd,
    BLOCK: tl.constexpr,
):
    """
    One program fills ``BLOCK`` Exp(lambd) samples via ``tle.dsa.tsingmicro.rand``.

    ``BLOCK`` must be a multiple of 32.  ``inv_lambd = 1 / lambd``.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    pid64 = pid.to(tl.int64)
    lane = tl.arange(0, 16).to(tl.int64)
    s0 = seed0 + pid64 * (-7046029254386353131) + lane
    s1 = seed1 + pid64 * (-4412409329100832499) + (lane << 1)
    s0 = s0 | 1
    s1 = s1 | 1

    u, _s0, _s1 = tle.dsa.tsingmicro.rand(s0, s1, BLOCK)
    # Typed f32 constants via `u * 0.0 + c` — bare Python floats become f64 and
    # crash MK→Tx81 AddVS (only 16/32-bit element types are legal).
    eps = u * 0.0 + 1.0e-7
    one = u * 0.0 + 1.0
    u = u * (one - (eps + eps)) + eps
    inv = u * 0.0 + inv_lambd
    vals = -inv * tl.log(u)
    tl.store(out_ptr + offs, vals, mask=mask)


def exponential_(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_TSINGMICRO EXPONENTIAL_")
    if lambd <= 0:
        raise ValueError(f"lambd must be positive, got {lambd}")

    dtype = x.dtype
    device = x.device
    assert dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"unsupported dtype for exponential_: {dtype}"

    N = x.numel()
    if N == 0:
        return x

    philox_seed, philox_offset = philox_backend_seed_offset(
        max(N, 1), generator=generator
    )

    # float64: DSA rand is float32-only; keep CPU for correctness.
    if dtype is torch.float64:
        logger.debug("GEMS_TSINGMICRO EXPONENTIAL_ f64 → CPU")
        return _native_cpu_exponential_(x, lambd)

    # Small tensors: launch floor loses to CPU fill + H2D into ``x``.
    if N <= _CPU_FALLBACK_MAX and device.type != "cpu":
        logger.debug(
            "GEMS_TSINGMICRO EXPONENTIAL_ cpu fallback N=%s (max=%s)",
            N,
            _CPU_FALLBACK_MAX,
        )
        return _native_cpu_exponential_(x, lambd)

    inv_lambd = 1.0 / float(lambd)
    seed0, seed1 = _philox_to_dsa_seed_scalars(philox_seed, philox_offset)

    # Prefer writing f32 scratch then cast for half types if needed.
    # In-place on float32/contiguous half: write through a float32 view when
    # element size matches; otherwise fill a float32 buffer and copy.
    if dtype is torch.float32 and x.is_contiguous():
        out = x
    else:
        out = torch.empty(x.shape, device=device, dtype=torch.float32)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(device):
        exponential_kernel[grid_fn](out.view(-1), seed0, seed1, N, inv_lambd)

    if out is not x:
        x.copy_(out.to(dtype=dtype))
    return x
