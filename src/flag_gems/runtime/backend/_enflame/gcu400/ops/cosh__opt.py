"""GCU400-specific Triton implementation of cosh: an out-of-place 1R1W flat grid-stride kernel.

Replaces the shared pointwise_dynamic codegen path (ENFLAME default config tile4096/block-ptr/
limited CTAs, 2.25ms at 512M fp16). The math body splits by dtype (probe probe_r1/r2,
do_bench median w1000/r100, new compiler TopsRider 20261116, L600):

- fp32 uses the grouped double-exp body `0.5*(exp(x)+exp(-x))`: 1396us at 512M, on the copy floor
  (1402us -- both exp calls are fully hidden by bandwidth); the distributed form
  `exp(x)*0.5+exp(-x)*0.5` regresses 22% in fp32 (1776us) -- the grouped/distributed quirk goes
  the opposite way versus the 16-bit tier (the mirror of the sinh precedent), and replacing one
  exp with IEEE div / rsqrt gains nothing in fp32 (2134us; rejected for the sinh/cosh_ family).
- fp16/bf16 use the single-exp body `t + rsqrt(t)^2` (t=exp(x)): in the 16-bit tier the second
  transcendentals of the double-exp body cannot be hidden by memory (512M: distributed 1128us /
  grouped 1565us). The identity cosh(x)=0.5*(t+1/t) replaces the reciprocal with a squared hardware
  rsqrt primitive: a constant-numerator `/` division still takes the slow path when the denominator
  is a bare exp result rather than a fresh add (1287us; the silu fast-reciprocal precedent does not
  transfer), while the rsqrt(t)^2 body gives 928us at 512M and 35.1us at 16.7M.
  +-inf/NaN/intermediate-overflow semantics are bit-identical to the double-exp body: on exp
  overflow t=inf, rsqrt(inf)=0 and its square is 0, so the result is still inf, and
  both bodies return inf for large |x|.
EVEN mask-free arm (drops the boundary predicate when N_total % BLOCK == 0): pow2-divisible tiers
gain 512M fp16 -2.6% and fp32 -4.2%; odd sizes in 16-bit take masked+nw2 (the mask predicate costs
+12% at nw4), while fp32 masked keeps nw4. The BLOCK/nw ladder is calibrated by boundary probes
over 2^12..2^29 (probe_r2.txt).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cosh import cosh_func as _common_cosh_func
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Number of resident CTAs aligned with the GCU400 SIP topology; reused inside the kernel via a strided loop
NUM_SIPS = 48

# int32 offset safety range: masked tail blocks are also evaluated for masked lanes
# block_id*BLOCK+arange, with upper bound = num_blocks*BLOCK-1; when BLOCK<=131072,
# n_total <= 2^31-2^17 ensures the flat int32 offset does not wrap around to a negative value.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cosh_f32_flat_kernel(
    x_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK + arange
        if EVEN:
            value = tl.load(x_ptr + offsets).to(tl.float32)
            result = 0.5 * (tl.exp(value) + tl.exp(-value))
            tl.store(out_ptr + offsets, result)
        else:
            mask = offsets < N_total
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            result = 0.5 * (tl.exp(value) + tl.exp(-value))
            tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cosh_16bit_flat_kernel(
    x_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK + arange
        if EVEN:
            value = tl.load(x_ptr + offsets).to(tl.float32)
            # cosh(x) = 0.5*(t + 1/t); 1/t is lowered to hardware primitives via the rsqrt(t)^2 identity,
            # eliminating the second transcendental evaluation and IEEE division slow paths in the double-exp body
            t = tl.exp(value)
            reciprocal_sqrt = tl.rsqrt(t)
            result = 0.5 * (t + reciprocal_sqrt * reciprocal_sqrt)
            tl.store(out_ptr + offsets, result)
        else:
            mask = offsets < N_total
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            t = tl.exp(value)
            reciprocal_sqrt = tl.rsqrt(t)
            result = 0.5 * (t + reciprocal_sqrt * reciprocal_sqrt)
            tl.store(out_ptr + offsets, result, mask=mask)


def _choose_block(n_total, is_fp32):
    """Small shapes use a power-of-two tile over many CTAs to amortize launch; large shapes tier by dtype."""
    if n_total <= 8192:
        return min(max(triton.next_power_of_2(n_total), 256), 2048), 1
    if is_fp32:
        # 2^16: 8192/nw4 (4.13 vs 4.67us @nw2); 2^18: 16384/nw4;
        # 2^21/2^22: 32768/nw4 (12.8/17.6us); 2^24: 65536/nw4 (51.2us);
        # 512M: 65536/nw4 (1396us at the floor; nw8 is equally fast and follows the family convention)
        if n_total <= (1 << 16):
            return 8192, 4
        if n_total <= (1 << 18):
            return 16384, 4
        if n_total <= (1 << 22):
            return 32768, 4
        return 65536, 4
    # 16-bit: 2^16: 4096/nw2 (4.07us); 2^18: 16384/nw4 (4.67us);
    # 2^21/2^22: 32768/nw4 (10.0/13.8-14.1us); 2^24: 65536/nw4 (35.1us);
    # 2^26: 65536/nw4 (114.8-116.2us); 512M: 131072/nw4 (official bench
    # 899-901/873-875us; nw2 is worse at 943-954us, and nw8 is -2.8% by probe measurement, but by the official
    # measurement
    # fp16 is unchanged and bf16 regresses by +4-6%; disproven and rolled back -- probe gains must be rechecked in
    # production form)
    if n_total <= (1 << 16):
        return 4096, 2
    if n_total <= (1 << 18):
        return 16384, 4
    if n_total <= (1 << 22):
        return 32768, 4
    if n_total <= (1 << 26):
        return 65536, 4
    return 131072, 4


def _cosh_forward(A):
    """Plain (no autograd) cosh: the flat grid-stride kernel or the common codegen path."""
    # Non-floating-point / very large tensors use the common codegen path to preserve promotion and stride semantics
    if A.dtype not in _FLAT_DTYPES or A.numel() >= _INT32_SAFE_LIMIT:
        return _common_cosh_func(A)
    inp = A.contiguous()
    out = torch.empty_like(inp)
    n_total = inp.numel()
    if n_total == 0:
        return out

    is_fp32 = inp.dtype == torch.float32
    block, num_warps = _choose_block(n_total, is_fp32)
    even = n_total % block == 0
    if not even and not is_fp32:
        # Tail-block mask predicates show up at 16-bit nw4 (+12%); reduce odd sizes to nw2
        num_warps = 2
    kernel = _cosh_f32_flat_kernel if is_fp32 else _cosh_16bit_flat_kernel
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS)
    # Occasional cold-compilation failures in the external llvm converter (P02 toolchain with converterResult=-2
    # flake family); 3 launch retries ride out the transient window, and deterministic corruption still raises on the
    # 3rd attempt
    # (precedents in the same family: sinh.py / cosh___opt.py / atan2_.py)
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "GEMS_ENFLAME cosh flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


class CoshFunction(torch.autograd.Function):
    """Autograd entry for cosh.

    The flat kernel path above returns a plain tensor, so without this wrapper
    the graph is cut and `aten::cosh` autograd (i.e. `cosh_backward`) can never
    run. The backward keeps the saved input and calls the backend Triton
    `cosh_backward` kernel (grad * sinh(x)).
    """

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return _cosh_forward(A)

    @staticmethod
    def backward(ctx, grad_output):
        (A,) = ctx.saved_tensors
        from .cosh_backward import cosh_backward

        return cosh_backward(grad_output, A)


def cosh(A):
    logger.debug("GEMS_ENFLAME COSH")
    if torch.is_grad_enabled() and A.requires_grad:
        return CoshFunction.apply(A)
    return _cosh_forward(A)


__all__ = ["cosh"]
