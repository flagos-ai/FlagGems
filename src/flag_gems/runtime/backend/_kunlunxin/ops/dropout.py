import os
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

try:  # hand-written-payload injection point (XPU cluster path)
    import triton.experimental.tle as tle

    _HAS_TLE = True
except Exception:  # pragma: no cover - toolchain without tle.raw
    tle = None
    _HAS_TLE = False


@triton.jit
def _dropout_mix32(z, key):
    """Counter-based 32-bit avalanche mixer (multiply-low only)."""
    z = z ^ key
    z = z ^ (z >> 16)
    z = z * 0x7FEB352D
    z = z ^ (z >> 15)
    z = z * 0x045D9F3B
    z = z ^ (z >> 16)
    return z


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 8
    seed = philox_seed.to(tl.int64)
    off = philox_offset.to(tl.int64)
    c0 = (off & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((off >> 32) & 0xFFFFFFFF).to(tl.uint32)
    key = (
        c1
        ^ (seed & 0xFFFFFFFF).to(tl.uint32)
        ^ ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    )

    pid = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    thr = (p * 65536.0).to(tl.uint32)
    cb = c0 + pid * BLOCK * 4 + ar
    z0 = _dropout_mix32(cb, key)
    z1 = _dropout_mix32(cb + BLOCK, key)
    z2 = _dropout_mix32(cb + 2 * BLOCK, key)
    z3 = _dropout_mix32(cb + 3 * BLOCK, key)
    mask0 = (z0 >> 16) > thr
    mask1 = (z0 & 0xFFFF) > thr
    mask2 = (z1 >> 16) > thr
    mask3 = (z1 & 0xFFFF) > thr
    mask4 = (z2 >> 16) > thr
    mask5 = (z2 & 0xFFFF) > thr
    mask6 = (z3 >> 16) > thr
    mask7 = (z3 & 0xFFFF) > thr
    scale = 1.0 / (1.0 - p)

    off_0 = pid * BLOCK * UNROLL + ar
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK

    x0 = tl.load(X + off_0)
    x1 = tl.load(X + off_1)
    x2 = tl.load(X + off_2)
    x3 = tl.load(X + off_3)
    x4 = tl.load(X + off_4)
    x5 = tl.load(X + off_5)
    x6 = tl.load(X + off_6)
    x7 = tl.load(X + off_7)

    tl.store(Y + off_0, x0 * scale * tl.where(mask0, 1.0, 0.0))
    tl.store(Y + off_1, x1 * scale * tl.where(mask1, 1.0, 0.0))
    tl.store(Y + off_2, x2 * scale * tl.where(mask2, 1.0, 0.0))
    tl.store(Y + off_3, x3 * scale * tl.where(mask3, 1.0, 0.0))
    tl.store(Y + off_4, x4 * scale * tl.where(mask4, 1.0, 0.0))
    tl.store(Y + off_5, x5 * scale * tl.where(mask5, 1.0, 0.0))
    tl.store(Y + off_6, x6 * scale * tl.where(mask6, 1.0, 0.0))
    tl.store(Y + off_7, x7 * scale * tl.where(mask7, 1.0, 0.0))
    tl.store(dropout_mask + off_0, mask0)
    tl.store(dropout_mask + off_1, mask1)
    tl.store(dropout_mask + off_2, mask2)
    tl.store(dropout_mask + off_3, mask3)
    tl.store(dropout_mask + off_4, mask4)
    tl.store(dropout_mask + off_5, mask5)
    tl.store(dropout_mask + off_6, mask6)
    tl.store(dropout_mask + off_7, mask7)


@libentry()
@triton.jit(do_not_specialize=["scale"])
def dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offset < N
        m = tl.load(dropout_mask + offset, mask=mask, other=0)
        dy = tl.load(DY + offset, mask=mask, other=0)
        dx = dy * (m.to(tl.int32) & 1).to(dy.dtype) * scale
        tl.store(DX + offset, dx, mask=mask)
    else:
        m = tl.load(dropout_mask + offset)
        dy = tl.load(DY + offset)
        dx = dy * (m.to(tl.int32) & 1).to(dy.dtype) * scale
        tl.store(DX + offset, dx)


UNROLL = 8
_MIN_BLOCK = 64


def _dropout_launch_config(N):
    if N <= 16384:
        blk, warps = 1024, 16
    elif N <= 131072:
        blk, warps = 2048, 16
    else:
        blk, warps = 4096, 32
    if N % (blk * UNROLL) != 0:
        b = blk
        while b >= _MIN_BLOCK:
            if N % (b * UNROLL) == 0:
                blk = b
                warps = max(4, min(32, b // 64))
                break
            b //= 2
    assert blk >= _MIN_BLOCK
    return blk, warps


def _dropout_backward_launch_config(N, dtype):
    if N <= 65536:
        return 1024, 16
    if N <= 4 * 1024 * 1024:
        if dtype == torch.float32:
            return 8192, 16
        return 32768, 16
    return 131072, 32


# ---------------------------------------------------------------------------
# Hand-written SIMD .xpu payload (cluster path; flag_gems' own kernel, not a
# call into xdnn/native).  It removes both plain-Triton walls documented in
# harness/solution/dropout/plan.md section 10:
#   * the bool mask store (`arith.extui`/`trunci`/`fptosi` have no vector form on
#     this backend) is done with `vfloat2ufix8_*`, packing four 16-lane f32 mask
#     vectors into one 64-byte u8 vector with values strictly 0/1;
#   * the RNG is a vectorised counter-based avalanche (`svxor/svsrl/svmul` on
#     `uint32x16_t`), one 32-bit word per TWO elements (high 16 bits -> element
#     i+j, low 16 bits -> element i+16+j).
# Critical measured detail: a SIMD masked store must NEVER take a runtime mask
# operand -- `vstore_lm_int8x64_mz(dst, src, keep)` with a runtime `keep` is
# ~150us slower on 16.7M than the same store with a constant mask.  The u8 path
# below always stores with the constant ~0ULL.
# ---------------------------------------------------------------------------

_SIMD_CB = 1024
_SIMD_MIN_N = 1024
_SIMD_GRID_MIN = 32
_SIMD_GRID_MAX = 512

_PAYLOAD_OBJ = os.path.join(
    os.path.dirname(__file__), "..", "payload", "obj", "drop.o"
)

if _HAS_TLE:
    _SIMD_DNS = ["N", "KEY", "THR", "SCALE", "VFP", "NP"]

    @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
    def dr_f32(x, y, mb, n, key, thr, scale, vfp, PID, NPROG):
        ...

    @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
    def dr_f16(x, y, mb, n, key, thr, scale, vfp, PID, NPROG):
        ...

    @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
    def dr_bf16(x, y, mb, n, key, thr, scale, vfp, PID, NPROG):
        ...

    @triton.jit(do_not_specialize=_SIMD_DNS)
    def _dr_k_f32(X, Y, MB, N, KEY, THR, SCALE, VFP, NP):
        tle.raw.call(dr_f32, (X, Y, MB, N, KEY, THR, SCALE, VFP, tl.program_id(0), NP))

    @triton.jit(do_not_specialize=_SIMD_DNS)
    def _dr_k_f16(X, Y, MB, N, KEY, THR, SCALE, VFP, NP):
        tle.raw.call(dr_f16, (X, Y, MB, N, KEY, THR, SCALE, VFP, tl.program_id(0), NP))

    @triton.jit(do_not_specialize=_SIMD_DNS)
    def _dr_k_bf16(X, Y, MB, N, KEY, THR, SCALE, VFP, NP):
        tle.raw.call(dr_bf16, (X, Y, MB, N, KEY, THR, SCALE, VFP, tl.program_id(0), NP))

    # bf16 cannot use bf16 SIMD arithmetic on this XPU3 part
    # (`vvmul_bfloat16x32` raises an illegal instruction), so dr_bf16 does the
    # whole computation on the fp32 bit pattern instead.
    _SIMD_KERNELS = {
        torch.float32: _dr_k_f32,
        torch.float16: _dr_k_f16,
        torch.bfloat16: _dr_k_bf16,
    }
else:  # pragma: no cover
    _SIMD_KERNELS = {}


def _s32(v):
    """Reinterpret a 32-bit value as a signed int32 (the payload takes i32)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _simd_grid(n):
    nwork = (n + _SIMD_CB - 1) // _SIMD_CB
    g = (max(nwork, 1) + 63) // 64
    g = max(_SIMD_GRID_MIN, min(_SIMD_GRID_MAX, g))
    return (triton.next_power_of_2(g),)


def dropout(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN NATIVE_DROPOUT_FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = input.device
    input = input.contiguous()
    N = input.numel()
    scale = 1.0 / (1.0 - p)
    thr = _s32(int(p * 65536.0))

    kernel = _SIMD_KERNELS.get(input.dtype)
    if kernel is not None and _SIMD_MIN_N <= N <= 0x7FFFFFFF:
        grid = _simd_grid(N)
        out = torch.empty_like(input)
        mask = torch.empty_like(input, dtype=torch.bool)
        with torch_device_fn.device(device):
            philox_seed, philox_offset = philox_backend_seed_offset(triton.cdiv(N, 2))
            key = _s32(
                int(philox_offset >> 32)
                ^ (int(philox_seed) & 0xFFFFFFFF)
                ^ ((int(philox_seed) >> 32) & 0xFFFFFFFF)
            )
            kernel[grid](
                input, out, mask.view(torch.uint8), N, key, thr, scale, scale, grid[0]
            )
        return out, mask

    # ---- plain-Triton fallback (flag_gems' own kernel, not an external call)
    BLOCK, num_warps = _dropout_launch_config(N)
    tile = BLOCK * UNROLL
    grid = (triton.cdiv(N, tile),)
    padded = grid[0] * tile
    if padded == N:
        x = input
    else:
        x = torch.empty(padded, dtype=input.dtype, device=device)
        x[:N].copy_(input.view(-1))
    out_full = torch.empty(padded, dtype=input.dtype, device=device)
    mask_full = torch.empty(padded, dtype=torch.bool, device=device)
    increment = triton.cdiv(N, 2)
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        dropout_forward_kernel[grid](
            x,
            out_full,
            mask_full,
            p,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out_full[:N].view(input.shape), mask_full[:N].view(input.shape)


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS_KUNLUNXIN DROPOUT_BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    BLOCK, num_warps = _dropout_backward_launch_config(N, grad_output.dtype)
    grid = (triton.cdiv(N, BLOCK),)
    with torch_device_fn.device(grad_output.device):
        dropout_backward_kernel[grid](
            grad_output,
            grad_input,
            mask.contiguous().view(torch.int8),
            N,
            scale,
            BLOCK=BLOCK,
            NEED_MASK=N % BLOCK != 0,
            num_warps=num_warps,
        )
    return grad_input
