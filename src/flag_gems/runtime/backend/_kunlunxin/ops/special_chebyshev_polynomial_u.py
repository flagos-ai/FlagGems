import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 2048
_NUM_WARPS = 8


@triton.jit
def _chebyshev_polynomial_u_tensor_n_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask)
        n_f32 = tl.load(n_ptr + offs, mask=mask).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs)
        n_f32 = tl.load(n_ptr + offs).to(tl.float32)
    x_f32 = x.to(tl.float32)

    # U_0 is the constant polynomial 1.0 for every x, including +-inf and NaN.
    # Building it as `x * 0.0 + 1.0` would poison it (inf * 0.0 = NaN),
    # so the constant tensor is materialised directly.
    ukm2 = tl.full(x_f32.shape, 1.0, tl.float32)
    ukm1 = 2.0 * x_f32
    result = tl.where(n_f32 < 0.5, ukm2, ukm1)

    for k in tl.static_range(2, 6):
        uk = 2.0 * x_f32 * ukm1 - ukm2
        result = tl.where(tl.abs(n_f32 - k) < 0.5, uk, result)
        ukm2, ukm1 = ukm1, uk

    if NEED_MASK:
        tl.store(out_ptr + offs, result.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(x.dtype))


@triton.jit
def _chebyshev_polynomial_u_scalar_n_kernel(
    x_ptr,
    n_idx: tl.constexpr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask)
    else:
        x = tl.load(x_ptr + offs)
    x_f32 = x.to(tl.float32)

    if n_idx == 0:
        # U_0 == 1.0 for every x, including +-inf and NaN (see tensor-n kernel).
        result = tl.full(x_f32.shape, 1.0, tl.float32)
    elif n_idx == 1:
        result = 2.0 * x_f32
    elif n_idx == 2:
        t = x_f32 * x_f32
        result = 4.0 * t - 1.0
    elif n_idx == 3:
        t = x_f32 * x_f32
        result = (8.0 * t - 4.0) * x_f32
    elif n_idx == 4:
        t = x_f32 * x_f32
        result = (16.0 * t - 12.0) * t + 1.0
    else:
        t = x_f32 * x_f32
        result = (32.0 * t - 32.0) * t * x_f32 + 6.0 * x_f32

    if NEED_MASK:
        tl.store(out_ptr + offs, result.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(x.dtype))


def _launch_flat(x, n, numel):
    out = torch.empty_like(x)
    grid = (triton.cdiv(numel, _BLOCK),)
    need_mask = (numel % _BLOCK) != 0
    if isinstance(n, torch.Tensor):
        _chebyshev_polynomial_u_tensor_n_kernel[grid](
            x,
            n,
            out,
            numel,
            BLOCK=_BLOCK,
            NEED_MASK=need_mask,
            num_warps=_NUM_WARPS,
        )
    else:
        _chebyshev_polynomial_u_scalar_n_kernel[grid](
            x,
            n,
            out,
            numel,
            BLOCK=_BLOCK,
            NEED_MASK=need_mask,
            num_warps=_NUM_WARPS,
        )
    return out


def special_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_chebyshev_polynomial_u only supports float32/float64, got {x.dtype}"
        )
    if x.numel() == 0:
        return torch.empty_like(x)
    x = x.contiguous()

    if isinstance(n, torch.Tensor):
        n_ref = n.detach().to("cpu", dtype=torch.int32)
        n_min = int(n_ref.amin().item())
        n_max = int(n_ref.amax().item())
        n = n.to(device=x.device, dtype=torch.int32)
    else:
        n_min = n_max = int(n)
        # ATen's Scalar overload truncates a non-integer n toward zero (that is
        # also what the generic implementation does via `torch.tensor(int(n))`).
        # The kernel selects its branch with exact `n_idx == k` comparisons, so
        # the raw float would fall through to the final `else` (U_5) instead of
        # the truncated order.  Normalise before handing it over as a constexpr.
        n = n_min

    if n_max > 5 or n_min < 0:
        raise ValueError(
            f"Chebyshev polynomial order n must be in [0, 5], "
            f"got values in [{n_min}, {n_max}]"
        )

    if isinstance(n, torch.Tensor) and n.shape != x.shape:
        n = torch.broadcast_to(n, x.shape).contiguous()

    return _launch_flat(x, n, x.numel())
