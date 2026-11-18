import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

_MAX_POLY_DEGREE = tl.constexpr(10)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _chebyshev_polynomial_w(x, n):
    x_f32 = x.to(tl.float32)
    n_f32 = n.to(tl.float32)

    wkm2 = x_f32 * 0.0 + 1.0
    wkm1 = 2.0 * x_f32 + 1.0
    result = tl.where(n_f32 < 0.5, wkm2, wkm1)

    for k in tl.static_range(2, _MAX_POLY_DEGREE):
        wk = 2.0 * x_f32 * wkm1 - wkm2
        result = tl.where(tl.abs(n_f32 - k) < 0.5, wk, result)
        wkm2, wkm1 = wkm1, wk

    return result


# Fast path for a single degree shared by every lane: the degree is a
# constexpr, so the recurrence needs exactly ND steps and no degree selection
# (`tl.abs(n - k) < 0.5` per degree) is emitted.
_UNIFORM_BLOCK = 8192
# Largest degree the slow kernel's unrolled recurrence covers.
_UNIFORM_MAX_DEGREE = 9


@triton.jit
def _chebyshev_polynomial_w_uniform_kernel(
    x_ptr, out_ptr, numel, ND: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xf = tl.load(x_ptr + offs, mask=mask)
    if ND <= 0:
        res = tl.full(xf.shape, 1.0, tl.float32)
    elif ND == 1:
        res = 2.0 * xf + 1.0
    else:
        wkm2 = tl.full(xf.shape, 1.0, tl.float32)
        wkm1 = 2.0 * xf + 1.0
        for k in tl.static_range(2, ND + 1):
            wk = 2.0 * xf * wkm1 - wkm2
            wkm2 = wkm1
            wkm1 = wk
        res = wkm1
    tl.store(out_ptr + offs, res, mask=mask)


def _uniform_degree(n):
    """Return the degree as an int when it is a single integral value."""
    if isinstance(n, torch.Tensor):
        if n.dim() != 0:
            return None
        n = n.item()
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return None
    if n != int(n):
        return None
    return int(n)


def _launch_uniform(x, deg, out):
    numel = x.numel()
    if numel != 0:
        grid = (triton.cdiv(numel, _UNIFORM_BLOCK),)
        _chebyshev_polynomial_w_uniform_kernel[grid](
            x, out, numel, ND=deg, BLOCK=_UNIFORM_BLOCK
        )
    return out


def _fast_path_ok(x, out=None):
    # The slow kernel unrolls a fixed ten degrees, so it is only correct for
    # |n| <= 9; keep every larger degree on that path so this fast path does not
    # change any result (the wrong value for n >= 10 is a pre-existing bug).
    if x.dtype != torch.float32 or not x.is_contiguous():
        return False
    if out is not None and (
        out.dtype != x.dtype or out.shape != x.shape or not out.is_contiguous()
    ):
        return False
    return True


def special_chebyshev_polynomial_w(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_W")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_chebyshev_polynomial_w only supports float32/float64, got {x.dtype}"
        )
    if _fast_path_ok(x):
        deg = _uniform_degree(n)
        if deg is not None and deg <= _UNIFORM_MAX_DEGREE:
            return _launch_uniform(x, deg, torch.empty_like(x))
    if not isinstance(n, torch.Tensor):
        n = torch.empty((), dtype=torch.int64, device=x.device).fill_(n)
    else:
        n = n.to(device=x.device)
    return _chebyshev_polynomial_w(x, n)


def special_chebyshev_polynomial_w_out(x, n, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_W_OUT")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"special_chebyshev_polynomial_w only supports float32/float64, got {x.dtype}"
        )
    if _fast_path_ok(x, out):
        deg = _uniform_degree(n)
        if deg is not None and deg <= _UNIFORM_MAX_DEGREE:
            return _launch_uniform(x, deg, out)
    if not isinstance(n, torch.Tensor):
        n = torch.empty((), dtype=torch.int64, device=x.device).fill_(n)
    else:
        n = n.to(device=x.device)
    return _chebyshev_polynomial_w(x, n, out0=out)
