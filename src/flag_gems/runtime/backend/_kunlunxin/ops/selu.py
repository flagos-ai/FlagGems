import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
#         = scale * where(x > 0, x, alpha * (exp(x) - 1))
# i.e. elu(x, alpha, scale, input_scale=1).
#
# 2026-08-19 perf closure (task #285) replaced the tuned pointwise_dynamic
# 512-lane tile with the flat kernel below (launch/ALU-bound on XPU for
# mid/large N, fp16 [4096,4096] 0.627ms vs torch 0.210ms).
# 2026-09-09 closure: on XPU tl.exp is scalarized (exp.f.rn/vextracti.f, no
# vload/vstore) and the tl.where select costs ~0.24ms, so the flat kernel
# now evaluates expm1(t) with t = clamp(min(x,0), -10, 0) using a degree-8
# Taylor polynomial on u = t/16 (|u| <= 0.625, truncation < 5e-8) plus four
# squarings p = p*(2+p): expm1(u) -> expm1(16u) = exp(t)-1. Pure float
# mul/add stays fully vectorized; measured ~2.3x (fp32) / ~2.8x (fp16)
# faster than the tl.exp flat kernel on [4096,4096], max error ~5e-7 in
# fp32, and it also beats the pointwise path for bf16 large N (0.243ms vs
# 0.584ms), so the bf16-big exception was removed. The pointwise_dynamic
# kernel below remains only as the non-contiguous fallback.
_ALPHA = tl.constexpr(1.6732632423543772848170429916717)
_SCALE = tl.constexpr(1.0507009873554804934193349852946)

# ---- pointwise_dynamic path (non-contiguous fallback) ----
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def selu_func(x):
    x_fp32 = x.to(tl.float32)
    return _SCALE * tl.where(x_fp32 > 0, x_fp32, _ALPHA * (tl.exp(x_fp32) - 1.0))


# ---- flat path: uncovered contiguous blocks, masked tail only ----
_TIERS = (
    (16384, 2048, 4),
    (262144, 8192, 8),
    (None, 16384, 16),
)


@triton.jit
def selu_flat_kernel(
    A,
    O,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(A + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(A + offsets)

    xf = x.to(tl.float32)
    # Arithmetic form of SELU with no tl.exp (scalarized on XPU) and no
    # tl.where (select costs ~0.24ms): expm1(t) for t = clamp(min(x,0),-10,0)
    # via degree-8 Taylor on u = t/16 + 4 squarings p = p*(2+p).
    x_pos = tl.maximum(xf, 0.0)
    t = tl.maximum(tl.minimum(xf, 0.0), -10.0)
    u = t * 0.0625  # t / 16, |u| <= 0.625
    p = 2.4801587301587302e-05  # 1/40320
    p = p * u + 1.9841269841269841e-04  # 1/5040
    p = p * u + 1.3888888888888889e-03  # 1/720
    p = p * u + 8.3333333333333332e-03  # 1/120
    p = p * u + 4.1666666666666667e-02  # 1/24
    p = p * u + 1.6666666666666666e-01  # 1/6
    p = p * u + 0.5
    p = p * u + 1.0
    p = p * u  # expm1(u)
    p = p * (2.0 + p)  # expm1(2u)
    p = p * (2.0 + p)  # expm1(4u)
    p = p * (2.0 + p)  # expm1(8u)
    p = p * (2.0 + p)  # expm1(16u) = exp(t) - 1
    y = _SCALE * x_pos + (_ALPHA * _SCALE) * p

    if NEED_MASK:
        tl.store(O + offsets, y.to(x.dtype), mask=mask)
    else:
        tl.store(O + offsets, y.to(x.dtype))


def _pick_tier(numel):
    for hi, block, warps in _TIERS:
        if hi is None or numel <= hi:
            return block, warps
    return 16384, 16


def _use_flat(A):
    # Fast path: any contiguous tensor (masked tail handled by NEED_MASK).
    # The polynomial kernel also beats the pointwise path for bf16-large N,
    # so no dtype/numel exception is needed.
    return A.is_contiguous()


def _launch_flat(A, out):
    n_elements = A.numel()
    if n_elements == 0:
        return out
    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    selu_flat_kernel[grid](
        A.reshape(-1),
        out,
        n_elements,
        BLOCK_SIZE=block,
        NEED_MASK=need_mask,
        num_warps=warps,
    )
    return out


def selu(A):
    logger.debug("GEMS_KUNLUNXIN SELU")
    if _use_flat(A):
        return _launch_flat(A, torch.empty_like(A))
    return selu_func(A)


def selu_(A):
    logger.debug("GEMS_KUNLUNXIN SELU_")
    if _use_flat(A):
        return _launch_flat(A, A)
    return selu_func(A, out0=A)
