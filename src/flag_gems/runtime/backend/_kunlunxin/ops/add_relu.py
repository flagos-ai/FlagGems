# Kunlunxin (XPU) override of `_add_relu` / `_add_relu_` (relu(x + y)).
#
# Why this file exists: `_add_relu.Tensor` was NOT overridden by the vendor
# backend, so `torch._add_relu(a, b)` (the functional test path) fell back to
# the generic bare `flag_gems.utils.pointwise_dynamic` codegen
# (`ops/_add_relu.py`), which on XPU lowers to discrete (non-unit-stride)
# memory access -> catastrophic latency: measured on XPU 5 (2026-09-04)
# 8.5 ms @ 2.56M elems, 53 ms @ 16M, 890 ms @ 256M, 2.28 s @ 655M (fp16/bf16),
# vs ~1.3/2.6 ms for native torch `relu(add(a,b))` (~1000x). The sibling
# kunlunxin `add`/`relu` (same framework `_kunlunxin.utils.pointwise_dynamic`)
# are memory-bound (10-12 us small, ~2 TB/s large), so this override restores
# the fast path for the op.
#
# Semantics: `_add_relu(a, b) == max(0, a + b)` elementwise. `tl.maximum`
# lowers to `maxnum` (NaN -> the other operand), which matches the native ATen
# |add_relu| kernel exactly (CPU ref: `torch._add_relu([nan, ...], [0, ...])`
# -> 0.0, i.e. max(0, NaN) == 0 -- NOTE: this differs from
# `torch.relu(x)` (NaN -> NaN) and `torch.clamp(x, min=0)` (NaN -> NaN)).
# `alpha` is honored (`relu(a + alpha*b)`), unlike the previous generic impl
# which silently ignored it.
import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def add_relu_func(x, y, alpha):
    # relu(x + alpha*y) = max(0, x + alpha*y); single maximum instruction.
    return tl.maximum(x + y * alpha, 0)


def _add_relu(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADD_RELU")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_relu_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_relu_func(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_relu_func(A, B, alpha)
    else:
        return torch.tensor(max(0, A + B * alpha))


def _add_relu_(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADD_RELU_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        add_relu_func(A, B, alpha, out0=A)
        return A
    else:
        raise ValueError("Unreachable.")