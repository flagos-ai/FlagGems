"""
logaddexp operator — FlagGems submission (赛道一 · 初级算子)

ATen schemas:
    logaddexp(Tensor self, Tensor other) -> Tensor
    logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

Platform: Iluvatar BI-V150 (天数智芯) / corex-4.4.0 / Triton 3.1.0
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _logaddexp_kernel(x, y):
    """Numerically stable logaddexp: m + log(exp(x-m) + exp(y-m))"""
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    m = tl.maximum(x_fp32, y_fp32)

    diff_x = tl.where(x_fp32 == m, 0.0, x_fp32 - m)
    diff_y = tl.where(y_fp32 == m, 0.0, y_fp32 - m)

    result = m + tl.log(tl.exp(diff_x) + tl.exp(diff_y))
    return result.to(x.dtype)


def logaddexp(x, y):
    logger.debug("GEMS LOGADDEXP")
    x_broadcast, y_broadcast = torch.broadcast_tensors(x, y)
    return _logaddexp_kernel(x_broadcast, y_broadcast)


def logaddexp_out(x, y, *, out):
    logger.debug("GEMS LOGADDEXP_OUT")
    x_broadcast, y_broadcast = torch.broadcast_tensors(x, y)
    if tuple(out.shape) != tuple(x_broadcast.shape):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match "
            f"broadcasted shape {tuple(x_broadcast.shape)}"
        )
    _logaddexp_kernel(x_broadcast, y_broadcast, out0=out)
    return out


# ---------------------------------------------------------------------------
# Quick self-test (保持不变)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flag_gems

    print("=== logaddexp self-test ===\n")
    device = "cuda"

    # 1. Basic correctness
    x = torch.tensor([0.0, 1.0, 2.0], device=device)
    y = torch.tensor([0.0, 1.0, 2.0], device=device)
    ref = torch.logaddexp(x, y)
    with flag_gems.use_gems():
        res = torch.logaddexp(x, y)
    print(f"logaddexp([0,1,2], [0,1,2]) = {res.tolist()}")
    assert torch.allclose(res, ref, atol=1e-4, rtol=1e-4), f"FAIL: {res} vs {ref}"
    print("  ✅ forward 正确")

    # 2. out=
    buf = torch.empty(3, device=device)
    with flag_gems.use_gems():
        ret = torch.logaddexp(x, y, out=buf)
    assert ret is buf, "logaddexp.out must return out"
    print(f"logaddexp([0,1,2], [0,1,2], out) = {buf.tolist()}  ✅ out= 正确")

    # 3. Different inputs
    x2 = torch.tensor([100.0, 0.0, -100.0], device=device)
    y2 = torch.tensor([101.0, 0.0, -90.0], device=device)
    ref2 = torch.logaddexp(x2, y2)
    with flag_gems.use_gems():
        res2 = torch.logaddexp(x2, y2)
    assert torch.allclose(res2, ref2, atol=1e-3, rtol=1e-3), f"FAIL: {res2} vs {ref2}"
    print("  ✅ different inputs 正确")

    # 4. Special values: inf handling (✅ 已修复)
    x3 = torch.tensor([float("inf"), 0.0, -float("inf")], device=device)
    y3 = torch.tensor([0.0, float("inf"), -float("inf")], device=device)
    ref3 = torch.logaddexp(x3, y3)
    res3 = logaddexp(x3, y3)
    assert torch.allclose(
        res3, ref3, equal_nan=True
    ), f"Special values FAIL: {res3} vs {ref3}"
    print("  ✅ special values (inf/-inf) 正确")

    # 5. Large value test
    x4 = torch.tensor([1000.0, -1000.0], device=device)
    y4 = torch.tensor([1000.0, 1000.0], device=device)
    ref4 = torch.logaddexp(x4, y4)
    res4 = logaddexp(x4, y4)
    assert torch.allclose(
        res4, ref4, atol=1e-3, rtol=1e-3
    ), f"Large value FAIL: {res4} vs {ref4}"
    print("  ✅ numerical stability (large values) 正确")

    # 6. Broadcast test
    x5 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    y5 = torch.tensor([1.0], device=device)
    ref5 = torch.logaddexp(x5, y5)
    res5 = logaddexp(x5, y5)
    assert torch.allclose(
        res5, ref5, atol=1e-4, rtol=1e-4
    ), f"Broadcast FAIL: {res5} vs {ref5}"
    print("  ✅ broadcast 正确")

    print("\n✅ 全部自测通过！")
