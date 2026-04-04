"""
log10 operator — FlagGems submission (赛道一 · 初级算子)

ATen schemas:
    log10(Tensor self) -> Tensor
    log10_(Tensor(a!) self) -> Tensor(a!)
    log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

Platform: Iluvatar BI-V150 (天数智芯) / corex-4.4.0 / Triton 3.1.0
"""

import logging

import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# log10(x) = ln(x) * log10(e)
# log10(e) = 1/ln(10) = 0.4342944819032518
# 预计算为编译期常数，避免运行时除法，同时保证 fp32/fp16/bf16 精度
# Triton 3.x 要求在 kernel 中使用的全局变量必须声明为 constexpr
_LOG10E = tl.constexpr(0.4342944819032518)


# promotion_methods 对齐 FlagGems 官方 abs.py 实测写法：
#   "COMPLEX_TO_FLOAT" — 输入若为复数则取模转实，普通浮点直接透传，
#   比 "DEFAULT" 在天数智芯 corex/Triton 上更稳定。
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def _log10_kernel(x):
    """log10(x) = ln(x) * log10(e)，利用 Triton 内置 tl.log 实现。

    tl.log 不支持 fp16/bf16，需要先转换为 fp32，计算后再转回原始类型。
    """
    # 类型转换：fp16/bf16 -> fp32 -> 计算 -> fp32 -> 原始类型
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.log(x_fp32) * _LOG10E
    return y_fp32.to(x.dtype)


def log10(self):
    """
    Compute element-wise base-10 logarithm.

    Mirrors torch.log10 semantics:
      x > 0  →  log10(x)
      x = 0  →  -inf
      x < 0  →  nan
      nan    →  nan
      +inf   →  +inf

    Args:
        self (Tensor): Input tensor (floating-point).

    Returns:
        Tensor: Same shape and dtype as input.
    """
    logger.debug("GEMS LOG10")
    return _log10_kernel(self)


def log10_(self):
    """
    In-place log10. Modifies ``self`` and returns it.

    Args:
        self (Tensor): Input/output tensor.

    Returns:
        Tensor: ``self`` (same object).
    """
    logger.debug("GEMS LOG10_")
    _log10_kernel(self, out0=self)
    return self


def log10_out(self, *, out):
    """
    Compute log10 and write into pre-allocated ``out``.

    Args:
        self (Tensor): Input tensor.
        out  (Tensor): Output buffer (same shape as self).

    Returns:
        Tensor: ``out`` (same object).
    """
    logger.debug("GEMS LOG10_OUT")
    _log10_kernel(self, out0=out)
    return out


# ---------------------------------------------------------------------------
# Quick self-test (python log10_pointwise_submit.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flag_gems
    import torch

    print("=== log10 self-test ===\n")

    device = "cuda"

    # 1. 基本正确性
    x = torch.tensor([1.0, 10.0, 100.0, 1000.0], device=device)
    with flag_gems.use_gems():
        y = torch.log10(x)
    print(f"log10([1,10,100,1000]) = {y.tolist()}")
    assert y.tolist() == [0.0, 1.0, 2.0, 3.0], f"FAIL: {y}"
    print("  ✅ forward 正确")

    # 2. in-place
    x2 = torch.tensor([1.0, 10.0, 100.0], device=device)
    with flag_gems.use_gems():
        ret = torch.log10_(x2)
    assert ret is x2, "log10_ 必须返回 self"
    print(f"log10_([1,10,100])     = {x2.tolist()}  ✅ in-place 正确")

    # 3. out=
    x3 = torch.tensor([10.0, 1000.0], device=device)
    buf = torch.empty(2, device=device)
    with flag_gems.use_gems():
        ret2 = torch.log10(x3, out=buf)
    assert ret2 is buf, "log10.out 必须返回 out"
    print(f"log10([10,1000], out)  = {buf.tolist()}  ✅ out= 正确")

    # 4. 特殊值：nan / -inf / NaN 语义必须与 torch 一致
    specials = torch.tensor([-1.0, 0.0, float("inf"), float("nan")], device=device)
    ref = torch.log10(specials)
    res = log10(specials)
    assert torch.allclose(
        res, ref, equal_nan=True
    ), f"特殊值语义不匹配!\n  ref={ref}\n  res={res}"
    print("特殊值测试              passed  ✅ (nan/−inf/+inf 与 torch 一致)")

    # 5. 大尺寸精度（float32，atol ≤ 1.3e-6）
    x_big = torch.rand(4096, 4096, device=device).abs() + 1e-3
    ref_big = torch.log10(x_big.cpu()).to(device)  # CPU 作为参考
    res_big = log10(x_big)
    torch.testing.assert_close(res_big, ref_big, atol=1.3e-6, rtol=1e-4)
    print("大尺寸精度 (4096×4096)  passed  ✅")

    print("\n✅ 全部自测通过！")
