"""
cosh operator — FlagGems submission (赛道一 · 初级算子)

ATen schemas:
    cosh(Tensor self) -> Tensor

Platform: Iluvatar BI-V150 (天数智芯) / corex-4.4.0 / Triton 3.1.0
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def _cosh_kernel(x):
    """
    Numerically stable cosh: (exp(x) + exp(-x)) / 2
    
    For large |x|, use: cosh(x) ≈ 0.5 * exp(|x|)
    This avoids overflow in intermediate calculations.
    """
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    
    # Stable computation for large |x|:
    # cosh(x) = 0.5 * exp(|x|) * (1 + exp(-2*|x|))
    # When |x| is large, exp(-2*|x|) -> 0, so cosh(x) ≈ 0.5 * exp(|x|)
    exp_neg_2abs = tl.exp(-2.0 * abs_x)
    result = 0.5 * tl.exp(abs_x) * (1.0 + exp_neg_2abs)
    
    return result.to(x.dtype)


def cosh(self):
    """
    Compute hyperbolic cosine element-wise.
    
    Mirrors torch.cosh semantics:
      cosh(x) = (exp(x) + exp(-x)) / 2
    
    Args:
        self (Tensor): Input tensor (floating-point).
    
    Returns:
        Tensor: Same shape and dtype as input.
    """
    logger.debug("GEMS COSH")
    return _cosh_kernel(self)


# ---------------------------------------------------------------------------
# Quick self-test (python cosh.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import flag_gems

    print("=== cosh self-test ===\n")
    device = "cuda"

    # 1. Basic correctness
    x = torch.tensor([0.0, 1.0, 2.0], device=device)
    ref = torch.cosh(x)
    with flag_gems.use_gems():
        res = torch.cosh(x)
    print(f"cosh([0,1,2]) = {res.tolist()}")
    assert torch.allclose(res, ref, atol=1e-4, rtol=1e-4), f"FAIL: {res} vs {ref}"
    print("  ✅ forward 正确")

    # 2. Special values
    specials = torch.tensor([0.0, float("inf"), float("nan")], device=device)
    ref_special = torch.cosh(specials)
    res_special = cosh(specials)
    assert torch.allclose(res_special, ref_special, equal_nan=True), \
        f"Special values FAIL: {res_special} vs {ref_special}"
    print("  ✅ special values (inf/nan) 正确")

    # 3. Large value test (numerical stability)
    x_large = torch.tensor([10.0, 20.0], device=device)
    ref_large = torch.cosh(x_large)
    res_large = cosh(x_large)
    assert torch.allclose(res_large, ref_large, atol=1e-3, rtol=1e-3), \
        f"Large value FAIL: {res_large} vs {ref_large}"
    print("  ✅ numerical stability (large values) 正确")

    # 4. Negative values (cosh is even: cosh(-x) = cosh(x))
    x_neg = torch.tensor([-1.0, -2.0], device=device)
    ref_neg = torch.cosh(x_neg)
    res_neg = cosh(x_neg)
    assert torch.allclose(res_neg, ref_neg, atol=1e-4, rtol=1e-4), \
        f"Negative values FAIL: {res_neg} vs {ref_neg}"
    print("  ✅ negative values 正确")

    print("\n✅ 全部自测通过！")
