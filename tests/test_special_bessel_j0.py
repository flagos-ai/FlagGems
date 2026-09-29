import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# torch.special.bessel_j0 supports float32 and float64. On the Kunlunxin XPU
# backend a float64 request is silently downgraded to float32, so it can never
# match a true-fp64 reference; skip fp64 on that backend only. Every other
# vendor keeps running float64 exactly as before.
_SKIP_FP64_ON_KUNLUNXIN = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "kunlunxin",
    reason="Kunlunxin XPU has no real float64 (silently downgraded to float32)",
)
FLOAT_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64_ON_KUNLUNXIN),
]

# Pointwise shapes covering small, medium, and batched 2D tensors
POINTWISE_SHAPES = [(128,), (512, 256), (2, 128, 128)]


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_bessel_j0(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.special.bessel_j0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_j0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
