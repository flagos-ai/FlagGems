import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# special.bessel_y1 only supports float32/float64; float16/bf16 raise RuntimeError.
# On the Kunlunxin XPU backend a float64 request is silently downgraded to float32,
# so it can never match a true-fp64 reference; skip fp64 on that backend only.
# Every other vendor keeps running float64 exactly as before.
_SKIP_FP64_ON_KUNLUNXIN = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "kunlunxin",
    reason="Kunlunxin XPU has no real float64 (silently downgraded to float32)",
)
_FLOAT_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64_ON_KUNLUNXIN),
]


@pytest.mark.special_bessel_y1
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_bessel_y1(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("Skipping fp64 test on platform without fp64 support")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.bessel_y1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
