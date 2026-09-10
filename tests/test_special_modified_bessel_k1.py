import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_k1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# PyTorch reference only supports float32 for special_modified_bessel_k1
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_modified_bessel_k1(shape, dtype):
    # The implementation uses series + asymptotic approximation with ~4% max error
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.01
    )  # Ensure positive values
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.special_modified_bessel_k1(ref_inp.cpu()).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_modified_bessel_k1(inp)

    # Use relaxed tolerance since approximation has ~4% max error
    utils.gems_assert_close(res_out.cpu(), ref_out, dtype, atol=0.05)


@pytest.mark.special_modified_bessel_k1_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# PyTorch reference only supports float32 for special_modified_bessel_k1
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_modified_bessel_k1_out(shape, dtype):
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.01
    )  # Ensure positive values
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_modified_bessel_k1(ref_inp.cpu()).to(dtype)
    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        torch.ops.aten.special_modified_bessel_k1.out(inp, out=out)
    utils.gems_assert_close(out.cpu(), ref_out, dtype, atol=0.05)
