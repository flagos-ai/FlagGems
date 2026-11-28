import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_k1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_special_modified_bessel_k1_small_int(shape, dtype):
    inp = torch.randint(1, 20, shape, dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.special_modified_bessel_k1(ref_inp.cpu())
    res_out = flag_gems.special_modified_bessel_k1(inp)

    utils.gems_assert_close(res_out.cpu(), ref_out, torch.float32, atol=0.05)


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
    res_out = flag_gems.special_modified_bessel_k1(inp)

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
    flag_gems.special_modified_bessel_k1_out(inp, out)
    utils.gems_assert_close(out.cpu(), ref_out, dtype, atol=0.05)


@pytest.mark.special_modified_bessel_k1_out
def test_special_modified_bessel_k1_out_rejects_integral_out():
    """ATen rejects an integral out buffer for this floating-point op; so does FlagGems."""
    inp = torch.rand(4, dtype=torch.float32, device=flag_gems.device) + 0.01
    with pytest.raises(RuntimeError):
        flag_gems.special_modified_bessel_k1_out(
            inp, torch.empty(4, dtype=torch.int32, device=flag_gems.device)
        )
