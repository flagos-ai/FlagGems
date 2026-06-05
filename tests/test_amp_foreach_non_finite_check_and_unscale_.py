import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# worktree explicitly excludes bfloat16 via PRIMARY_FLOAT_DTYPES
PRIMARY_FLOAT_DTYPES = [torch.float16, torch.float32]


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale_(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with normal tensors."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.randn(16, 32, device=flag_gems.device, dtype=dtype),
        torch.randn(8, 16, device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    with flag_gems.use_gems():
        getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
            res_tensors, res_found_inf, inv_scale
        )

    # Compare mutated inputs (in-place operation)
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(inp, ref_inp, dtype)

    # Compare found_inf (also mutated in-place)
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__inf(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with inf values."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.tensor(
            [1.0, 2.0, float("inf"), 4.0], device=flag_gems.device, dtype=dtype
        ),
        torch.tensor([5.0, 6.0, 7.0], device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    with flag_gems.use_gems():
        getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
            res_tensors, res_found_inf, inv_scale
        )

    # Compare mutated inputs (in-place operation)
    # Note: inf values remain unchanged, only finite values are scaled
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(inp, ref_inp, dtype)

    # Compare found_inf (also mutated in-place) - should be 1.0 when inf is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__nan(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with nan values."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.tensor(
            [1.0, 2.0, float("nan"), 4.0], device=flag_gems.device, dtype=dtype
        ),
        torch.tensor([5.0, 6.0, 7.0], device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    with flag_gems.use_gems():
        getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
            res_tensors, res_found_inf, inv_scale
        )

    # Compare mutated inputs with equal_nan=True since tensors contain NaN
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)

    # Compare found_inf - should be 1.0 when nan is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)
