import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_y1
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_special_bessel_y1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    # Compute reference in float32 since torch.special.bessel_y1 doesn't support float16
    ref_inp_f32 = ref_inp.to(torch.float32)
    ref_out = torch.special.bessel_y1(ref_inp_f32).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_y1(inp)

    # Bessel Y1 produces NaN for negative inputs and -inf for large inputs;
    # handle comparison by checking special values first
    res_cpu = res_out.to("cpu")
    ref_cpu = ref_out.to("cpu")
    res_inf = torch.isinf(res_cpu)
    ref_inf = torch.isinf(ref_cpu)
    res_nan = torch.isnan(res_cpu)
    ref_nan = torch.isnan(ref_cpu)
    # Check that inf/-inf and NaN positions match
    assert (res_inf == ref_inf).all(), "Inf positions don't match"
    assert (res_nan == ref_nan).all(), "NaN positions don't match"
    # For finite values, check closeness
    valid_mask = ~res_inf & ~ref_inf & ~res_nan & ~ref_nan
    if valid_mask.sum() > 0:
        gems_assert_close(res_cpu[valid_mask], ref_cpu[valid_mask], dtype)


@pytest.mark.inplace
@pytest.mark.special_bessel_y1_
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_special_bessel_y1_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    # Compute reference in float32 since torch.special.bessel_y1 doesn't support float16
    ref_inp_f32 = ref_inp.to(torch.float32)
    ref_out = ref_inp_f32.special_bessel_y1_().to(dtype)
    with flag_gems.use_gems():
        res_out = inp.special_bessel_y1_()

    # Bessel Y1 produces NaN for negative inputs and -inf for large inputs;
    # handle comparison by checking special values first
    res_cpu = res_out.to("cpu")
    ref_cpu = ref_out.to("cpu")
    res_inf = torch.isinf(res_cpu)
    ref_inf = torch.isinf(ref_cpu)
    res_nan = torch.isnan(res_cpu)
    ref_nan = torch.isnan(ref_cpu)
    # Check that inf/-inf and NaN positions match
    assert (res_inf == ref_inf).all(), "Inf positions don't match"
    assert (res_nan == ref_nan).all(), "NaN positions don't match"
    # For finite values, check closeness
    valid_mask = ~res_inf & ~ref_inf & ~res_nan & ~ref_nan
    if valid_mask.sum() > 0:
        gems_assert_close(res_cpu[valid_mask], ref_cpu[valid_mask], dtype)
