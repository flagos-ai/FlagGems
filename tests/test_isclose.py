import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.isclose
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("zero_tol", [False, True])
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_isclose(shape, dtype, zero_tol, equal_nan, gen_nan):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.int32:
        pytest.skip("Issue #2822: Skiping bool isclose test on tsingmicro platform")

    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = (
        torch.rand(1, dtype=torch.float32, device=flag_gems.device).item() * 0.0001
        if not zero_tol
        else 0
    )

    if dtype in utils.ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device=flag_gems.device,
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
        atol = (
            torch.finfo(dtype).tiny
            * torch.randint(0, 4, (1,), device=flag_gems.device).item()
            if not zero_tol
            else 0
        )
    else:
        inp1 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        if dtype in [torch.int64]:
            inp1.view(-1)[0] = 2**63 - 1
            inp2.view(-1)[0] = -(2**63)
            if inp1.numel() > 2 and inp2.numel() > 2:
                inp1.view(-1)[1] = 2**60 + 2**20
                inp2.view(-1)[1] = 2**60
                inp1.view(-1)[2] = 2**60 + 1
                inp2.view(-1)[2] = 2**60
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        elif dtype in [torch.int32]:
            inp1.view(-1)[0] = 2**31 - 1
            inp2.view(-1)[0] = -(2**31)
            if inp1.numel() > 2 and inp2.numel() > 2:
                inp1.view(-1)[1] = 2**30 + 2**5
                inp2.view(-1)[1] = 2**30
                inp1.view(-1)[2] = 2**30 + 1
                inp2.view(-1)[2] = 2**30
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        else:
            atol = (
                (
                    torch.finfo(torch.float16).eps
                    * torch.randint(0, 10, (1,), device=flag_gems.device).item()
                )
                if not zero_tol
                else 0
            )

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)
    with flag_gems.use_gems():
        res_out = torch.isclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)

    ref_out = torch.isclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    ref_flat = ref_out.view(-1)
    res_flat = res_out.view(-1)
    if inp1.numel() > 2 and dtype in [torch.int64, torch.int32]:
        assert (
            res_flat[1] == ref_flat[1] and res_flat[2] == ref_flat[2]
        ), "res vs ref: {} vs {}, {} vs {}".format(
            res_flat[1], ref_flat[1], res_flat[2], ref_flat[2]
        )

    utils.gems_assert_equal(res_out, ref_out)
