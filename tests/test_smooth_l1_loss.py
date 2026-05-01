import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_smooth_l1_loss(shape, dtype, reduction, beta):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        reduce_dim=shape[dim],
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_smooth_l1_loss_aten_out_mean(dtype):
    shape = (2, 8)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    rout = torch.empty((), dtype=dtype, device="cpu")
    torch.ops.aten.smooth_l1_loss.out(
        ref_inp, ref_target, 1, 1.0, out=rout
    )

    out = torch.empty((), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss.out(inp, target, 1, 1.0, out=out)

    utils.gems_assert_close(out, rout.to(flag_gems.device), dtype, equal_nan=True)
