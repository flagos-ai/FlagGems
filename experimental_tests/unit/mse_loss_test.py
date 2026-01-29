# MSE_LOSS operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.mse_loss import mse_loss as gems_mse_loss  # noqa: E402
from flag_gems.experimental_ops.mse_loss import (  # noqa: E402
    mse_loss_out as gems_mse_loss_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison aligned with flag_gems.testing.assert_close
        from flag_gems.testing import assert_close as fg_assert_close  # noqa: E402

        kwargs = dict(kwargs)
        reduce_dim = kwargs.pop("reduce_dim", 1)
        equal_nan = kwargs.pop("equal_nan", False)
        fg_assert_close(res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.mse_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_mse_loss_tensor(shape, dtype, reduction):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)
    if reduction in (1, 2) and dtype == torch.bfloat16:
        ref_x = ref_x.float()
        ref_y = ref_y.float()
    ref_out = torch.ops.aten.mse_loss(ref_x, ref_y, reduction)
    if reduction in (1, 2) and dtype == torch.bfloat16:
        ref_out = ref_out.to(dtype)

    with flag_gems.use_gems():
        act_out = gems_mse_loss(x, y, reduction)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.mse_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_mse_loss_out(shape, dtype, reduction):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)

    if reduction in (1, 2) and dtype == torch.bfloat16:
        ref_compute_dtype = torch.float32
    else:
        ref_compute_dtype = dtype

    if reduction == 0:
        ref_out_buf = torch.empty(shape, dtype=ref_compute_dtype, device=ref_x.device)
        act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        ref_out_buf = torch.empty((), dtype=ref_compute_dtype, device=ref_x.device)
        act_out_buf = torch.empty((), dtype=dtype, device=flag_gems.device)

    if reduction in (1, 2) and dtype == torch.bfloat16:
        ref_x = ref_x.float()
        ref_y = ref_y.float()
    ref_out = torch.ops.aten.mse_loss.out(ref_x, ref_y, reduction, out=ref_out_buf)
    if reduction in (1, 2) and dtype == torch.bfloat16:
        ref_out_buf = ref_out_buf.to(dtype)
        ref_out = ref_out_buf

    with flag_gems.use_gems():
        act_out = gems_mse_loss_out(x, y, reduction, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)
