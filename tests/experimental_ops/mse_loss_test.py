# MSE_LOSS operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.mse_loss import mse_loss as gems_mse_loss  # noqa: E402
from flag_gems.experimental_ops.mse_loss import (  # noqa: E402
    mse_loss_out as gems_mse_loss_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
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
    ref_out = torch.ops.aten.mse_loss(ref_x, ref_y, reduction)

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

    if reduction == 0:
        ref_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        ref_out_buf = torch.empty((), dtype=dtype, device=flag_gems.device)
        act_out_buf = torch.empty((), dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_y = to_reference(y)
    ref_out = torch.ops.aten.mse_loss.out(ref_x, ref_y, reduction, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_mse_loss_out(x, y, reduction, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.mse_loss
def test_perf_aten_mse_loss():
    # Define input generation logic matching the operator arguments
    def mse_loss_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=mse_loss_input_fn,
        op_name="mse_loss",
        torch_op=torch.ops.aten.mse_loss,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
