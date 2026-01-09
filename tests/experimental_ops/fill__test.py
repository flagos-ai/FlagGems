# FILL_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.fill_ import fill__Scalar, fill__Tensor  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


def to_reference(inp):
    """Convert tensor to reference device (CPU if TO_CPU is True)."""
    if TO_CPU:
        return inp.to("cpu")
    return inp.clone()


@pytest.mark.fill_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 1.25, -3.5])
def test_fill__scalar(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_ref = to_reference(x)
    x_act = x.clone()

    ref_out = torch.ops.aten.fill_(x_ref, value)

    with flag_gems.use_gems():
        act_out = fill__Scalar(x_act, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fill_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 2.0, -1.5])
def test_fill__tensor(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_ref = to_reference(x)
    x_act = x.clone()

    val = torch.tensor(value, dtype=dtype, device=flag_gems.device)
    val_ref = to_reference(val)
    val_act = val.clone()

    ref_out = torch.ops.aten.fill_(x_ref, val_ref)

    with flag_gems.use_gems():
        act_out = fill__Tensor(x_act, val_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fill_
def test_perf_aten_fill_():
    # Define input generation logic matching the operator arguments
    def fill__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        value = 1.0  # Scalar value to fill
        yield inp, value

    # Initialize benchmark - using fill__Scalar for performance test
    def fill__Scalar_wrapper(self, value):
        return fill__Scalar(self, value)

    bench = GenericBenchmark(
        input_fn=fill__input_fn,
        op_name="fill_",
        torch_op=torch.ops.aten.fill_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    # Replace the op with fill__Scalar
    bench.op = fill__Scalar_wrapper

    return bench.run()
