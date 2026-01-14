# LOG2_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.log2_ import log2_ as gems_log2_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp):
    """Convert tensor to reference device (CPU if TO_CPU is True)."""
    if TO_CPU:
        return inp.to("cpu")
    return inp.clone()


@pytest.mark.log2_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_log2__tensor(shape, dtype, noncontig):
    base = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    eps = torch.tensor(0.1, dtype=dtype, device=flag_gems.device)
    if noncontig:
        inp = (base + eps).transpose(0, 1)
    else:
        inp = base + eps
    ref_input = to_reference(inp)
    act_input = inp.clone()

    ref_out = torch.ops.aten.log2_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_log2_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.log2_
def test_perf_aten_log2_():
    # Define input generation logic matching the operator arguments
    def log2__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp = (
            torch.randn(shape, dtype=dtype, device=flag_gems.device) + 0.1
        )  # Adding 0.1 to avoid log2(0)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=log2__input_fn,
        op_name="log2_",
        torch_op=torch.ops.aten.log2_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
