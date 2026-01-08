# ZERO_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.zero_ import zero_ as gems_zero_  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.zero_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_zero__tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = x.clone()
    act_input = x.clone()

    ref_out = torch.ops.aten.zero_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_zero_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.zero_
def test_perf_aten_zero_():
    # Define input generation logic matching the operator arguments
    def zero__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=zero__input_fn,
        op_name="zero_",
        torch_op=torch.ops.aten.zero_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
