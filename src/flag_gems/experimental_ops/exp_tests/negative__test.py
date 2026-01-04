# NEGATIVE_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.negative_ import negative_ as gems_negative_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.negative_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_negative__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.negative_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_negative_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.negative_
def test_perf_aten_negative_():
    # Define input generation logic matching the operator arguments
    def negative__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=negative__input_fn,
        op_name="negative_",
        torch_op=torch.ops.aten.negative_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
