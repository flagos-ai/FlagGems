# ABS_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.abs_ import abs_ as gems_abs_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.abs_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_abs__tensor(shape, dtype):
    ref_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    act_input = ref_input.clone()

    ref_out = torch.ops.aten.abs_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_abs_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_input, ref_input, dtype=dtype)


@pytest.mark.abs_
def test_perf_aten_abs_():
    # Define input generation logic matching the operator arguments
    def abs__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=abs__input_fn,
        op_name="abs_",
        torch_op=torch.ops.aten.abs_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
