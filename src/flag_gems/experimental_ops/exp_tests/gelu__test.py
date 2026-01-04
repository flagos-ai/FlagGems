# GELU_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.gelu_ import gelu_ as gems_gelu_

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


@pytest.mark.gelu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu__tensor(shape, dtype, approximate):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = base.clone()
    ref_out = torch.ops.aten.gelu_(ref_input, approximate=approximate)

    act_input = base.clone()
    with flag_gems.use_gems():
        act_out = gems_gelu_(act_input, approximate=approximate)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.gelu_
def test_perf_aten_gelu_():
    # Define input generation logic matching the operator arguments
    def gelu__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=gelu__input_fn,
        op_name="gelu_",
        torch_op=torch.ops.aten.gelu_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
