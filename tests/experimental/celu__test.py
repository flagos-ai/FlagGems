# CELU_ operator test

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
from flag_gems.experimental_ops.celu_ import celu_ as gems_celu_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.celu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_celu__default_alpha(shape, dtype):
    inp_ref = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_act = inp_ref.clone()

    ref_out = torch.ops.aten.celu_(inp_ref)

    with flag_gems.use_gems():
        act_out = gems_celu_(inp_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.celu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_celu__alpha(shape, dtype, alpha):
    inp_ref = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_act = inp_ref.clone()

    ref_out = torch.ops.aten.celu_(inp_ref, alpha)

    with flag_gems.use_gems():
        act_out = gems_celu_(inp_act, alpha)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.celu_
def test_perf_aten_celu_():
    # Define input generation logic matching the operator arguments
    def celu__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

        # You can also yield kwargs if needed
        # yield inp, {'alpha': alpha_value}

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=celu__input_fn,
        op_name="celu_",
        torch_op=torch.ops.aten.celu_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
