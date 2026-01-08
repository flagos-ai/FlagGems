# ATANH_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.atanh_ import atanh_ as gems_atanh_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("contig", [True, False])
def test_atanh__tensor(shape, dtype, contig):
    base = (torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8) - 0.9
    if contig:
        ref_input = base.clone()
        act_input = base.clone()
    else:
        base_ref = base.clone()
        base_act = base.clone()
        ref_input = base_ref.transpose(0, 1)
        act_input = base_act.transpose(0, 1)

    ref_out = torch.ops.aten.atanh_(ref_input)
    with flag_gems.use_gems():
        act_out = gems_atanh_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.atanh_
def test_perf_aten_atanh_():
    # Define input generation logic matching the operator arguments
    def atanh__input_fn(shape, dtype, device):
        inp = (torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8) - 0.9
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=atanh__input_fn,
        op_name="atanh_",
        torch_op=torch.ops.aten.atanh_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
