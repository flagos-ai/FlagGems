# MASKED_SELECT operator test

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


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.masked_select import (  # noqa: E402
    masked_select as gems_masked_select,
)
from flag_gems.experimental_ops.masked_select import (  # noqa: E402
    masked_select_out as gems_masked_select_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.masked_select
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_select_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) > 0.5

    ref_x = x.clone()
    ref_mask = mask.clone()

    ref_out = torch.ops.aten.masked_select(ref_x, ref_mask)

    with flag_gems.use_gems():
        act_out = gems_masked_select(x, mask)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_select
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_select_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) > 0.5

    ref_x = x.clone()
    ref_mask = mask.clone()

    ref_n = int(ref_mask.sum().item())
    ref_out_buf = torch.empty((ref_n,), dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.masked_select.out(ref_x, ref_mask, out=ref_out_buf)

    with flag_gems.use_gems():
        act_n = int(mask.sum().item())
        act_out_buf = torch.empty((act_n,), dtype=dtype, device=flag_gems.device)
        act_out = gems_masked_select_out(x, mask, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_select
def test_perf_aten_masked_select():
    # Define input generation logic matching the operator arguments
    def masked_select_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        mask = torch.rand(shape, device=flag_gems.device) > 0.5
        yield x, mask

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=masked_select_input_fn,
        op_name="masked_select",
        torch_op=torch.ops.aten.masked_select,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
