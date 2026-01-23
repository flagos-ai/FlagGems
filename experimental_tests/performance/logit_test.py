# LOGIT operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison aligned with flag_gems.testing.assert_close
        from flag_gems.testing import assert_close as fg_assert_close  # noqa: E402

        kwargs = dict(kwargs)
        reduce_dim = kwargs.pop("reduce_dim", 1)
        equal_nan = kwargs.pop("equal_nan", False)
        fg_assert_close(res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim)


def to_reference(inp):
    """Move to CPU when TO_CPU is set, keep dtype/device otherwise."""
    if inp is None:
        return None
    return inp.to("cpu") if TO_CPU else inp.clone()


@pytest.mark.logit
def test_perf_aten_logit():
    # Define input generation logic matching the operator arguments
    def logit_input_fn(shape, dtype, device):
        base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
        input_tensor = base.to(dtype)
        yield input_tensor,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=logit_input_fn,
        op_name="logit",
        torch_op=torch.ops.aten.logit,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
