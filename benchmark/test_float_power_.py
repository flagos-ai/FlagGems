import pytest
import torch

from . import base


@pytest.mark.float_power_
def test_float_power_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="float_power_",
        torch_op=lambda a, b: torch.float_power(
            a.to(torch.float64), b.to(torch.float64)
        ).to(a.dtype),
        # float_power_ reference: PyTorch's implementation is broken for non-float64,
        # so we use a custom reference that computes the expected result
        # (compute in float64 then convert back to input dtype)
        # Skip bfloat16 as it causes issues with .to(torch.float64)
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()
