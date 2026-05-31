import pytest
import torch

from . import base


@pytest.mark.special_scaled_modified_bessel_k0
def test_special_scaled_modified_bessel_k0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_scaled_modified_bessel_k0",
        torch_op=torch.special.scaled_modified_bessel_k0,
        # Only float32 since PyTorch doesn't support half/bfloat16 for special operators
        dtypes=[torch.float32],
    )
    bench.run()
