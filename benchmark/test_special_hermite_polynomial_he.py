import pytest
import torch

from . import base


@pytest.mark.special_hermite_polynomial_he
def test_special_hermite_polynomial_he():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_hermite_polynomial_he",
        torch_op=torch.special.hermite_polynomial_he,
        # CUDA does not support half/bfloat16 for this special function
        dtypes=[torch.float32],
    )
    bench.run()
