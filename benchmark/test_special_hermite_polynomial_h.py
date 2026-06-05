import pytest
import torch

from . import base


@pytest.mark.special_hermite_polynomial_h
def test_special_hermite_polynomial_h():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_hermite_polynomial_h",
        torch_op=torch.special.hermite_polynomial_h,
        # special.* ops only support float32
        dtypes=[torch.float32],
    )
    bench.run()
