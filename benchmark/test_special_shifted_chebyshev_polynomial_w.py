import pytest
import torch

from . import base


@pytest.mark.special_shifted_chebyshev_polynomial_w
def test_special_shifted_chebyshev_polynomial_w():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_w",
        torch_op=torch.special.shifted_chebyshev_polynomial_w,
        # PyTorch reference only supports float32 for this operator
        dtypes=[torch.float32],
    )
    bench.run()
