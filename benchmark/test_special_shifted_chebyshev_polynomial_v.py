import pytest
import torch

from . import base


@pytest.mark.special_shifted_chebyshev_polynomial_v
def test_special_shifted_chebyshev_polynomial_v():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_v",
        torch_op=torch.special.shifted_chebyshev_polynomial_v,
        # special.* operators only support float32
        dtypes=[torch.float32],
    )
    bench.run()
