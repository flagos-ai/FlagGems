import pytest
import torch

from . import base


@pytest.mark.special_shifted_chebyshev_polynomial_t
def test_special_shifted_chebyshev_polynomial_t():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_t",
        torch_op=torch.special.shifted_chebyshev_polynomial_t,
        # special.* operators only support float32
        dtypes=[torch.float32],
    )
    bench.run()
