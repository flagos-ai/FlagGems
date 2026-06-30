import pytest
import torch

from . import base


@pytest.mark.special_shifted_chebyshev_polynomial_u
def test_special_shifted_chebyshev_polynomial_u():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_u",
        torch_op=torch.special.shifted_chebyshev_polynomial_u,
        # shifted_chebyshev_polynomial_u_cuda only supports float32
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_shifted_chebyshev_polynomial_u_
def test_special_shifted_chebyshev_polynomial_u_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_shifted_chebyshev_polynomial_u_",
        torch_op=torch.special.shifted_chebyshev_polynomial_u,
        # shifted_chebyshev_polynomial_u_cuda only supports float32
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
