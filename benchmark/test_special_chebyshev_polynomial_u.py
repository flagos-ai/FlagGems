import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    # torch.special.chebyshev_polynomial_u only supports float32 on CPU/CUDA
    x = torch.randn(shape, dtype=torch.float32, device=device)
    # Use a fixed n value for benchmarking
    n = 3
    yield x, n


# torch.special.chebyshev_polynomial_u only supports float32
@pytest.mark.special_chebyshev_polynomial_u
def test_special_chebyshev_polynomial_u():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="special_chebyshev_polynomial_u",
        # torch.special.chebyshev_polynomial_u only supports float32 on CPU/CUDA
        dtypes=[torch.float32],
        torch_op=torch.special.chebyshev_polynomial_u,
    )
    bench.run()
