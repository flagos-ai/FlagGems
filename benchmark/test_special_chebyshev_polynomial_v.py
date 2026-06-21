import pytest
import torch

from . import base


@pytest.mark.special_chebyshev_polynomial_v
def test_special_chebyshev_polynomial_v():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_chebyshev_polynomial_v",
        torch_op=torch.special.chebyshev_polynomial_v,
        # torch reference only supports float32 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
