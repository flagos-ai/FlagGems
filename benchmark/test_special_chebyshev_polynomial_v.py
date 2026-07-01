import pytest
import torch

import flag_gems

from . import base


@pytest.mark.special_chebyshev_polynomial_v
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_chebyshev_polynomial_v():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_chebyshev_polynomial_v",
        torch_op=torch.special.chebyshev_polynomial_v,
        # torch reference only supports float32 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
