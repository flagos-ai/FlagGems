import pytest
import torch

from . import base, consts


@pytest.mark.gcd
def test_gcd():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
