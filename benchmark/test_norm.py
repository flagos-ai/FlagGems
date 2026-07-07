import pytest
import torch

from . import base, consts


@pytest.mark.norm
def test_norm():
    bench = base.UnaryReductionBenchmark(
        op_name="norm",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
