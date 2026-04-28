import pytest
import torch

from . import base, consts


@pytest.mark.mean
def test_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="mean", torch_op=torch.mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
