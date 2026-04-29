import pytest
import torch

from . import base, consts


@pytest.mark.max
def test_max():
    bench = base.UnaryReductionBenchmark(
        op_name="max", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
