import pytest
import torch

from . import base, consts


@pytest.mark.min
def test_min():
    bench = base.UnaryReductionBenchmark(
        op_name="min", torch_op=torch.min, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
