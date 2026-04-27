import pytest
import torch

from . import base, consts


@pytest.mark.any
def test_any():
    bench = base.UnaryReductionBenchmark(
        op_name="any", torch_op=torch.any, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
