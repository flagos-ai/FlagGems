import pytest
import torch

from . import base, consts


@pytest.mark.std
def test_std():
    bench = base.UnaryReductionBenchmark(
        op_name="std", torch_op=torch.std, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
