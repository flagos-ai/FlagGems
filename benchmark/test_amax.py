import pytest
import torch

from . import base, consts


@pytest.mark.test_amax
def test_amax():
    bench = base.UnaryReductionBenchmark(
        op_name="amax", torch_op=torch.amax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
