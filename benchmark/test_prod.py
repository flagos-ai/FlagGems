import pytest
import torch

from . import base, consts


@pytest.mark.prod
def test_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
