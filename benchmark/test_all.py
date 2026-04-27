import pytest
import torch

from . import base, consts


@pytest.mark.all
def test_all():
    bench = base.UnaryReductionBenchmark(
        op_name="all", torch_op=torch.all, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
