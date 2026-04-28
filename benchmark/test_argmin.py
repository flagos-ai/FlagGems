import pytest
import torch

from . import base, consts


@pytest.mark.argmin
def test_argmin():
    bench = base.UnaryReductionBenchmark(
        op_name="argmin", torch_op=torch.argmin, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
