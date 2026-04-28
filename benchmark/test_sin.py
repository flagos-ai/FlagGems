import pytest
import torch

from . import base, consts


@pytest.mark.sin
def test_sin():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sin", torch_op=torch.sin, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sin_
def test_sin_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sin_", torch_op=torch.sin_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
