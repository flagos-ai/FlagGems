import pytest
import torch

from . import base, consts


@pytest.mark.abs
def test_abs():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs", torch_op=torch.abs, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.abs_
def test_abs_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs_", torch_op=torch.abs_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
