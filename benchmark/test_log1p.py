import pytest
import torch

from . import base, consts


@pytest.mark.log1p
def test_log1p():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log1p", torch_op=torch.log1p, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.log1p_
def test_log1p_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log1p_",
        torch_op=lambda a: a.log1p_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
