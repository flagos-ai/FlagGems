import pytest
import torch

from . import base, consts


@pytest.mark.sinh
def test_sinh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh",
        torch_op=torch.sinh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sinh_
def test_sinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh_",
        torch_op=lambda a: a.sinh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
