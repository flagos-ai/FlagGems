import pytest
import torch

from . import base, consts


@pytest.mark.acosh
def test_acosh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh",
        torch_op=torch.acosh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.acosh_
def test_acosh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acosh_",
        torch_op=torch.acosh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
