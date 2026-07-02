import pytest
import torch

from . import base, consts


@pytest.mark.sinc
def test_sinc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinc",
        torch_op=torch.sinc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sinc_
def test_sinc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinc_",
        torch_op=torch.sinc_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
