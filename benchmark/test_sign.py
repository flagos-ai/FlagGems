import pytest
import torch

from . import base, consts


@pytest.mark.sign
def test_sign():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sign", torch_op=torch.sign, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sign_
def test_sign_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sign_", torch_op=torch.sign_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
