import pytest
import torch

from . import base, consts


@pytest.mark.neg
def test_neg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg", torch_op=torch.neg, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.neg_
def test_neg_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg_", torch_op=torch.neg_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
