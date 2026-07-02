import pytest
import torch

from . import base, consts


@pytest.mark.asin
def test_asin():
    bench = base.UnaryPointwiseBenchmark(
        op_name="asin",
        torch_op=torch.asin,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.asin_
def test_asin_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="asin_",
        torch_op=torch.asin_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
