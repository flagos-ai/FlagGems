import pytest
import torch

from . import base, consts


@pytest.mark.exp2
def test_exp2():
    bench = base.UnaryPointwiseBenchmark(
        op_name="exp2", torch_op=torch.exp2, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.exp2_
def test_exp2_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="exp2_",
        torch_op=torch.exp2_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
