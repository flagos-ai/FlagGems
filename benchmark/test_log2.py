import pytest
import torch

from . import base, consts


@pytest.mark.log2
def test_log2():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2",
        torch_op=torch.log2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.log2_
def test_log2_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2_",
        torch_op=torch.log2_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
