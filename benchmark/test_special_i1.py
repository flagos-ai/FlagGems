import pytest
import torch

from . import base, consts


@pytest.mark.special_i1
def test_special_i1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i1", torch_op=torch.special.i1, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.special_i1_out
def test_special_i1_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_i1_out",
        torch_op=torch.special.i1,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
