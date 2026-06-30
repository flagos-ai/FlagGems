import pytest
import torch

from . import base, consts


@pytest.mark.i0
def test_i0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="i0", torch_op=torch.i0, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.i0_
def test_i0_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="i0_",
        torch_op=torch.Tensor.i0_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.i0_out
def test_i0_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="i0_out",
        torch_op=torch.i0,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
