import pytest
import torch

from . import base, consts


@pytest.mark.log10
def test_log10():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log10", torch_op=torch.log10, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.log10_
def test_log10_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log10_",
        torch_op=torch.log10_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.log10_out
def test_log10_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="log10_out",
        torch_op=torch.log10,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
