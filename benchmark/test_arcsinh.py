import pytest
import torch

from . import base, consts


@pytest.mark.arcsinh
def test_arcsinh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsinh", torch_op=torch.arcsinh, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.arcsinh_
def test_arcsinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsinh_",
        torch_op=lambda a: a.arcsinh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.arcsinh_out
def test_arcsinh_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="arcsinh_out",
        torch_op=torch.arcsinh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
