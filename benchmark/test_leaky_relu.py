import pytest
import torch

from . import base, consts


@pytest.mark.leaky_relu
def test_leaky_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.leaky_relu_
def test_leaky_relu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu_",
        torch_op=torch.nn.functional.leaky_relu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.leaky_relu_out
def test_leaky_relu_out():
    bench = base.UnaryPointwiseBenchmark(
        op_name="leaky_relu_out",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
