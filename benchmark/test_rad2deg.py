import pytest
import torch

from . import base, consts


@pytest.mark.rad2deg
def test_rad2deg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rad2deg",
        torch_op=torch.rad2deg,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rad2deg_
def test_rad2deg_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rad2deg_",
        torch_op=torch.Tensor.rad2deg_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
