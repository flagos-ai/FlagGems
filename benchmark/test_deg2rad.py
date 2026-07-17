import pytest
import torch

from . import base, consts


@pytest.mark.deg2rad
def test_deg2rad():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad", torch_op=torch.deg2rad, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.deg2rad_
def test_deg2rad_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad_",
        torch_op=lambda x: x.deg2rad_(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.deg2rad_out
def test_deg2rad_out():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad_out",
        torch_op=lambda x: torch.deg2rad(x, out=x),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
