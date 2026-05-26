import pytest
import torch

from . import base, consts


@pytest.mark.fmax
def test_fmax():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmax",
        torch_op=torch.fmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmax_out
def test_fmax_out():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmax_out",
        torch_op=torch.fmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
