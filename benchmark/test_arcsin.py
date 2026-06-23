import pytest
import torch

from . import base, consts


@pytest.mark.arcsin
def test_arcsin():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsin", torch_op=torch.arcsin, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.arcsin_
def test_arcsin_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsin_",
        torch_op=lambda a: a.arcsin_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.arcsin_out
def test_arcsin_out():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsin_out",
        torch_op=lambda x, out=None: torch.arcsin(x, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
