import pytest
import torch

from . import base, consts


@pytest.mark.round
def test_round():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round", torch_op=torch.round, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.round_
def test_round_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round_",
        torch_op=torch.round_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.round_out
def test_round_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="round_out",
        torch_op=torch.round,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
