import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.round
def test_round():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round", torch_op=torch.round, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.round_
def test_round_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round_",
        torch_op=torch.round_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.round_out
def test_round_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="round_out",
        torch_op=torch.round,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
