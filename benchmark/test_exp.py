import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.exp
def test_exp():
    bench = base.UnaryPointwiseBenchmark(
        op_name="exp", torch_op=torch.exp, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.exp_
def test_exp_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="exp_", torch_op=torch.exp_, dtypes=attrs.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()


@pytest.mark.exp_out
def test_exp_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="exp_out",
        torch_op=torch.exp,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
