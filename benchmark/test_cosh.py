import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.cosh
def test_cosh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cosh", torch_op=torch.cosh, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.cosh_
def test_cosh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cosh_",
        torch_op=torch.cosh_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.cosh_out
def test_cosh_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="cosh_out",
        torch_op=torch.cosh,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
