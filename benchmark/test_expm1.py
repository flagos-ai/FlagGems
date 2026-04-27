import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.expm1
def test_expm1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expm1", torch_op=torch.expm1, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.expm1_
def test_expm1_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expm1_",
        torch_op=torch.expm1_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.expm1_out
def test_expm1_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="expm1_out",
        torch_op=torch.expm1,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
