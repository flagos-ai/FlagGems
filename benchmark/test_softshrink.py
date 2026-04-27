import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.softshrink
def test_softshrink():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softshrink",
        torch_op=torch.nn.functional.softshrink,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softshrink_out
def test_softshrink_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="softshrink_out",
        torch_op=torch.nn.functional.softshrink,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
