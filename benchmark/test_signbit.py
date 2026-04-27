import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.signbit
def test_signbit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="signbit", torch_op=torch.signbit, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.skip(reason="No support to non-boolean outputs: issue #2689.")
@pytest.mark.signbit_out
def test_signbit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="signbit_out",
        torch_op=torch.signbit,
        dtypes=attrs.FLOAT_DTYPES,
    )

    bench.run()
