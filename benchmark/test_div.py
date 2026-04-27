import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.div
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div",
        torch_op=torch.div,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_
def test_div_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
