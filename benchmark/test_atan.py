import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.atan
def test_atan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan", torch_op=torch.atan, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.atan_
def test_atan_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan_",
        torch_op=torch.atan_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
