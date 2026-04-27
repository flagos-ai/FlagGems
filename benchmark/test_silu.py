import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.silu
def test_silu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="silu", torch_op=torch.nn.functional.silu, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.silu_
def test_silu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="silu_",
        torch_op=lambda a: torch.nn.functional.silu(a, inplace=True),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
