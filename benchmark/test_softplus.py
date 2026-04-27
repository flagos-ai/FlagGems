import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.softplus
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
