import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.tril
def test_tril():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=base.unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
