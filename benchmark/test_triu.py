import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.triu
def test_triu():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=base.unary_input_fn,
        op_name="triu",
        torch_op=torch.triu,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.triu_
def test_triu_inplace():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=base.unary_input_fn,
        op_name="triu_",
        torch_op=torch.Tensor.triu_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
