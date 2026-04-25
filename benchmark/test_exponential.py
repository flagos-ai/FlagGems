import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.exponential_
def test_exponential_inplace():
    bench = utils.GenericBenchmark(
        op_name="exponential_",
        input_fn=utils.unary_input_fn,
        torch_op=torch.Tensor.exponential_,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
