import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


@pytest.mark.uniform_
def test_uniform_inplace():
    bench = utils.GenericBenchmark(
        input_fn=utils.unary_input_fn,
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
