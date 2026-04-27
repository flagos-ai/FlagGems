import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},


@pytest.mark.unique
def test_unique():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=attr_utils.INT_DTYPES,
    )

    bench.run()
