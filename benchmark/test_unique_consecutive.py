import pytest
import torch

from .attri_util import INT_DTYPES
from .performance_utils import GenericBenchmark2DOnly, generate_tensor_input


def unique_consecutive_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp, {"return_inverse": True, "return_counts": False}
    yield inp, {"return_inverse": True, "return_counts": True}


@pytest.mark.unique_consecutive
def test_unique_consecutive():
    bench = GenericBenchmark2DOnly(
        input_fn=unique_consecutive_input_fn,
        op_name="unique_consecutive",
        torch_op=torch.unique_consecutive,
        dtypes=INT_DTYPES,
    )
    bench.run()
