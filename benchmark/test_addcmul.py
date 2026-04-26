import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


def _input_fn(shape, cur_dtype, device):
    inp1 = base.generate_tensor_input(shape, cur_dtype, device)
    inp2 = base.generate_tensor_input(shape, cur_dtype, device)
    inp3 = base.generate_tensor_input(shape, cur_dtype, device)

    yield inp1, inp2, inp3, {"value": 0.5}


@pytest.mark.addcmul
def test_addcmul():
    bench = base.GenericBenchmark(
        op_name="addcmul",
        input_fn=_input_fn,
        torch_op=torch.addcmul,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
