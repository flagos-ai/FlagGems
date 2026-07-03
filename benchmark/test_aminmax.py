import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import Config, GenericBenchmark, generate_tensor_input


def aminmax_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp,
    yield inp, {"dim": -1}
    if len(shape) > 1:
        yield inp, {"dim": 0}
    if Config.bench_level.value == "comprehensive" and len(shape) > 1:
        yield inp, {"dim": 1, "keepdim": True}


@pytest.mark.aminmax
def test_aminmax():
    bench = GenericBenchmark(
        input_fn=aminmax_input_fn,
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
