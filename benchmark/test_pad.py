import random

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    input = torch.randn(shape, device=device, dtype=dtype)
    rank = input.ndim
    pad_params = [random.randint(0, 10) for _ in range(rank * 2)]
    pad_value = float(torch.randint(0, 1024, [1]))
    yield input, {
        "pad": pad_params,
        "mode": "constant",
        "value": pad_value,
    },


@pytest.mark.pad
def test_pad():
    bench = utils.GenericBenchmark(
        input_fn=_input_fn,
        op_name="pad",
        torch_op=torch.nn.functional.pad,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
