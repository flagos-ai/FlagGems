import math
import random

import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    limit = torch.finfo(dtype).max - 1
    num = int(min(limit, math.prod(shape)))
    yield {
        "start": 0,
        "end": num,
        "steps": random.randint(1, num),
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.linspace
def test_linspace():
    bench = base.GenericBenchmark(
        op_name="linspace", input_fn=_input_fn, torch_op=torch.linspace
    )
    bench.run()
