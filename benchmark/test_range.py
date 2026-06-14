import math

import pytest
import torch

from . import base


def _input_fn(shape, dtype, device):
    yield {
        "start": 0,
        "end": math.prod(shape),
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.range
def test_range():
    # torch.range does not support bfloat16 on CUDA
    dtypes = [torch.float16, torch.float32]
    bench = base.GenericBenchmark(
        op_name="range",
        input_fn=_input_fn,
        torch_op=torch.range,
        dtypes=dtypes,
    )
    bench.run()
