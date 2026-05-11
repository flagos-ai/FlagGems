import math

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    yield {
        "end": math.prod(shape),
        "device": device,
        "dtype": dtype,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "start": 0,
            "end": math.prod(shape),
            "step": 2,
            "device": device,
            "dtype": dtype,
        },


@pytest.mark.arange
def test_arange():
    bench = base.GenericBenchmark(
        op_name="arange", input_fn=_input_fn, torch_op=torch.arange
    )
    bench.run()
