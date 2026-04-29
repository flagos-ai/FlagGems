import math
import random

import pytest
import torch

import flag_gems

from . import base, utils


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
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="Only torch >= 2.5 is supported on Kunlunxin",
)
def test_linspace():
    bench = base.GenericBenchmark(
        op_name="linspace", input_fn=_input_fn, torch_op=torch.linspace
    )
    bench.run()
