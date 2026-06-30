import math

import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    base = 1.05
    # calculate the max limit according to dtype
    limit = math.log2(torch.finfo(dtype).max - 1) / math.log2(base)
    end = int(limit)
    yield {
        "start": 0,
        "end": end,
        "steps": math.prod(shape),  # steps influence speed up a lot
        "base": base,
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.logspace
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logspace():
    bench = base.GenericBenchmark(
        op_name="logspace", input_fn=_input_fn, torch_op=torch.logspace
    )
    bench.run()
