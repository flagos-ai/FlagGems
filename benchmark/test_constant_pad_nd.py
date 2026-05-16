import random

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, device=device, dtype=dtype)
    rank = inp.ndim
    pad = [random.randint(0, 10) for _ in range(rank * 2)]
    value = 1.5
    yield inp, {"pad": pad, "value": value}


@pytest.mark.constant_pad_nd
def test_constant_pad_nd():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="constant_pad_nd",
        torch_op=torch.constant_pad_nd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
