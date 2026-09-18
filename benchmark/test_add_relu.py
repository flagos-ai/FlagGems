import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.add_relu
def test_add_relu():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_relu",
        torch_op=lambda a, b: torch.relu(a + b),
        gems_op=flag_gems._add_relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
