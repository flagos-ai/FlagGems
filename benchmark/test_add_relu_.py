import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.add_relu_
def test_add_relu_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_relu_",
        torch_op=lambda a, b: a.copy_(torch.relu(a + b)),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        gems_op=flag_gems._add_relu_,
    )
    bench.run()
