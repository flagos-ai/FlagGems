import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.positive
def test_positive():
    bench = base.UnaryPointwiseBenchmark(
        op_name="positive",
        torch_op=torch.positive,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.positive)
    bench.run()
