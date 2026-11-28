import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.xor
def test_xor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xor",
        torch_op=torch.bitwise_xor,
        gems_op=flag_gems.xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
