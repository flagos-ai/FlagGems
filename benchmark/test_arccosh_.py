import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.arccosh_
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_arccosh_(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
        pytest.skip("Kunlunxin native arccosh_ does not support bfloat16 timing")
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccosh_",
        torch_op=lambda a: a.arccosh_(),
        dtypes=[dtype],
        is_inplace=True,
    )
    bench.run()
