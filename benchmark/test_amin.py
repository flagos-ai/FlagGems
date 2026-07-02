import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.amin
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin",
        torch_op=torch.amin,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.amin_
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin_(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin_",
        torch_op=lambda *a: a[0].copy_(torch.amin(*a, keepdim=True)),
        dtypes=[dtype],
        is_inplace=True,
        gems_op=flag_gems.amin_,
    )
    bench.run()
