import pytest
import torch

from . import base, consts


@pytest.mark.lcm
def test_lcm():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm",
        torch_op=torch.lcm,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.lcm_
def test_lcm_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm_",
        torch_op=lambda a, b: a.lcm_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
