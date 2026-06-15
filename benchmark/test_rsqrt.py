import pytest
import torch

from . import base, consts


@pytest.mark.rsqrt
def test_rsqrt():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rsqrt", torch_op=torch.rsqrt, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.rsqrt_
def test_rsqrt_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rsqrt_",
        torch_op=torch.rsqrt_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
