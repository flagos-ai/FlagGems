import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.round
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_round():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round", torch_op=torch.round, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.round_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_round_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="round_",
        torch_op=torch.round_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.round_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_round_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="round_out",
        torch_op=torch.round,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
