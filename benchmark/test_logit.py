import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.logit
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit",
        torch_op=lambda a: torch.logit(a, eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.logit_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logit_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit_",
        torch_op=lambda a: a.logit_(eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.logit_out
def test_logit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="logit_out",
        torch_op=lambda a, out: torch.logit(a, eps=1e-6, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
