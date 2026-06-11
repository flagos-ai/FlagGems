import pytest
import torch

from . import base, consts


@pytest.mark.special_logit
def test_special_logit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_logit",
        torch_op=lambda a: torch.special.logit(a, eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_logit_out
def test_special_logit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_logit_out",
        torch_op=lambda a, out=None: torch.special.logit(a, out=out, eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_logit_
def test_special_logit_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_logit_",
        torch_op=lambda a: a.copy_(torch.special.logit(a, eps=1e-6)),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
