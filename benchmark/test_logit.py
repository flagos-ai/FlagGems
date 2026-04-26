import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.logit
def test_logit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit",
        torch_op=lambda a: torch.logit(a, eps=1e-6),
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.logit_
def test_logit_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit_",
        torch_op=lambda a: a.logit_(eps=1e-6),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.skip(reason="The `out` parameter is not supported: issue #2688")
@pytest.mark.logit_out
def test_logit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="logit_out",
        torch_op=lambda a: torch.logit(a, eps=1e-6),
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
