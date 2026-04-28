import pytest
import torch

from . import base, consts


@pytest.mark.isneginf
def test_isneginf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isneginf", torch_op=torch.isneginf, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.skip(reason="No support to non-boolean outputs: issue #2687")
@pytest.mark.isneginf_out
def test_isneginf_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="isneginf_out",
        torch_op=torch.isneginf,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
