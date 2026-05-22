import pytest
import torch

from . import base, consts


@pytest.mark.not_equal
def test_perf_not_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="not_equal",
        torch_op=torch.not_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
