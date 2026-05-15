import pytest
import torch

from . import base, consts


@pytest.mark.gt
def test_gt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
