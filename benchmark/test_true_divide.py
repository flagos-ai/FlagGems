import pytest
import torch

from . import base, consts


@pytest.mark.true_divide
def test_true_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="true_divide",
        torch_op=torch.true_divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
