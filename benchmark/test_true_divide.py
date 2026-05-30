import pytest
import torch

from . import base, consts


@pytest.mark.div_tensor
def test_true_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.true_divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
