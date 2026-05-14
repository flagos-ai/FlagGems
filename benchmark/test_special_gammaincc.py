import pytest
import torch

from . import base


@pytest.mark.special_gammaincc
def test_special_gammaincc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_gammaincc",
        torch_op=torch.special_gammaincc,
        dtypes=[torch.float32],
    )
    bench.run()
