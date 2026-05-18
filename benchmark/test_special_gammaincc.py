import pytest
import torch

from . import base, consts


@pytest.mark.special_gammaincc
def test_special_gammaincc():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_gammaincc",
        torch_op=torch.special.gammaincc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
