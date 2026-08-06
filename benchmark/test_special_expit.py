import pytest
import torch

from . import base, consts


@pytest.mark.special_expit
def test_special_expit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_expit",
        torch_op=torch.special.expit,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
