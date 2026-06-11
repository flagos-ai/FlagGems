import pytest
import torch

from . import base, consts


@pytest.mark.special_log1p
def test_special_log1p():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_log1p",
        torch_op=torch.special.log1p,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
