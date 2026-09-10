import pytest
import torch

from . import base, consts


@pytest.mark.special_sinc
def test_special_sinc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_sinc",
        torch_op=torch.special.sinc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
