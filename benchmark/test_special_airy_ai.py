import pytest
import torch

from . import base, consts


@pytest.mark.special_airy_ai
def test_special_airy_ai():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_airy_ai",
        torch_op=torch.special.airy_ai,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
