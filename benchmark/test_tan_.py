import pytest
import torch

from . import base, consts


@pytest.mark.tan_
def test_tan_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tan_", torch_op=torch.tan_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
