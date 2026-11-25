import pytest
import torch

from . import base, consts


@pytest.mark.clone
def test_clone():
    bench = base.UnaryPointwiseBenchmark(
        op_name="clone",
        torch_op=torch.clone,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
