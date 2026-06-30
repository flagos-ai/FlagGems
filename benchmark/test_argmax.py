import pytest
import torch

from . import base, consts


@pytest.mark.argmax
def test_argmax():
    bench = base.UnaryReductionBenchmark(
        op_name="argmax", torch_op=torch.argmax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
