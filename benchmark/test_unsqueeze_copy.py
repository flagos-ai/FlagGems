import pytest
import torch

from . import base, consts


@pytest.mark.unsqueeze_copy
def test_unsqueeze_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="unsqueeze_copy",
        torch_op=lambda a: torch.unsqueeze_copy(a, 0),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
