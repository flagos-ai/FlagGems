import pytest
import torch

from . import base, consts


@pytest.mark.view_copy
def test_view_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="view_copy",
        torch_op=lambda a: torch.view_copy(a, (-1,)),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
