import pytest
import torch

from . import base, consts


@pytest.mark.expand_copy
def test_expand_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expand_copy",
        torch_op=lambda a: torch.ops.aten.expand_copy(
            a, tuple(a.shape[:-1]) + (2,) if a.shape[-1] == 1 else a.shape
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
