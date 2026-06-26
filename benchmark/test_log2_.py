import pytest
import torch

from . import base, consts


@pytest.mark.log2_
def test_log2_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log2_",
        torch_op=torch.log2_,
        # Generated from the matching unary pointwise entry; worktree lacks a dedicated inplace benchmark.
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
