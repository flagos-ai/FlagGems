import pytest
import torch

from . import base, consts


@pytest.mark.nonzero_numpy
def test_nonzero_numpy():
    bench = base.GenericBenchmark2DOnly(
        input_fn=base.unary_input_fn,
        op_name="nonzero_numpy",
        torch_op=torch.ops.aten.nonzero_numpy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
