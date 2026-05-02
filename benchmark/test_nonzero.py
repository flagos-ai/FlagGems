import pytest
import torch

from . import base, consts, utils


@pytest.mark.nonzero
def test_nonzero():
    bench = base.GenericBenchmark2DOnly(
        input_fn=utils.unary_input_fn,
        op_name="nonzero",
        torch_op=torch.nonzero,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
