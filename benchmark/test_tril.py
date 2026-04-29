import pytest
import torch

from . import base, consts, utils


@pytest.mark.tril
def test_tril():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril_
def test_tril_inplace():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril_",
        torch_op=torch.Tensor.tril_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
