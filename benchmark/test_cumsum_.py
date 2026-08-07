import pytest
import torch

from . import base, consts, utils


def input_fn(shape, dtype, device):
    # Clone for inplace operation to avoid modifying the same tensor across iterations
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp.clone(), 1


@pytest.mark.cumsum_
def test_cumsum_():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumsum_",
        torch_op=torch.Tensor.cumsum_,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()
