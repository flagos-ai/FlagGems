import pytest
import torch

from . import base, consts, utils


def frobenius_norm_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = [1]
    keepdim = False
    yield inp, dim, keepdim


@pytest.mark.frobenius_norm
def test_frobenius_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=frobenius_norm_input_fn,
        op_name="frobenius_norm",
        torch_op=torch.ops.aten.frobenius_norm.dim,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
