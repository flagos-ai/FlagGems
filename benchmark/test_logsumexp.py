import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


@pytest.mark.logsumexp
def test_logsumexp():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="logsumexp",
        torch_op=torch.logsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
