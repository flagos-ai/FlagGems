import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    if dtype in attr_utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    inp = inp[::2]

    yield inp,


@pytest.mark.contiguous
def test_contiguous():
    bench = utils.GenericBenchmark(
        op_name="contiguous",
        input_fn=_input_fn,
        torch_op=torch.Tensor.contiguous,
        dtypes=attr_utils.FLOAT_DTYPES + attr_utils.INT_DTYPES,
    )

    bench.run()
