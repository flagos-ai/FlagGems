import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    inp = inp[::2]

    yield inp,


@pytest.mark.contiguous
def test_contiguous():
    bench = base.GenericBenchmark(
        op_name="contiguous",
        input_fn=_input_fn,
        torch_op=torch.Tensor.contiguous,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()
