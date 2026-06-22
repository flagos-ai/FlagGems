import pytest
import torch

from . import base, consts, utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


@pytest.mark.cumsum
def test_cumsum():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumsum",
        torch_op=torch.cumsum,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()


@pytest.mark.cumsum_out
def test_cumsum_out():
    bench = base.GenericBenchmark(
        op_name="cumsum_out",
        torch_op=lambda inp, dim, out: torch.cumsum(inp, dim, out=out),
        input_fn=lambda shape, dtype, device: (
            (
                torch.randn(shape, dtype=dtype, device=device),
                1 if len(shape) > 1 else 0,
                torch.empty(shape, dtype=dtype, device=device),
            )
            for _ in [None]
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
