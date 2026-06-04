import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"input": inp, "high": 10},


@pytest.mark.randint_like
def test_randint_like():
    bench = base.GenericBenchmark(
        op_name="randint_like",
        input_fn=_input_fn,
        torch_op=torch.randint_like,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
