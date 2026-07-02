import pytest
import torch

from . import base, consts


@pytest.mark.is_nonzero
def test_is_nonzero():
    def is_nonzero_input_fn(shape, dtype, device):
        # is_nonzero only accepts single-element tensors
        yield torch.tensor([1], dtype=dtype, device=device)

    bench = base.GenericBenchmark(
        input_fn=is_nonzero_input_fn,
        op_name="is_nonzero",
        torch_op=torch.is_nonzero,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(lambda x: torch.is_nonzero(x))
    bench.run()
