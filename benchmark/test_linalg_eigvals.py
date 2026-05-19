import pytest
import torch

from . import base


@pytest.mark.linalg_eigvals
def test_linalg_eigvals():
    def input_fn(shape, dtype, device):
        batch = shape[0]
        return (torch.randn((batch, 2, 2), dtype=dtype, device=device),)

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="linalg_eigvals",
        torch_op=torch.linalg.eigvals,
        # Triton kernel uses float32 arithmetic for eigenvalue computation; Half/BFloat16 not supported
        dtypes=[torch.float32],
        shapes=[(16,), (64,), (256,), (1024,)],
    )
    bench.run()
