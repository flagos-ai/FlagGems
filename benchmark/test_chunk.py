import pytest
import torch

from . import base


def chunk_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Default: split into 3 chunks along dim 0
    yield inp, {"chunks": 3, "dim": 0}


@pytest.mark.chunk
def test_chunk():
    bench = base.GenericBenchmark(
        input_fn=chunk_input_fn,
        op_name="chunk",
        torch_op=torch.chunk,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
