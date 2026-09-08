import pytest
import torch

from . import base, consts, utils


def unsafe_chunk_input_fn(shape, dtype, device):
    # Generate different chunks values for different shapes
    for chunks in [2, 3, 4, 5]:
        # Skip invalid combinations
        dim_size = shape[0]
        if chunks > dim_size:
            continue
        inp = utils.generate_tensor_input(shape, dtype, device)
        yield inp, {
            "chunks": chunks,
            "dim": 0,
        }


@pytest.mark.unsafe_chunk
def test_unsafe_chunk():
    bench = base.GenericBenchmark(
        input_fn=unsafe_chunk_input_fn,
        op_name="unsafe_chunk",
        torch_op=torch.unsafe_chunk,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
