import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unsafe_chunk
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("chunks", [2, 3, 5])
@pytest.mark.parametrize("dim", [0, -1])
def test_unsafe_chunk(shape, dtype, chunks, dim):
    """Test unsafe_chunk accuracy against PyTorch implementation."""
    # Skip invalid dim combinations
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dimension")

    # Skip when chunks > size of dim
    dim_size = shape[dim] if dim >= 0 else shape[dim + len(shape)]
    if chunks > dim_size:
        pytest.skip("chunks > dim size")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unsafe_chunk(ref_inp, chunks, dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_chunk(inp, chunks, dim)

    # Compare number of chunks
    assert len(res_out) == len(ref_out), "Number of chunks mismatch"

    # Compare each chunk
    for res_chunk, ref_chunk in zip(res_out, ref_out):
        res_chunk_cpu = utils.to_reference(res_chunk)
        ref_chunk_cpu = utils.to_reference(ref_chunk)
        utils.gems_assert_close(res_chunk_cpu, ref_chunk_cpu, dtype)
