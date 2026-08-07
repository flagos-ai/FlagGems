import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.chunk
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunks", [2, 3, 7])
@pytest.mark.parametrize("dim", [0, -1])
def test_chunk(shape, dtype, chunks, dim):
    """Test chunk accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.chunk(ref_inp, chunks, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.chunk(inp, chunks, dim=dim)

    assert len(res_out) == len(
        ref_out
    ), f"chunk count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)
