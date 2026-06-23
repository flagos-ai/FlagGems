import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Accuracy shapes cover 1D and batched vector inputs while keeping the n x n output small.
VANDER_SHAPES = [(4,), (8,), (16,), (2, 4), (2, 8), (3, 16), (2, 3, 4)]


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
# torch.linalg.vander does not support fp16/bf16 on CUDA in this PyTorch build.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_vander(shape, dtype):
    # linalg_vander input is (*, n) where last dim is the vector
    # Output is (*, n, N) where N defaults to n (or can be specified)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Convert to float32 for reference since torch.linalg.vander doesn't support fp16/bf16 on CPU
    ref_inp = utils.to_reference(inp).to(torch.float32)

    ref_out = torch.linalg.vander(ref_inp).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.linalg.vander(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_vander
@pytest.mark.parametrize("shape", VANDER_SHAPES)
# torch.linalg.vander does not support fp16/bf16 on CUDA in this PyTorch build.
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("N", [2, 3, 5])
def test_linalg_vander_with_N(shape, dtype, N):
    # Test with explicit N parameter
    if N > shape[-1]:
        # Skip if N is larger than input vector length
        return
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Convert to float32 for reference since torch.linalg.vander doesn't support fp16/bf16 on CPU
    ref_inp = utils.to_reference(inp).to(torch.float32)

    ref_out = torch.linalg.vander(ref_inp, N=N).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.linalg.vander(inp, N=N)

    utils.gems_assert_close(res_out, ref_out, dtype)
