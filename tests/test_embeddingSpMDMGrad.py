import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Unsupported on cambricon"
)
@pytest.mark.embeddingSpMDMGrad
@pytest.mark.parametrize(
    "Batch, M, N, embeddingsize",
    [
        (2, 4, 8, 16),
        (4, 8, 32, 64),
        (1, 3, 64, 128),
    ],
)
@pytest.mark.parametrize("padding_idx", [-1, 0, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("seed", [42])
def test_embedding_sp_mdm_grad(Batch, M, N, embeddingsize, padding_idx, dtype, seed):
    """Test for embeddingSpMDMGrad operator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    grad_output = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    indices = torch.randint(
        0, embeddingsize, (Batch, M), device=flag_gems.device, dtype=torch.long
    )
    if padding_idx >= 0 and embeddingsize > 0:
        mask = torch.rand((Batch, M), device=flag_gems.device) < 0.25
        indices = torch.where(mask, torch.full_like(indices, padding_idx), indices)
    num_weights = embeddingsize

    # Reference implementation using embedding_dense_backward
    ref_grad_output = utils.to_reference(grad_output)
    ref_indices = utils.to_reference(indices)
    ref_out = torch.ops.aten.embedding_dense_backward(
        ref_grad_output,
        ref_indices,
        num_weights,
        padding_idx,
        False,  # scale_grad_by_freq=False for comparison
    )

    # GEMS implementation - call directly since embeddingSpMDMGrad is not a standard aten op
    with flag_gems.use_gems():
        res_out = flag_gems.embeddingSpMDMGrad(
            grad_output, indices, num_weights, padding_idx
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
