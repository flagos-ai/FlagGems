import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test configurations for embeddingSpMDM
EMBEDDING_SPMDM_NUM_EMBEDDINGS_LIST = [100, 1000, 10000]
EMBEDDING_SPMDM_EMBEDDING_DIM_LIST = [128, 256, 512]
EMBEDDING_SPMDM_OUTPUT_DIM_LIST = [64, 128, 256]
EMBEDDING_SPMDM_BATCH_SIZE_LIST = [1, 8, 32]


@pytest.mark.embeddingSpMDM
@pytest.mark.parametrize("num_embeddings", EMBEDDING_SPMDM_NUM_EMBEDDINGS_LIST)
@pytest.mark.parametrize("embedding_dim", EMBEDDING_SPMDM_EMBEDDING_DIM_LIST)
@pytest.mark.parametrize("output_dim", EMBEDDING_SPMDM_OUTPUT_DIM_LIST)
@pytest.mark.parametrize("batch_size", EMBEDDING_SPMDM_BATCH_SIZE_LIST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embeddingSpMDM(
    num_embeddings,
    embedding_dim,
    output_dim,
    batch_size,
    dtype,
):
    """Test embeddingSpMDM accuracy against PyTorch reference."""
    # Generate random embedding table
    weight = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )

    # Generate random indices
    indices = torch.randint(0, num_embeddings, (batch_size,), device=flag_gems.device)

    # Generate random dense matrix
    dense = torch.randn(embedding_dim, output_dim, dtype=dtype, device=flag_gems.device)

    # Reference: compute using PyTorch
    # Disable TF32 to ensure fair comparison - Triton kernel uses full FP32 accumulation
    _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ref_weight = utils.to_reference(weight)
    ref_indices = utils.to_reference(indices)
    ref_dense = utils.to_reference(dense)

    ref_out = ref_weight[ref_indices] @ ref_dense
    torch.backends.cuda.matmul.allow_tf32 = _prev_tf32

    # Compute using FlagGems
    with flag_gems.use_gems():
        res_out = flag_gems.embeddingSpMDM(weight, indices, dense)

    # Compare
    utils.gems_assert_close(res_out, ref_out, dtype)
