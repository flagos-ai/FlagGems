import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.embedding
@pytest.mark.parametrize("EmbeddingSize", [1024] if cfg.TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if cfg.TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if cfg.TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if cfg.TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device=flag_gems.device, requires_grad=False
    )
    res_embedding = torch.randn(
        (EmbeddingSize, N), device=flag_gems.device, dtype=dtype, requires_grad=True
    )
    ref_embedding = utils.to_reference(res_embedding)
    ref_indices = utils.to_reference(res_indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            res_indices,
            res_embedding,
            padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding_backward
@pytest.mark.parametrize("EmbeddingSize", [1024] if cfg.TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if cfg.TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if cfg.TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if cfg.TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [-1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_backward(
    EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype
):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_grad = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    res_indices = torch.randint(0, EmbeddingSize, (Batch, M), device=flag_gems.device)
    num_weights = EmbeddingSize
    sparse = False

    ref_grad = utils.to_reference(res_grad)
    ref_indices = utils.to_reference(res_indices)

    ref_in_grad = torch.ops.aten.embedding_backward(
        ref_grad, ref_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.embedding_backward(
            res_grad, res_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.embedding_dense_backward
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.parametrize(
    "Batch, M, N, embeddingsize",
    [
        (2, 4, 8, 16),
        (4, 8, 32, 64),
        (1, 3, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "padding_idx, scale_grad_by_freq", [(-1, False), (0, True), (5, False)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("seed", [42])
def test_embedding_dense_backward(
    Batch, M, N, embeddingsize, padding_idx, scale_grad_by_freq, dtype, seed
):
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
    ref_grad_output = utils.to_reference(grad_output)
    ref_indices = utils.to_reference(indices)
    ref_out = torch.ops.aten.embedding_dense_backward(
        ref_grad_output,
        ref_indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.embedding_dense_backward(
            grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
        )
    # res_out = torch.ops.aten.embedding_dense_backward(
    # grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)

    utils.gems_assert_close(res_out, ref_out, dtype)
