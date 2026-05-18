import pytest
import torch

import flag_gems
from flag_gems.fused.sparse_attn_indexer.top_k_per_row_decode import (
    top_k_per_row_decode,
)

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


def _load_vllm_top_k_per_row_decode():
    pytest.importorskip("vllm")
    try:
        import vllm._C

        torch.ops.load_library(vllm._C.__file__)
        return torch.ops._C.top_k_per_row_decode
    except Exception as exc:
        pytest.skip(f"vLLM top_k_per_row_decode is unavailable: {exc}")


def _selected_values(logits, indices):
    return logits.gather(1, indices.long()).sort(dim=1).values


@pytest.mark.parametrize(
    "vocab_size, seq_len, top_k",
    [
        (4096, 4096, 64),
        (8192, 6144, 128),
        (32768, 32768, 512),
        (129280, 129280, 1024),
    ],
)
def test_top_k_per_row_decode_matches_vllm(vocab_size, seq_len, top_k):
    vllm_top_k_per_row_decode = _load_vllm_top_k_per_row_decode()

    torch.manual_seed(2026)
    logits = torch.randn(1, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.zeros(1, dtype=torch.int32, device=device)
    row_ends = torch.tensor([seq_len], dtype=torch.int32, device=device)
    gems_indices = torch.empty((1, top_k), dtype=torch.int32, device=device)
    vllm_indices = torch.empty_like(gems_indices)

    gems_logits = logits.clone()
    vllm_logits = logits.clone()

    top_k_per_row_decode(
        gems_logits,
        row_starts,
        row_ends,
        gems_indices,
        1,
        gems_logits.stride(0),
        gems_logits.stride(1),
        top_k,
    )
    vllm_top_k_per_row_decode(
        vllm_logits,
        1,
        row_ends,
        vllm_indices,
        1,
        vllm_logits.stride(0),
        vllm_logits.stride(1),
        top_k,
    )
    torch.cuda.synchronize()

    assert torch.all(gems_indices >= 0)
    assert torch.all(gems_indices < seq_len)
    assert torch.all(vllm_indices >= 0)
    assert torch.all(vllm_indices < seq_len)

    torch.testing.assert_close(
        _selected_values(logits, gems_indices),
        _selected_values(logits, vllm_indices),
        rtol=1e-5,
        atol=1e-5,
    )


def test_top_k_per_row_decode_rejects_batched_input():
    logits = torch.randn(2, 4096, device=device, dtype=torch.float32)
    row_starts = torch.zeros(2, dtype=torch.int32, device=device)
    row_ends = torch.full((2,), 4096, dtype=torch.int32, device=device)
    indices = torch.empty((2, 128), dtype=torch.int32, device=device)

    with pytest.raises(ValueError):
        top_k_per_row_decode(
            logits,
            row_starts,
            row_ends,
            indices,
            2,
            logits.stride(0),
            logits.stride(1),
            128,
        )
