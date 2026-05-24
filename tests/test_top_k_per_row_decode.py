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


def _try_load_vllm_top_k_per_row_decode():
    try:
        import vllm
        import vllm._C

        torch.ops.load_library(vllm._C.__file__)
        return torch.ops._C.top_k_per_row_decode
    except Exception:
        return None


def _torch_top_k_per_row_decode_reference(logits, row_starts, row_ends, top_k):
    num_rows = logits.shape[0]
    indices = torch.empty(
        (num_rows, top_k),
        dtype=torch.int32,
        device=logits.device,
    )

    for row_id in range(num_rows):
        start = int(row_starts[row_id].item())
        end = int(row_ends[row_id].item())

        row_slice = logits[row_id, start:end]
        _, topk_idx = torch.topk(
            row_slice,
            top_k,
            largest=True,
            sorted=False,
        )
        indices[row_id] = topk_idx.to(torch.int32)

    return indices


def _selected_values(logits, indices, row_starts):
    abs_indices = indices.long() + row_starts.long().view(-1, 1)
    return logits.gather(1, abs_indices).sort(dim=1).values


def _check_decode_case(vocab_size, seq_len, top_k, row_start=0, check_vllm=True):
    torch.manual_seed(2026)

    logits = torch.randn(1, vocab_size, device=device, dtype=torch.float32)
    row_starts = torch.tensor([row_start], dtype=torch.int32, device=device)
    row_ends = torch.tensor([seq_len], dtype=torch.int32, device=device)

    gems_indices = torch.empty((1, top_k), dtype=torch.int32, device=device)
    gems_logits = logits.clone()

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
    torch.cuda.synchronize()

    torch_indices = _torch_top_k_per_row_decode_reference(
        logits,
        row_starts,
        row_ends,
        top_k,
    )

    valid_len = seq_len - row_start

    assert torch.all(gems_indices >= 0)
    assert torch.all(gems_indices < valid_len)

    torch.testing.assert_close(
        _selected_values(logits, gems_indices, row_starts),
        _selected_values(logits, torch_indices, row_starts),
        rtol=1e-5,
        atol=1e-5,
    )

    if row_start > 0:
        assert torch.isneginf(gems_logits[0, :row_start]).all()

    if seq_len < vocab_size:
        assert torch.isneginf(gems_logits[0, seq_len:]).all()

    assert torch.allclose(
        gems_logits[0, row_start:seq_len],
        logits[0, row_start:seq_len],
    )

    if not check_vllm:
        return

    vllm_top_k_per_row_decode = _try_load_vllm_top_k_per_row_decode()
    if vllm_top_k_per_row_decode is None:
        return

    if row_start != 0:
        return

    vllm_indices = torch.empty_like(gems_indices)
    vllm_logits = logits.clone()

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

    assert torch.all(vllm_indices >= 0)
    assert torch.all(vllm_indices < seq_len)

    torch.testing.assert_close(
        _selected_values(logits, gems_indices, row_starts),
        _selected_values(logits, vllm_indices, row_starts),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "vocab_size, seq_len, top_k",
    [
        (4096, 4096, 64),
        (8192, 6144, 128),
        (32768, 32768, 512),
        (129280, 129280, 1024),
    ],
)
def test_top_k_per_row_decode_matches_reference_and_vllm_if_available(
    vocab_size,
    seq_len,
    top_k,
):
    _check_decode_case(
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
        row_start=0,
        check_vllm=True,
    )


@pytest.mark.parametrize(
    "vocab_size, seq_len, top_k, row_start",
    [
        (4096, 3500, 128, 33),
        (8192, 7000, 256, 777),
    ],
)
def test_top_k_per_row_decode_nonzero_row_start_matches_torch(
    vocab_size,
    seq_len,
    top_k,
    row_start,
):
    _check_decode_case(
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
        row_start=row_start,
        check_vllm=False,
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
