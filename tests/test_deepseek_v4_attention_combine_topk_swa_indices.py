import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)


@pytest.mark.parametrize(
    (
        "topk_values",
        "query_start_values",
        "seq_len_values",
        "gather_len_values",
        "window_size",
        "compress_ratio",
        "topk",
        "M",
        "N",
    ),
    [
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [0, 2, 3],
            [8, 10],
            [8, 10],
            4,
            2,
            4,
            64,
            16,
        ),
        (
            [
                [100, 101, 102, 103],
                [110, 111, 112, 113],
                [120, 121, 122, 123],
                [130, 131, 132, 133],
                [140, 141, 142, 143],
            ],
            [0, 3, 5],
            [6, 4],
            [4, 3],
            3,
            2,
            4,
            20,
            8,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_accuracy(
    topk_values,
    query_start_values,
    seq_len_values,
    gather_len_values,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    device = "cuda"
    topk_indices = torch.tensor(topk_values, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor(query_start_values, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(seq_len_values, device=device, dtype=torch.int32)
    gather_lens = torch.tensor(gather_len_values, device=device, dtype=torch.int32)

    actual, actual_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )

    expected = torch.full_like(actual, -1)
    expected_lens = torch.empty_like(actual_lens)
    for batch in range(seq_lens.numel()):
        start = int(query_start_loc[batch].item()) - int(query_start_loc[0].item())
        end = int(query_start_loc[batch + 1].item()) - int(query_start_loc[0].item())
        query_len = end - start
        seq_len = int(seq_lens[batch].item())
        gather_len = int(gather_lens[batch].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        for token_idx in range(start, end):
            token_in_query = token_idx - start
            pos = start_pos + token_in_query
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            if topk_len:
                expected[token_idx, :topk_len] = (
                    topk_indices[token_idx, :topk_len] + M * batch
                )
            for j in range(swa_len):
                expected[token_idx, topk_len + j] = (
                    M * batch + N + j + pos - swa_len + 1 - gather_start
                )
            expected_lens[token_idx] = topk_len + swa_len

    fg_testing.assert_equal(actual, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)
