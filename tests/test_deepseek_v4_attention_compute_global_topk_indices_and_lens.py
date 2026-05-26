import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)


@pytest.mark.parametrize(
    ("topk_values", "req_values", "block_table_values", "valid_values", "block_size"),
    [
        (
            [[0, 3, -1, 7], [2, -1, 4, 5], [1, 0, -1, -1]],
            [0, 1, 0],
            [[5, 6], [9, 10]],
            [1, 1, 0],
            4,
        ),
        (
            [[0, 63, 64, 127], [128, -1, 191, 255], [7, 8, -1, -1]],
            [0, 1, 1],
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            [1, 1, 0],
            64,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_compute_global_topk_indices_and_lens_accuracy(
    topk_values, req_values, block_table_values, valid_values, block_size
):
    device = "cuda"
    topk_indices = torch.tensor(topk_values, device=device, dtype=torch.int32)
    token_to_req_indices = torch.tensor(req_values, device=device, dtype=torch.int32)
    block_table = torch.tensor(block_table_values, device=device, dtype=torch.int32)
    is_valid_token = torch.tensor(valid_values, device=device, dtype=torch.int32)

    actual_indices, actual_lens = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
    )

    expected = torch.full_like(topk_indices, -1)
    expected_lens = torch.zeros(
        (topk_indices.shape[0],), device=device, dtype=torch.int32
    )
    for token in range(topk_indices.shape[0]):
        req = int(token_to_req_indices[token].item())
        count = 0
        for i in range(topk_indices.shape[1]):
            local = int(topk_indices[token, i].item())
            if local >= 0:
                block_idx = local // block_size
                block_off = local % block_size
                expected[token, i] = (
                    block_table[req, block_idx] * block_size + block_off
                )
                count += 1
        expected_lens[token] = count if int(is_valid_token[token].item()) else 0

    fg_testing.assert_equal(actual_indices, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)
