import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)

from . import base


def torch_combine_topk_swa_indices(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    num_tokens = topk_indices.shape[0]
    alignment = 128
    combined_topk = (topk + window_size + alignment - 1) // alignment * alignment
    combined = torch.full(
        (num_tokens, combined_topk),
        -1,
        device=topk_indices.device,
        dtype=torch.int32,
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    base_start = int(query_start_loc[0].item())
    for batch_idx in range(seq_lens.numel()):
        query_start = int(query_start_loc[batch_idx].item()) - base_start
        query_end = int(query_start_loc[batch_idx + 1].item()) - base_start
        query_len = query_end - query_start
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        for token_idx in range(query_start, query_end):
            token_in_query = token_idx - query_start
            pos = start_pos + token_in_query
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            if topk_len > 0:
                combined[token_idx, :topk_len] = (
                    topk_indices[token_idx, :topk_len] + M * batch_idx
                )
            if swa_len > 0:
                swa_values = (
                    M * batch_idx
                    + N
                    + torch.arange(
                        swa_len, device=topk_indices.device, dtype=torch.int32
                    )
                    + pos
                    - swa_len
                    + 1
                    - gather_start
                )
                combined[token_idx, topk_len : topk_len + swa_len] = swa_values
            lens[token_idx] = topk_len + swa_len
    return combined, lens


class CombineTopkSwaIndicesBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices",
            torch_combine_topk_swa_indices,
            [torch.int32],
            gems_op=combine_topk_swa_indices,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(4096, 128)]

    def get_input_iter(self, dtype):
        _ = dtype
        for num_tokens, topk in self.shapes:
            topk_indices = torch.randint(
                -1, 2048, (num_tokens, topk), device="cuda", dtype=torch.int32
            )
            query_start_loc = torch.tensor(
                [0, num_tokens], device="cuda", dtype=torch.int32
            )
            seq_lens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)
            gather_lens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)
            yield (
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                256,
                8,
                topk,
                8192,
                4096,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_combine_topk_swa_indices_benchmark():
    CombineTopkSwaIndicesBenchmark().run()
