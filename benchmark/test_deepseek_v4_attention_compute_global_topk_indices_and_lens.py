import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)

from . import base


def torch_compute_global_topk_indices_and_lens(
    topk_indices,
    token_to_req_indices,
    block_table,
    block_size,
    is_valid_token=None,
):
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
    lens = torch.empty(
        (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
    )
    for token_idx in range(topk_indices.shape[0]):
        req_idx = int(token_to_req_indices[token_idx].item())
        valid_count = 0
        for topk_idx in range(topk_indices.shape[1]):
            local_idx = int(topk_indices[token_idx, topk_idx].item())
            if local_idx >= 0:
                block_idx = local_idx // block_size
                block_off = local_idx % block_size
                block_no = int(block_table[req_idx, block_idx].item())
                global_indices[token_idx, topk_idx] = block_no * block_size + block_off
                valid_count += 1
            else:
                global_indices[token_idx, topk_idx] = -1
        lens[token_idx] = valid_count if bool(is_valid_token[token_idx].item()) else 0
    return global_indices, lens


class ComputeGlobalTopkIndicesAndLensBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "compute_global_topk_indices_and_lens",
            torch_compute_global_topk_indices_and_lens,
            [torch.int32],
            gems_op=compute_global_topk_indices_and_lens,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(4096, 128)]

    def get_input_iter(self, dtype):
        _ = dtype
        for num_tokens, topk in self.shapes:
            topk_indices = torch.randint(
                -1, 64, (num_tokens, topk), device="cuda", dtype=torch.int32
            )
            token_to_req_indices = torch.zeros(
                (num_tokens,), device="cuda", dtype=torch.int32
            )
            block_table = torch.arange(0, 256, device="cuda", dtype=torch.int32).view(
                1, -1
            )
            yield (topk_indices, token_to_req_indices, block_table, 64, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_compute_global_topk_indices_and_lens_benchmark():
    ComputeGlobalTopkIndicesAndLensBenchmark().run()
