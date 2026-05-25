import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)

from . import base


class ComputeGlobalTopkIndicesAndLensBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "compute_global_topk_indices_and_lens",
            compute_global_topk_indices_and_lens,
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
