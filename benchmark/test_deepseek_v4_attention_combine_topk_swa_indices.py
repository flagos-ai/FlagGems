import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)

from . import base


class CombineTopkSwaIndicesBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices",
            combine_topk_swa_indices,
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
