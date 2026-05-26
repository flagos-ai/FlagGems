import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )

    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = True
except Exception:
    vllm_combine_topk_swa_indices = None
    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = False

from . import base


class CombineTopkSwaIndicesBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices",
            vllm_combine_topk_swa_indices,
            [torch.int32],
            gems_op=combine_topk_swa_indices,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            ([3, 2], [6, 4], [4, 3], 4, 4, 2, 20, 8),
            ([128], [512], [256], 32, 128, 4, 42240, 40960),
            ([512, 256], [2048, 1024], [1024, 512], 64, 256, 4, 45056, 40960),
            ([4096], [4096], [4096], 128, 256, 4, 45056, 40960),
            ([1024, 1024], [8192, 4096], [2048, 1024], 128, 256, 4, 45056, 40960),
            ([128], [4096], [512], 32, 256, 128, 5632, 1280),
            ([4096], [4096], [4096], 128, 256, 128, 8448, 1280),
        ]

    def get_input_iter(self, dtype):
        _ = dtype
        for (
            query_lens,
            seq_lens_values,
            gather_lens_values,
            topk,
            window_size,
            compress_ratio,
            M,
            N,
        ) in self.shapes:
            num_tokens = sum(query_lens)
            topk_indices = torch.randint(
                -1, max(N, 1), (num_tokens, topk), device="cuda", dtype=torch.int32
            )
            query_start_values = [0]
            for query_len in query_lens:
                query_start_values.append(query_start_values[-1] + query_len)
            query_start_loc = torch.tensor(
                query_start_values, device="cuda", dtype=torch.int32
            )
            seq_lens = torch.tensor(seq_lens_values, device="cuda", dtype=torch.int32)
            gather_lens = torch.tensor(
                gather_lens_values, device="cuda", dtype=torch.int32
            )
            yield (
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


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not _HAS_VLLM_COMBINE_TOPK_SWA_INDICES),
    reason="requires cuda and vllm deepseek_v4_ops.combine_topk_swa_indices",
)
def test_combine_topk_swa_indices_benchmark():
    CombineTopkSwaIndicesBenchmark().run()
