import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        compute_global_topk_indices_and_lens as vllm_compute_global_topk_indices_and_lens,
    )

    _HAS_VLLM_COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS = True
except Exception:
    vllm_compute_global_topk_indices_and_lens = None
    _HAS_VLLM_COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS = False

from . import base


class ComputeGlobalTopkIndicesAndLensBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "compute_global_topk_indices_and_lens",
            vllm_compute_global_topk_indices_and_lens,
            [torch.int32],
            gems_op=compute_global_topk_indices_and_lens,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        # (num_tokens, topk, num_reqs, blocks_per_req, block_size)
        self.shapes = [
            # --- tiny: single/few tokens ---
            (1, 4, 1, 16, 64),
            (5, 4, 2, 4, 64),
            (16, 8, 4, 32, 64),
            # --- small batch decode ---
            (32, 32, 1, 64, 64),
            (128, 32, 4, 64, 64),
            (128, 64, 1, 128, 64),
            # --- medium: varying topk ---
            (512, 4, 2, 128, 64),
            (512, 16, 2, 128, 64),
            (512, 64, 2, 128, 64),
            (512, 256, 2, 128, 64),
            # --- medium prefill: varying num_reqs ---
            (2048, 128, 1, 320, 64),
            (2048, 128, 4, 320, 64),
            (2048, 128, 16, 320, 64),
            (4096, 128, 1, 640, 64),
            (4096, 128, 4, 640, 64),
            (4096, 128, 8, 640, 64),
            # --- large prefill ---
            (8192, 128, 4, 1280, 64),
            (8192, 128, 8, 1280, 64),
            (8192, 256, 16, 1280, 64),
            (16384, 128, 8, 2560, 64),
            (16384, 256, 4, 2560, 64),
            (32768, 256, 8, 2560, 64),
            # --- edge: large topk ---
            (4096, 512, 4, 2048, 64),
            # --- edge: many requests ---
            (8192, 128, 32, 640, 64),
        ]

    def get_input_iter(self, dtype):
        _ = dtype
        for num_tokens, topk, num_reqs, blocks_per_req, block_size in self.shapes:
            topk_indices = torch.randint(
                -1,
                blocks_per_req * block_size,
                (num_tokens, topk),
                device="cuda",
                dtype=torch.int32,
            )
            token_to_req_indices = (
                torch.arange(num_tokens, device="cuda", dtype=torch.int32) % num_reqs
            )
            block_table = torch.arange(
                num_reqs * blocks_per_req, device="cuda", dtype=torch.int32
            ).view(num_reqs, blocks_per_req)
            is_valid_token = torch.ones((num_tokens,), device="cuda", dtype=torch.int32)
            yield (
                topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
            )


@pytest.mark.skipif(
    (not torch.cuda.is_available())
    or (not _HAS_VLLM_COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS),
    reason="requires cuda and vllm deepseek_v4_ops.compute_global_topk_indices_and_lens",
)
def test_compute_global_topk_indices_and_lens_benchmark():
    ComputeGlobalTopkIndicesAndLensBenchmark().run()
