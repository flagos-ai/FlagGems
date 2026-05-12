import pytest
import torch

from flag_gems.fused.fused_qnorm_rope_kv import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref,
)

from . import base


class FusedQNormRopeKVBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 128),
            (4, 128),
            (17, 128),
            (64, 128),
            (1024, 128),
            (2048, 128),
        ]

    def get_input_iter(self, dtype):
        cache_block_size = 16
        max_pos = 8192

        for N, H in self.shapes:
            torch.manual_seed(42)
            q = torch.randn((N, H, 512), dtype=torch.bfloat16, device=self.device)
            kv = torch.randn((N, 576), dtype=torch.bfloat16, device=self.device)

            num_blocks = (N + cache_block_size - 1) // cache_block_size + 1
            cache_stride = cache_block_size * 576 + cache_block_size * 8
            k_cache = torch.zeros(
                (num_blocks, cache_stride), dtype=torch.uint8, device=self.device
            )

            slot_mapping = torch.arange(N, dtype=torch.int32, device=self.device)
            position_ids = torch.randint(
                0, max_pos, (N,), dtype=torch.int64, device=self.device
            )
            cos_sin_cache = torch.randn(
                (max_pos, 64), dtype=torch.float32, device=self.device
            )

            yield (
                q, kv, k_cache, slot_mapping,
                position_ids, cos_sin_cache,
                1e-6, cache_block_size,
            )


@pytest.mark.fused_qnorm_rope_kv
def test_fused_qnorm_rope_kv():
    bench = FusedQNormRopeKVBenchmark(
        op_name="fused_qnorm_rope_kv",
        torch_op=fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref,
        gems_op=fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
        dtypes=[torch.bfloat16],
    )
    bench.run()
