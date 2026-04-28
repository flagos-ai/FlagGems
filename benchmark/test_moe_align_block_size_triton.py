import pytest
import torch

import flag_gems

from . import performance_utils as utils

try:
    import os

    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    import vllm._custom_ops as vllm_ops

    HAS_VLLM = True
    WARP_SIZE = 32
except ImportError:
    HAS_VLLM = False
    WARP_SIZE = 0


def _input_fn(shape, dtype, device):
    num_experts = shape[0]
    block_size = shape[1]
    dtype = torch.int32
    topk_ids = torch.randint(
        0, num_experts, (shape[2], shape[3]), dtype=dtype, device=device
    )
    max_num_tokens_padded = ((num_experts + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE

    # padded_num_experts in vllm._custom_ops.moe_align_block_size
    # must be less than 1024
    if max_num_tokens_padded >= 1024:
        return

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    yield (
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


class MoeAlignBlockSizeBenchmark(utils.GenericBenchmark4DOnly):
    def set_shapes(self, shape_file_path: None):
        moe_align_block_size_shape = [
            (512, 64, 16384, 10),
            (512, 64, 6152, 10),
            (512, 64, 4727, 10),
            (512, 64, 1905, 10),
            (512, 64, 11575, 10),
            (512, 64, 1032, 10),
            (512, 64, 4201, 10),
            (512, 64, 2056, 10),
            (512, 64, 7561, 10),
            (512, 64, 4104, 10),
            (512, 64, 14281, 10),
        ]
        self.shapes = moe_align_block_size_shape

    def set_more_shapes(self):
        return []


@pytest.mark.moe_align_block_size_triton
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_moe_align_block_size_triton():
    gems_op = flag_gems.moe_align_block_size_triton
    bench = MoeAlignBlockSizeBenchmark(
        op_name="moe_align_block_size_triton",
        input_fn=_input_fn,
        torch_op=vllm_ops.moe_align_block_size,
        dtypes=[
            torch.int32,
        ],
    )

    bench.set_gems(gems_op)
    bench.run()
