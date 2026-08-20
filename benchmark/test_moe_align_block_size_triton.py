# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems
from flag_gems.fused.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)

from . import base

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

    # Correct buffer size calculation (matches moe_align_block_size implementation)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
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


class MoeAlignBlockSizeBenchmark(base.GenericBenchmark4DOnly):
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


def _fast_path_input_fn(shape, dtype, device):
    fast_path, num_experts, block_size, num_tokens, topk = shape
    num_routes = num_tokens * topk
    topk_ids = (torch.arange(num_routes, device=device) % 4).to(torch.int32)
    yield topk_ids.reshape(num_tokens, topk), num_experts, block_size, fast_path


def _standard_align(topk_ids, num_experts, block_size, _fast_path):
    return moe_align_block_size(topk_ids, block_size, num_experts)


def _fast_align(topk_ids, num_experts, block_size, fast_path):
    if fast_path == "singleton":
        return moe_align_block_size_singleton(topk_ids, block_size)
    return moe_align_block_size_small_grouped(topk_ids, num_experts, block_size)


class MoeAlignBlockSizeFastPathBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ("singleton", 8, 8, 1, 2),
            ("small_grouped", 8, 8, 4, 2),
        ]

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


@pytest.mark.moe_align_block_size_triton
def test_moe_align_block_size_fast_paths():
    bench = MoeAlignBlockSizeFastPathBenchmark(
        op_name="moe_align_block_size_fast_paths",
        input_fn=_fast_path_input_fn,
        torch_op=_standard_align,
        dtypes=[torch.int32],
    )
    bench.set_gems(_fast_align)
    print(
        "Column mapping: Torch Latency = standard FlagGems alignment; "
        "Gems Latency = singleton/small_grouped fast path "
        "(selected path shown in Size Detail)"
    )
    bench.run()
