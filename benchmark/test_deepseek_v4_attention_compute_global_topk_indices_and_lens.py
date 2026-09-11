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

import os

import pytest
import torch
import yaml

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
    # (num_tokens, topk, num_reqs, blocks_per_req, block_size)
    DEFAULT_SHAPES = [
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
    DEFAULT_SHAPE_DESC = "num_tokens, topk, num_reqs, blocks_per_req, block_size"

    def __init__(self):
        super().__init__(
            "compute_global_topk_indices_and_lens",
            vllm_compute_global_topk_indices_and_lens,
            [torch.int32],
            gems_op=compute_global_topk_indices_and_lens,
        )

    def set_shapes(self, shape_file_path=None):
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shapes = self.DEFAULT_SHAPES
        if not shape_file_path or not os.path.isfile(shape_file_path):
            return

        with open(shape_file_path, "r") as file:
            yaml_config = yaml.safe_load(file) or {}
        config = yaml_config.get(self.op_name)
        if not config:
            return
        shapes = config.get("shapes", self.DEFAULT_SHAPES)
        self.shapes = [self._normalize_shape(shape) for shape in shapes]

    def _normalize_shape(self, shape):
        shape = tuple(int(dim) for dim in shape)
        if len(shape) == 5:
            return shape
        if len(shape) == 2:
            num_tokens, topk = shape
            block_size = 64
            num_reqs = max(1, min(num_tokens, 32))
            blocks_per_req = max(1, (topk + block_size - 1) // block_size)
            return (num_tokens, topk, num_reqs, blocks_per_req, block_size)
        raise ValueError(
            "compute_global_topk_indices_and_lens expects shape "
            "(num_tokens, topk) or "
            "(num_tokens, topk, num_reqs, blocks_per_req, block_size)"
        )

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
@pytest.mark.compute_global_topk_indices_and_lens
def test_compute_global_topk_indices_and_lens_benchmark():
    ComputeGlobalTopkIndicesAndLensBenchmark().run()
