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

import dataclasses
import random

import pytest
import torch

import flag_gems
from flag_gems.utils.device_info import get_device_capability

from . import base


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


VLLM_REF_AVAILABLE = hasattr(
    torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
)
HEAD_DIM = 512
ROPE_DIM = 64
HEAD_BYTES = 584


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    num_tokens: int
    num_heads: int
    num_tokens_insert: int
    block_size: int
    max_pos: int
    eps: float
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


_random_counter = 0


class FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
            [torch.bfloat16],
        )
        self.set_gems(flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for (
            param
        ) in (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.get_performance_test_params()
        ):
            yield from FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input(
                param
            )

    @staticmethod
    def get_performance_test_params():
        cases = [
            TestParam(
                num_tokens,
                num_heads,
                num_tokens_insert=num_tokens,
                block_size=64,
                max_pos=4096,
                eps=1e-6,
            )
            for num_tokens in [
                1,
                4,
                17,
                64,
                1024,
                2048,
                8192,
                32768,
                65536,
                98304,
                131072,
            ]
            for num_heads in [64, 128]
        ]
        return cases

    @staticmethod
    def init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
        if max_pos <= 8192:
            base = 10000.0
        elif max_pos <= 32768:
            base = 20000.0
        elif max_pos <= 65536:
            base = 40000.0
        elif max_pos <= 98304:
            base = 60000.0
        else:
            base = 100000.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
        return cache.to(dtype)

    @staticmethod
    def make_input(param: TestParam):
        num_tokens = param.num_tokens
        num_heads = param.num_heads
        num_tokens_insert = param.num_tokens_insert
        block_size = param.block_size
        max_pos = max(param.max_pos, num_tokens)
        eps = param.eps
        dtype = param.dtype
        device = param.device

        global _random_counter
        FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.init_seed(_random_counter)
        _random_counter = _random_counter + 1

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_cos_sin_cache(
                max_pos, ROPE_DIM, torch.float32, device
            )
        )

        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        slot_mapping = torch.arange(num_tokens_insert, dtype=torch.int64, device=device)
        k_cache = torch.zeros(
            num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
        )
        yield (q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not VLLM_REF_AVAILABLE, reason="The referenced vLLM implementation is not installed"
)
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
def test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert():
    bench = FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark()
    bench.run()
