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

import random

import pytest
import torch

import flag_gems

from . import base, consts

# Try to import vllm reshape_and_cache
try:
    from vllm._custom_ops import reshape_and_cache as vllm_reshape_and_cache

    _HAS_VLLM_RESHAPE_AND_CACHE = True
except Exception:
    vllm_reshape_and_cache = None
    _HAS_VLLM_RESHAPE_AND_CACHE = False


class ReshapeAndCacheBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _make_input_kwargs(shape, dtype, device):
    """Shared input generator for reshape_and_cache benchmarks."""
    (
        num_tokens,
        num_heads,
        head_size,
        block_size,
        num_blocks,
    ) = shape
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype, device=device)
    _, key, value = qkv.unbind(dim=1)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
    key_cache.uniform_(-scale, scale)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
    value_cache.uniform_(-scale, scale)

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    yield (
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        {
            "kv_cache_dtype": "auto",
            "k_scale": k_scale,
            "v_scale": v_scale,
        },
    )


@pytest.mark.reshape_and_cache
@pytest.mark.skipif(not _HAS_VLLM_RESHAPE_AND_CACHE, reason="vLLM not installed")
def test_reshape_and_cache():
    """Benchmark FlagGems reshape_and_cache against vllm baseline."""
    bench = ReshapeAndCacheBenchmark(
        op_name="reshape_and_cache",
        input_fn=_make_input_kwargs,
        torch_op=vllm_reshape_and_cache,
        gems_op=flag_gems.reshape_and_cache,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
