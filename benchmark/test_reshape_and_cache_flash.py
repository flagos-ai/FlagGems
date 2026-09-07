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

# Try to import vllm reshape_and_cache_flash
try:
    from vllm._custom_ops import reshape_and_cache_flash as vllm_reshape_and_cache_flash

    _HAS_VLLM_RESHAPE_AND_CACHE_FLASH = True
except Exception:
    vllm_reshape_and_cache_flash = None
    _HAS_VLLM_RESHAPE_AND_CACHE_FLASH = False


class ReshapeAndCacheFlashBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _make_input_kwargs(shape, dtype, device):
    """Shared input generator for reshape_and_cache_flash benchmarks."""
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

    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5
    key_value_cache = torch.empty(
        size=key_value_cache_shape, dtype=dtype, device=device
    )
    key_value_cache.uniform_(-scale, scale)
    key_cache = key_value_cache[:, 0].contiguous()
    value_cache = key_value_cache[:, 1].contiguous()

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


@pytest.mark.reshape_and_cache_flash
@pytest.mark.skipif(not _HAS_VLLM_RESHAPE_AND_CACHE_FLASH, reason="vLLM not installed")
def test_reshape_and_cache_flash():
    """Benchmark FlagGems reshape_and_cache_flash against vllm baseline."""
    bench = ReshapeAndCacheFlashBenchmark(
        op_name="reshape_and_cache_flash",
        input_fn=_make_input_kwargs,
        torch_op=vllm_reshape_and_cache_flash,
        gems_op=flag_gems.reshape_and_cache_flash,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
