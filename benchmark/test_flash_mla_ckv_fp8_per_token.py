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

import math

import pytest
import torch

from flag_gems.fused.flash_mla_ckv_fp8_per_token import (
    HAS_TLE,
    prepare_flash_mla_ckv_fp8_per_token,
    quantize_k_ckv_per_token,
    quantize_q_ckv_per_token,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as bf16_flash_mla,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    get_mla_metadata as get_bf16_mla_metadata,
)

from . import base

STANDARD_SHAPES = [
    (batch, seqlen, h_q)
    for seqlen, h_q in ((640, 128), (8192, 64), (33280, 64))
    for batch in (1, 2, 4, 8, 16, 32, 64, 128)
]
_PREPARED = {}


class FlashMLACKVFP8PerTokenBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        self.shapes = STANDARD_SHAPES
        self.shape_desc = "batch, sequence length, query heads"


def _input_fn(shape, dtype, device):
    batch, seqlen, h_q = shape
    pages_per_row = math.ceil(seqlen / 64)
    total_pages = batch * pages_per_row
    q = torch.randn(batch, 1, h_q, 576, dtype=dtype, device=device) * 0.1
    blocked_k = torch.randn(total_pages, 64, 1, 576, dtype=dtype, device=device) * 0.1
    q_nope, q_rope, q_scale = quantize_q_ckv_per_token(q)
    k_lora, k_rope, k_scale = quantize_k_ckv_per_token(blocked_k.squeeze(2))
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_row
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    yield (
        q,
        blocked_k,
        q_nope,
        q_rope,
        q_scale,
        k_lora,
        k_rope,
        k_scale,
        block_table,
        cache_seqlens,
        (seqlen,) * batch,
    )


def _bf16(
    q,
    blocked_k,
    q_nope,
    q_rope,
    q_scale,
    k_lora,
    k_rope,
    k_scale,
    block_table,
    cache_seqlens,
    lengths,
):
    del q_nope, q_rope, q_scale, k_lora, k_rope, k_scale, lengths
    metadata, _ = get_bf16_mla_metadata()
    return bf16_flash_mla(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        512,
        metadata,
        causal=False,
    )


def _fp8(
    q,
    blocked_k,
    q_nope,
    q_rope,
    q_scale,
    k_lora,
    k_rope,
    k_scale,
    block_table,
    cache_seqlens,
    lengths,
):
    del q, blocked_k
    key = (
        int(q_nope.data_ptr()),
        int(k_lora.data_ptr()),
        tuple(lengths),
    )
    prepared = _PREPARED.get(key)
    if prepared is None:
        # Keep only the active shape so the 24-shape matrix does not retain
        # every KV cache and prepared workspace on the GPU.
        _PREPARED.clear()
        handle, (out, lse) = prepare_flash_mla_ckv_fp8_per_token(
            q_nope,
            q_rope,
            k_lora,
            k_rope,
            q_scale,
            k_scale,
            block_table,
            cache_seqlens,
            512,
            initial_cache_seqlens=lengths,
            max_cache_seqlens=lengths,
        )
        prepared = (handle, out, lse)
        _PREPARED[key] = prepared
    handle, out, lse = prepared
    return handle(out=out, lse=lse)


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(
    not (HAS_TLE and _is_hopper()),
    reason="requires an NVIDIA Hopper GPU and FlagTree TLE support",
)
@pytest.mark.flash_mla_ckv_fp8_per_token
def test_flash_mla_ckv_fp8_per_token():
    bench = FlashMLACKVFP8PerTokenBenchmark(
        op_name="flash_mla_ckv_fp8_per_token",
        input_fn=_input_fn,
        torch_op=_bf16,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_fp8)
    bench.run()
