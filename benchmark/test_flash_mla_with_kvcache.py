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
import math
import random
from typing import Optional

import pytest
import torch

import flag_gems

from . import base

try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache as cuda_flash_mla,
    )
    from vllm.third_party.flashmla.flash_mla_interface import (
        get_mla_metadata as cuda_get_mla_metadata,
    )

    HAS_CUDA_FLASHMLA = True
except ImportError:
    HAS_CUDA_FLASHMLA = False

from flag_gems.fused.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as triton_flash_mla,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    get_mla_metadata as triton_get_mla_metadata,
)

FP8_MAX = 448.0


@dataclasses.dataclass
class TestParam:
    __test__ = False

    batch: int
    topk: int = 0
    seqlen: int = 0
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 576
    d_v: int = 512
    page_block_size: int = 64
    num_pages: Optional[int] = None
    is_fp8: bool = True
    have_attn_sink: bool = False
    have_topk_length: bool = False
    extra_topk: int = 0
    extra_page_block_size: int = 64
    extra_num_pages: Optional[int] = None
    use_out: bool = False
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


_counter = 0


def _init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_v32_fp8_kv_cache(
    num_pages, page_block_size, h_kv=1, d_nope=512, d_rope=64, device="cuda"
):
    """Generate V32 FP8 KV cache data (656 bytes per token)."""
    total_tokens = num_pages * page_block_size

    nope_data = (
        torch.randn(total_tokens, h_kv, d_nope, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    quantized = (groups.float() / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8_data = quantized.reshape(-1, d_nope).to(torch.float8_e4m3fn)

    rope_data = (
        torch.randn(total_tokens, h_kv, d_rope, dtype=torch.bfloat16, device=device)
        * 0.1
    )

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_kv, 656, dtype=torch.uint8, device=device
    )
    kv_cache[:, :, :, :512] = fp8_data.view(torch.uint8).reshape(
        num_pages, page_block_size, h_kv, 512
    )
    kv_cache[:, :, :, 512:528] = (
        scales.reshape(num_pages, page_block_size, h_kv, 4)
        .to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_kv, 16)
    )
    kv_cache[:, :, :, 528:656] = (
        rope_data.reshape(num_pages, page_block_size, h_kv, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_kv, 128)
    )
    return kv_cache


def generate_model1_fp8_kv_cache(
    num_pages, page_block_size, h_kv=1, d_nope=448, d_rope=64, device="cuda"
):
    """Generate MODEL1 page-oriented KV cache data (584 bytes per token)."""
    assert h_kv == 1, "FlashMLA sparse decode currently supports h_kv == 1"
    total_tokens = num_pages * page_block_size
    token_data_bytes = d_nope + d_rope * 2
    scale_bytes = 8

    nope_data = (
        torch.randn(total_tokens, d_nope, dtype=torch.bfloat16, device=device) * 0.1
    )
    groups = nope_data.reshape(total_tokens, 7, 64).float()
    block_max = torch.clamp(groups.abs().amax(dim=-1), min=1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scales = torch.exp2(exponent)
    fp8_data = (
        torch.clamp(groups / scales[:, :, None], -FP8_MAX, FP8_MAX)
        .reshape(total_tokens, d_nope)
        .to(torch.float8_e4m3fn)
    )
    rope_data = (
        torch.randn(total_tokens, d_rope, dtype=torch.bfloat16, device=device) * 0.1
    )

    data_bytes = torch.empty(
        total_tokens, token_data_bytes, dtype=torch.uint8, device=device
    )
    data_bytes[:, :d_nope] = fp8_data.view(torch.uint8).reshape(total_tokens, d_nope)
    data_bytes[:, d_nope:] = rope_data.view(torch.uint8).reshape(total_tokens, 128)

    encoded_scales = torch.zeros(
        total_tokens, scale_bytes, dtype=torch.uint8, device=device
    )
    encoded_scales[:, :7] = torch.clamp(exponent + 127.0, 0, 255).to(torch.uint8)

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_kv, 584, dtype=torch.uint8, device=device
    )
    page_flat = kv_cache.view(num_pages, -1)
    page_flat[:, : page_block_size * token_data_bytes] = data_bytes.reshape(
        num_pages, page_block_size * token_data_bytes
    )
    page_flat[:, page_block_size * token_data_bytes :] = encoded_scales.reshape(
        num_pages, page_block_size * scale_bytes
    )
    return kv_cache


def _cuda_wrapper(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    meta, _ = cuda_get_mla_metadata()
    return cuda_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, meta, **kwargs
    )


def _triton_wrapper(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    meta, _ = triton_get_mla_metadata()
    return triton_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, meta, **kwargs
    )


class FlashMLAWithKVCacheBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache",
            _cuda_wrapper,
            [torch.bfloat16],
        )
        self.set_gems(_triton_wrapper)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for param in self.get_performance_test_params():
            yield from self.make_input(param)

    @staticmethod
    def get_performance_test_params():
        cases = (
            # V32 sparse FP8 decode: d_qk=576, 656-byte cache.
            [
                TestParam(
                    batch=128,
                    topk=topk,
                    h_q=128,
                    d_qk=576,
                    is_fp8=True,
                    have_attn_sink=True,
                )
                for topk in [128, 256, 512, 1024, 2048]
            ]
            + [
                TestParam(
                    batch=64,
                    topk=topk,
                    h_q=128,
                    d_qk=576,
                    is_fp8=True,
                    have_attn_sink=True,
                )
                for topk in [256, 512, 1024, 2048, 4096]
            ]
            # MODEL1 sparse FP8 decode: log-like d_qk=512, 584-byte cache.
            + [
                TestParam(
                    batch=512,
                    topk=128,
                    h_q=64,
                    d_qk=512,
                    num_pages=512,
                    is_fp8=True,
                    have_attn_sink=True,
                    have_topk_length=True,
                    use_out=True,
                ),
                TestParam(
                    batch=512,
                    topk=128,
                    h_q=64,
                    d_qk=512,
                    num_pages=512,
                    is_fp8=True,
                    have_attn_sink=True,
                    have_topk_length=True,
                    extra_topk=512,
                    extra_page_block_size=64,
                    extra_num_pages=512,
                    use_out=True,
                ),
                TestParam(
                    batch=512,
                    topk=128,
                    h_q=64,
                    d_qk=512,
                    num_pages=512,
                    is_fp8=True,
                    have_attn_sink=True,
                    have_topk_length=True,
                    extra_topk=8192,
                    extra_page_block_size=2,
                    extra_num_pages=512,
                    use_out=True,
                ),
            ]
            # Dense BF16 decode: seq_q=1 only.
            + [
                TestParam(
                    batch=128,
                    topk=0,
                    seqlen=seqlen,
                    h_q=128,
                    d_qk=576,
                    is_fp8=False,
                )
                for seqlen in [256, 512, 1024, 2048, 4096]
            ]
        )
        return cases

    @staticmethod
    def make_input(param: TestParam):
        global _counter
        _init_seed(_counter)
        _counter += 1

        batch = param.batch
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        d_v = param.d_v
        page_block_size = param.page_block_size
        dtype = param.dtype
        device = param.device

        q = torch.randn(batch, 1, h_q, d_qk, dtype=dtype, device=device) / 10
        kwargs = {}

        if param.topk > 0:
            topk = param.topk
            num_pages = param.num_pages or (math.ceil(topk / page_block_size) + 4)
            if d_qk == 576:
                k_cache = generate_v32_fp8_kv_cache(
                    num_pages, page_block_size, h_kv, device=device
                )
            elif d_qk == 512:
                k_cache = generate_model1_fp8_kv_cache(
                    num_pages, page_block_size, h_kv, device=device
                )
            else:
                raise ValueError(f"Unsupported sparse d_qk: {d_qk}")

            total_tokens = num_pages * page_block_size
            indices = torch.randint(
                0, total_tokens, (batch, 1, topk), dtype=torch.int32, device=device
            )

            kwargs["is_fp8_kvcache"] = True
            kwargs["indices"] = indices

            if param.have_attn_sink:
                attn_sink = torch.randn(h_q, dtype=torch.float32, device=device)
                mask = torch.randn(h_q, dtype=torch.float32, device=device)
                attn_sink[mask < -0.5] = float("-inf")
                attn_sink[mask > 0.5] = float("+inf")
                kwargs["attn_sink"] = attn_sink

            if param.have_topk_length:
                kwargs["topk_length"] = torch.randint(
                    1, topk + 1, (batch,), dtype=torch.int32, device=device
                )

            if param.extra_topk > 0:
                extra_num_pages = param.extra_num_pages or (
                    math.ceil(param.extra_topk / param.extra_page_block_size) + 4
                )
                extra_k_cache = generate_model1_fp8_kv_cache(
                    extra_num_pages,
                    param.extra_page_block_size,
                    h_kv,
                    device=device,
                )
                extra_indices = torch.randint(
                    0,
                    extra_num_pages * param.extra_page_block_size,
                    (batch, 1, param.extra_topk),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["extra_k_cache"] = extra_k_cache
                kwargs["extra_indices_in_kvcache"] = extra_indices
                kwargs["extra_topk_length"] = torch.randint(
                    1,
                    param.extra_topk + 1,
                    (batch,),
                    dtype=torch.int32,
                    device=device,
                )

            if param.use_out:
                kwargs["out"] = torch.empty(
                    batch, 1, h_q, d_v, dtype=dtype, device=device
                )

            block_table = None
            cache_seqlens = None
        else:
            seqlen = param.seqlen
            max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
            total_pages = batch * max_pages_per_seq
            k_cache = (
                torch.randn(
                    total_pages, page_block_size, h_kv, d_qk, dtype=dtype, device=device
                )
                * 0.1
            )
            block_table = torch.arange(
                total_pages, dtype=torch.int32, device=device
            ).view(batch, max_pages_per_seq)
            cache_seqlens = torch.full(
                (batch,), seqlen, dtype=torch.int32, device=device
            )
            cache_seqlens[0] = max(seqlen // 2, 1)
            if batch > 1:
                cache_seqlens[-1] = min(
                    seqlen + page_block_size, total_pages * page_block_size
                )
            kwargs["causal"] = param.causal

        yield (q, k_cache, block_table, cache_seqlens, d_v, kwargs)


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.skipif(not HAS_CUDA_FLASHMLA, reason="vLLM FlashMLA not installed")
def test_flash_mla_with_kvcache():
    bench = FlashMLAWithKVCacheBenchmark()
    bench.run()
