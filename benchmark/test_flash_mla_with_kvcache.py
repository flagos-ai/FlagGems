import dataclasses
import math
import random

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
    is_fp8: bool = True
    have_attn_sink: bool = False
    have_topk_length: bool = False
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


_counter = 0


def _init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_fp8_kv_cache(
    num_pages, page_block_size, h_kv, d_nope=512, d_rope=64, device="cuda"
):
    """Generate realistic FP8 KV cache data (no NaN byte patterns)."""
    total_tokens = num_pages * page_block_size

    nope_data = (
        torch.randn(total_tokens, h_kv, d_nope, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / 448.0
    scales = scales.clamp(min=1e-12)
    quantized = (groups.float() / scales[:, :, None]).clamp(-448, 448)
    fp8_data = quantized.reshape(-1, 512).to(torch.float8_e4m3fn)

    rope_data = (
        torch.randn(total_tokens, h_kv, d_rope, dtype=torch.bfloat16, device=device)
        * 0.1
    )

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_kv, 656, dtype=torch.uint8, device=device
    )
    fp8_bytes = fp8_data.view(torch.uint8).reshape(
        num_pages, page_block_size, h_kv, 512
    )
    kv_cache[:, :, :, :512] = fp8_bytes
    scales_reshaped = scales.reshape(num_pages, page_block_size, h_kv, 4)
    scales_bytes = (
        scales_reshaped.to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_kv, 16)
    )
    kv_cache[:, :, :, 512:528] = scales_bytes
    rope_bytes = (
        rope_data.reshape(num_pages, page_block_size, h_kv, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_kv, 128)
    )
    kv_cache[:, :, :, 528:656] = rope_bytes

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
            # Sparse FP8 decode: batch=128, varying topk
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
            # Sparse FP8 decode: batch=64, larger topk
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
            # Dense BF16 decode: batch=128, varying seqlen
            + [
                TestParam(
                    batch=128, topk=0, seqlen=seqlen, h_q=128, d_qk=576, is_fp8=False
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
            # Sparse mode
            topk = param.topk
            num_pages = math.ceil(topk / page_block_size) + 4
            k_cache = generate_fp8_kv_cache(
                num_pages, page_block_size, h_kv, device=device
            )
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
                topk_length = torch.randint(
                    1, topk + 1, (batch,), dtype=torch.int32, device=device
                )
                kwargs["topk_length"] = topk_length

            block_table = None
            cache_seqlens = None
        else:
            # Dense mode
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
            # Add some variation
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
