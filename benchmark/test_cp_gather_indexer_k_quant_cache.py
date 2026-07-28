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
import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

from flag_gems.fused import cp_gather_indexer_k_quant_cache

from . import base

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")


def run_vllm_benchmark(bench):
    original_str = base.BenchmarkResult.__str__

    def vllm_str(result):
        return (
            original_str(result)
            .replace("Torch Latency (ms)", "vLLM CUDA Latency (ms)")
            .replace("Torch GBPS ", "vLLM CUDA GBPS ")
        )

    base.BenchmarkResult.__str__ = vllm_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


def load_vllm_cuda_op_and_fp8_dtype():
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if getattr(torch.version, "cuda", None) is None:
        pytest.skip("vLLM CUDA custom op requires a CUDA PyTorch build")
    vllm = pytest.importorskip("vllm")
    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            pytest.skip(
                "cp_gather_indexer_k_quant_cache benchmark targets "
                "vLLM CUDA >= 0.20.2 and < 0.21.0"
            )
    except InvalidVersion:
        pass
    try:
        import vllm._custom_ops as ops
        from vllm.platforms import current_platform
    except Exception as exc:
        pytest.skip(f"vLLM CUDA custom ops are unavailable: {exc}")

    if not hasattr(ops, "cp_gather_indexer_k_quant_cache"):
        pytest.skip("vLLM does not provide cp_gather_indexer_k_quant_cache")

    def vllm_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k,
            dst_scale,
            block_table,
            cu_seq_lens,
        )

    return vllm_gather, current_platform.fp8_dtype()


def fill_cache_with_valid_fp8(k_cache, fp8_dtype, head_dim, quant_block_size):
    num_blocks, block_size, _ = k_cache.shape
    num_quant_blocks = head_dim // quant_block_size
    flat_cache = k_cache.view(num_blocks, -1)
    value = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    value.copy_(torch.randn(value.shape, device=k_cache.device).to(fp8_dtype))
    scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    scales.copy_(
        torch.rand(
            num_blocks,
            block_size * num_quant_blocks,
            dtype=torch.float32,
            device=k_cache.device,
        )
        + 0.01
    )


def make_gather_metadata(batch_size, seq_len, block_size, device):
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_lens, dim=0)

    blocks_per_seq = math.ceil(seq_len / block_size)
    block_table = torch.arange(
        batch_size * blocks_per_seq,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_seq)

    return block_table, cu_seqlen


class CpGatherIndexerKQuantCacheBenchmark(base.Benchmark):
    def __init__(self, vllm_op, fp8_dtype):
        super().__init__(
            op_name="cp_gather_indexer_k_quant_cache",
            torch_op=vllm_op,
            dtypes=[torch.float16],
        )
        self.set_gems(cp_gather_indexer_k_quant_cache)
        self.fp8_dtype = fp8_dtype
        self.shape_desc = "batch_size, seq_len, block_size, head_dim, quant_block_size"

    def set_shapes(self, shape_file_path=None):
        block_size = 16
        block_size_deepseek_insert = 64
        deepseek_head_dim = 512
        quant_block_size = 128
        single_seq_shapes = [
            (1, num_tokens, block_size, deepseek_head_dim, quant_block_size)
            for num_tokens in (
                1,
                2,
                4,
                8,
                16,
                17,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
            )
        ]
        decode_batch_shapes = [
            (batch_size, 1, block_size, deepseek_head_dim, quant_block_size)
            for batch_size in (2, 4, 8, 16, 32, 64, 128, 256)
        ]
        multi_batch_shapes = [
            (2, 4096, block_size, deepseek_head_dim, quant_block_size),
            (4, 4096, block_size, deepseek_head_dim, quant_block_size),
            (8, 4096, block_size, deepseek_head_dim, quant_block_size),
            (16, 2048, block_size, deepseek_head_dim, quant_block_size),
            (32, 1024, block_size, deepseek_head_dim, quant_block_size),
            (64, 512, block_size, deepseek_head_dim, quant_block_size),
            (128, 256, block_size, deepseek_head_dim, quant_block_size),
            (256, 128, block_size, deepseek_head_dim, quant_block_size),
        ]
        block_size_64_shapes = [
            (1, 8192, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
            (1, 32768, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
            (1, 65536, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
            (16, 2048, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
            (64, 512, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
            (256, 128, block_size_deepseek_insert, deepseek_head_dim, quant_block_size),
        ]
        self.shapes = (
            single_seq_shapes
            + decode_batch_shapes
            + multi_batch_shapes
            + block_size_64_shapes
        )

    def get_input_iter(self, dtype):
        del dtype
        for batch_size, seq_len, block_size, head_dim, quant_block_size in self.shapes:
            block_table, cu_seqlen = make_gather_metadata(
                batch_size,
                seq_len,
                block_size,
                self.device,
            )
            num_blocks = block_table.numel()
            num_tokens = batch_size * seq_len
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            k_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            fill_cache_with_valid_fp8(
                k_cache,
                self.fp8_dtype,
                head_dim,
                quant_block_size,
            )
            k_fp8 = torch.empty(
                num_tokens,
                head_dim,
                dtype=self.fp8_dtype,
                device=self.device,
            )
            k_fp8_scale = torch.empty(
                num_tokens,
                head_dim * 4 // quant_block_size,
                dtype=torch.uint8,
                device=self.device,
            )
            yield k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cp_gather_indexer_k_quant_cache
def test_cp_gather_indexer_k_quant_cache_benchmark():
    vllm_op, fp8_dtype = load_vllm_cuda_op_and_fp8_dtype()
    bench = CpGatherIndexerKQuantCacheBenchmark(vllm_op, fp8_dtype)
    run_vllm_benchmark(bench)
