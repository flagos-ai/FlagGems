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
from packaging.version import InvalidVersion, Version

from flag_gems.fused import indexer_k_quant_and_cache

from . import base

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")


def is_fp8e4nv_supported():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


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


def load_vllm_cuda_op():
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if getattr(torch.version, "cuda", None) is None:
        pytest.skip("vLLM CUDA custom op requires a CUDA PyTorch build")
    vllm = pytest.importorskip("vllm")
    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            pytest.skip(
                "indexer_k_quant_and_cache benchmark targets "
                "vLLM CUDA >= 0.20.2 and < 0.21.0"
            )
    except InvalidVersion:
        pass
    try:
        import vllm._custom_ops as ops
    except Exception as exc:
        pytest.skip(f"vLLM CUDA custom ops are unavailable: {exc}")

    if not hasattr(ops, "indexer_k_quant_and_cache"):
        pytest.skip("vLLM does not provide indexer_k_quant_and_cache")

    def vllm_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    return vllm_indexer


class IndexerKQuantAndCacheBenchmark(base.Benchmark):
    def __init__(self, vllm_op):
        super().__init__(
            op_name="indexer_k_quant_and_cache",
            torch_op=vllm_op,
            dtypes=[torch.float16, torch.bfloat16],  # vLLM supports both K dtypes.
        )
        self.set_gems(indexer_k_quant_and_cache)
        self.shape_desc = (
            "num_tokens, num_blocks, block_size, head_dim, quant_block_size"
        )

    def set_shapes(self, shape_file_path=None):
        head_dim = 512
        quant_block_size = 128
        block_size = 16
        token_sweep = (
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
        self.shapes = [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in token_sweep
        ]
        block_size = 64
        self.shapes += [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (8192, 32768, 65536)
        ]

    def get_input_iter(self, dtype):
        for (
            num_tokens,
            num_blocks,
            block_size,
            head_dim,
            quant_block_size,
        ) in self.shapes:
            k = torch.randn(
                num_tokens,
                head_dim,
                dtype=dtype,
                device=self.device,
            )
            slot_mapping = torch.randperm(
                num_blocks * block_size,
                device=self.device,
            )[
                :num_tokens
            ].to(torch.long)
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            kv_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            yield k, kv_cache, slot_mapping, quant_block_size, {"scale_fmt": "ue8m0"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not is_fp8e4nv_supported(),
    reason="fp8e4nv requires device capability >= 8.9",
)
@pytest.mark.indexer_k_quant_and_cache
def test_indexer_k_quant_and_cache_benchmark():
    bench = IndexerKQuantAndCacheBenchmark(load_vllm_cuda_op())
    run_vllm_benchmark(bench)
