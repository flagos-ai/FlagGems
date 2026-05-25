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
            dtypes=[torch.float16, torch.bfloat16],
        )
        self.set_gems(indexer_k_quant_and_cache)
        self.shape_desc = (
            "num_tokens, num_blocks, block_size, head_dim, quant_block_size"
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (128, 16, 16, 128, 128),
            (512, 64, 16, 128, 128),
            (1024, 128, 16, 512, 128),
            (2048, 256, 16, 512, 128),
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
