import math
import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

from flag_gems.fused import indexer_k_quant_and_cache

from . import base

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")
_SHAPE_KEYS = (
    "case",
    "num_tokens",
    "num_blocks",
    "block_size",
    "head_dim",
    "quant_block_size",
    "cache_stride",
    "scale_fmt",
)


def is_fp8e4nv_supported():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


def run_vllm_benchmark(bench):
    original_str = base.BenchmarkResult.__str__

    def vllm_str(result):
        formatted = format_indexer_benchmark_result(result)
        if formatted is not None:
            return formatted
        return original_str(result).replace(
            "Torch Latency (ms)", "vLLM CUDA Latency (ms)"
        ).replace("Torch GBPS ", "vLLM CUDA GBPS ")

    base.BenchmarkResult.__str__ = vllm_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


def format_indexer_benchmark_result(result):
    if not result.result:
        return None

    shape_details = [metric.shape_detail for metric in result.result]
    if not all(isinstance(shape_detail, dict) for shape_detail in shape_details):
        return None

    fixed_items = []
    variable_keys = []
    for key in _SHAPE_KEYS:
        values = [shape_detail.get(key) for shape_detail in shape_details]
        if any(value is None for value in values):
            continue
        if all(value == values[0] for value in values):
            fixed_items.append((key, values[0]))
        else:
            variable_keys.append(key)

    fixed_desc = ", ".join(f"{key}={value}" for key, value in fixed_items)
    title = (
        f"\nOperator: {result.op_name}  Performance Test "
        f"(dtype={result.dtype}, mode={result.mode}, level={result.level})\n"
    )
    if fixed_desc:
        title += f"Fixed: {fixed_desc}\n"

    columns = [
        ("vLLM CUDA Latency (ms)", 24),
        ("Gems Latency (ms)", 20),
        ("Gems Speedup", 16),
    ]
    columns.extend((key, max(len(key) + 2, 14)) for key in variable_keys)
    header = "".join(f"{name:>{width}}" for name, width in columns) + "\n"
    lines = [title, header, "-" * len(header.rstrip()) + "\n"]

    for metric in result.result:
        latency_base = (
            f"{metric.latency_base:.6f}" if metric.latency_base is not None else "N/A"
        )
        latency = f"{metric.latency:.6f}" if metric.latency is not None else "N/A"
        speedup = f"{metric.speedup:.3f}" if metric.speedup is not None else "N/A"
        line = (
            f"{latency_base:>{columns[0][1]}}"
            f"{latency:>{columns[1][1]}}"
            f"{speedup:>{columns[2][1]}}"
        )
        for key, width in columns[3:]:
            line += f"{metric.shape_detail[key]:>{width}}"
        lines.append(line + "\n")

    return "".join(lines)


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
            "case, num_tokens, num_blocks, block_size, head_dim, quant_block_size"
        )
        self._shape_detail_queue = []

    def set_shapes(self, shape_file_path=None):
        head_dim = 512
        quant_block_size = 128
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

        block_size = 16
        packed_shapes = [
            (
                "packed",
                num_tokens,
                math.ceil(num_tokens / block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in token_sweep
        ]
        decode_shapes = [
            (
                "decode",
                num_tokens,
                num_tokens,
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (2, 4, 8, 16, 32, 64, 128, 256)
        ]

        block_size = 64
        block64_shapes = [
            (
                "packed",
                num_tokens,
                math.ceil(num_tokens / block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (8192, 32768, 65536)
        ]
        block64_decode_shapes = [
            (
                "decode",
                num_tokens,
                num_tokens,
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (16, 64, 128, 256)
        ]
        self.shapes = (
            packed_shapes + decode_shapes + block64_shapes + block64_decode_shapes
        )

    def record_shapes(self, *args, **kwargs):
        if self._shape_detail_queue:
            return self._shape_detail_queue.pop(0)
        return super().record_shapes(*args, **kwargs)

    def make_slot_mapping(self, case, num_tokens, block_size, device):
        if case == "decode":
            return torch.arange(
                num_tokens,
                dtype=torch.long,
                device=device,
            ) * block_size
        return torch.arange(num_tokens, dtype=torch.long, device=device)

    def get_input_iter(self, dtype):
        self._shape_detail_queue = []
        for (
            case,
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
            slot_mapping = self.make_slot_mapping(
                case,
                num_tokens,
                block_size,
                device=self.device,
            )
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            kv_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            self._shape_detail_queue.append(
                {
                    "case": case,
                    "num_tokens": num_tokens,
                    "num_blocks": num_blocks,
                    "block_size": block_size,
                    "head_dim": head_dim,
                    "quant_block_size": quant_block_size,
                    "cache_stride": cache_stride,
                    "scale_fmt": "ue8m0",
                }
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
