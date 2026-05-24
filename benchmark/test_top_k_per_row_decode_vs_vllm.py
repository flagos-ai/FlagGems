import pytest
import torch

from flag_gems.fused.sparse_attn_indexer.top_k_per_row_decode import (
    top_k_per_row_decode as flaggems_top_k_per_row_decode,
)

from . import base


def run_vllm_benchmark(bench):
    original_str = base.BenchmarkResult.__str__

    def vllm_str(result):
        return (
            original_str(result)
            .replace("Torch Latency (ms)", "vLLM Latency (ms)")
            .replace("Torch GBPS ", "vLLM GBPS ")
        )

    base.BenchmarkResult.__str__ = vllm_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


def load_vllm_topk_decode():
    try:
        import vllm._C

        torch.ops.load_library(vllm._C.__file__)
        return torch.ops._C.top_k_per_row_decode
    except Exception as exc:
        pytest.skip(f"vLLM top_k_per_row_decode is unavailable: {exc}")


def make_vllm_topk_wrapper(vllm_op):
    def wrapper(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    ):
        del row_starts
        return vllm_op(
            logits,
            1,
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
        )

    return wrapper


class TopKPerRowDecodeBenchmark(base.Benchmark):
    def __init__(self, vllm_op):
        super().__init__(
            op_name="top_k_per_row_decode_vs_vllm",
            torch_op=make_vllm_topk_wrapper(vllm_op),
            dtypes=[torch.float32],
        )
        self.set_gems(flaggems_top_k_per_row_decode)
        self.shape_desc = "num_rows, vocab_size, top_k"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4096, 64),
            (1, 8192, 128),
            (1, 16384, 256),
            (1, 32768, 512),
            (1, 129280, 1024),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, top_k in self.shapes:
            logits = torch.randn(
                num_rows,
                vocab_size,
                dtype=dtype,
                device=self.device,
            )
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device=self.device)
            row_ends = torch.full(
                (num_rows,),
                vocab_size,
                dtype=torch.int32,
                device=self.device,
            )
            indices = torch.empty(
                (num_rows, top_k),
                dtype=torch.int32,
                device=self.device,
            )
            yield (
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                top_k,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.top_k_per_row_decode
def test_top_k_per_row_decode_vs_vllm_benchmark():
    vllm_op = load_vllm_topk_decode()
    bench = TopKPerRowDecodeBenchmark(vllm_op)
    run_vllm_benchmark(bench)
