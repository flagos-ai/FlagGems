import pytest
import torch

from flag_gems.fused.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    deepseek_v4_fp8_einsum,
    fused_q_kv_rmsnorm,
    persistent_topk,
    top_k_per_row_prefill,
)

from . import base


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _has_op(lib_name: str, op_name: str) -> bool:
    try:
        return hasattr(getattr(torch.ops, lib_name), op_name)
    except Exception:
        return False


class FusedQKVRMSNormBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_q_kv_rmsnorm",
            fused_q_kv_rmsnorm,
            [torch.bfloat16],
            gems_op=fused_q_kv_rmsnorm,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(32,)]

    def get_input_iter(self, dtype):
        device = "cuda"
        for (tokens,) in self.shapes:
            qr = torch.randn((tokens, 64 * 576), device=device, dtype=dtype)
            kv = torch.randn((tokens, 576), device=device, dtype=dtype)
            q_weight = torch.randn((64 * 576,), device=device, dtype=dtype)
            kv_weight = torch.randn((576,), device=device, dtype=dtype)
            yield (qr, kv, q_weight, kv_weight, 1e-6)


class ComputeGlobalTopkIndicesAndLensBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "compute_global_topk_indices_and_lens",
            compute_global_topk_indices_and_lens,
            [torch.int32],
            gems_op=compute_global_topk_indices_and_lens,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(4096, 128)]

    def get_input_iter(self, dtype):
        _ = dtype
        device = "cuda"
        for num_tokens, topk in self.shapes:
            topk_indices = torch.randint(
                -1, 64, (num_tokens, topk), device=device, dtype=torch.int32
            )
            token_to_req_indices = torch.zeros(
                (num_tokens,), device=device, dtype=torch.int32
            )
            block_table = torch.arange(0, 256, device=device, dtype=torch.int32).view(
                1, -1
            )
            yield (topk_indices, token_to_req_indices, block_table, 64, None)


class CombineTopkSwaIndicesBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices",
            combine_topk_swa_indices,
            [torch.int32],
            gems_op=combine_topk_swa_indices,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(4096, 128)]

    def get_input_iter(self, dtype):
        _ = dtype
        device = "cuda"
        for num_tokens, topk in self.shapes:
            topk_indices = torch.randint(
                -1, 2048, (num_tokens, topk), device=device, dtype=torch.int32
            )
            query_start_loc = torch.arange(
                0, num_tokens + 1, device=device, dtype=torch.int32
            )
            seq_lens = torch.tensor([4096], device=device, dtype=torch.int32)
            gather_lens = torch.tensor([2048], device=device, dtype=torch.int32)
            yield (
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                256,
                8,
                topk,
                8192,
                4096,
            )


class DeepseekV4Fp8EinsumBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "deepseek_v4_fp8_einsum",
            deepseek_v4_fp8_einsum,
            [torch.float8_e4m3fn],
            gems_op=deepseek_v4_fp8_einsum,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(16, 64, 128, 128)]

    def get_input_iter(self, dtype):
        device = "cuda"
        for batch, groups, kdim, ndim in self.shapes:
            a = torch.randn(
                (batch, groups, kdim), device=device, dtype=torch.float32
            ).to(dtype)
            b = torch.randn(
                (groups, ndim, kdim), device=device, dtype=torch.float32
            ).to(dtype)
            a_scale = torch.ones(
                (batch, groups, kdim // 128), device=device, dtype=torch.float32
            )
            b_scale = torch.ones(
                (groups, ndim // 128, kdim // 128), device=device, dtype=torch.float32
            )
            out = torch.empty(
                (batch, groups, ndim), device=device, dtype=torch.bfloat16
            )
            yield (
                a,
                a_scale,
                b,
                b_scale,
                out,
                {"equation": "bhr,hdr->bhd", "recipe": [1, 128, 128]},
            )


class PersistentTopkBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "persistent_topk",
            persistent_topk,
            [torch.float32],
            gems_op=persistent_topk,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(16, 512)]

    def get_input_iter(self, dtype):
        device = "cuda"
        for rows, max_seq_len in self.shapes:
            logits = torch.randn((rows, max_seq_len), device=device, dtype=dtype)
            lengths = torch.full((rows,), max_seq_len, device=device, dtype=torch.int32)
            output = torch.empty((rows, 512), device=device, dtype=torch.int32)
            workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
            yield (logits, lengths, output, workspace, 512, max_seq_len)


class TopKPerRowPrefillBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "top_k_per_row_prefill",
            top_k_per_row_prefill,
            [torch.float32],
            gems_op=top_k_per_row_prefill,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(256, 128, 32)]

    def get_input_iter(self, dtype):
        device = "cuda"
        for num_rows, row_len, topk in self.shapes:
            logits = torch.randn((num_rows, row_len), device=device, dtype=dtype)
            row_starts = torch.arange(
                0, num_rows * row_len, row_len, device=device, dtype=torch.int32
            )
            row_ends = row_starts + row_len
            out = torch.empty((num_rows, topk), device=device, dtype=torch.int32)
            yield (
                logits,
                row_starts,
                row_ends,
                out,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk,
            )


@pytest.mark.skipif(not _has_hopper(), reason="requires Hopper SM90")
def test_deepseek_v4_ops_core_benchmarks():
    FusedQKVRMSNormBenchmark().run()
    ComputeGlobalTopkIndicesAndLensBenchmark().run()
    CombineTopkSwaIndicesBenchmark().run()
    DeepseekV4Fp8EinsumBenchmark().run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_deepseek_v4_ops_external_topk_benchmarks():
    if _has_op("_C", "persistent_topk"):
        PersistentTopkBenchmark().run()
    if _has_op("_C", "top_k_per_row_prefill"):
        TopKPerRowPrefillBenchmark().run()
