import random
from itertools import product
from math import ceil

import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark

random.seed(42)


def is_vllm_available():
    try:
        import vllm._custom_ops as ops  # noqa: F401

        return True
    except ImportError:
        return False


VLLM_AVAILABLE = is_vllm_available()


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


HOPPER_AVAILABLE = is_hopper_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


class CutlassScaledMMPerfKit:
    num_perf_cases = 4
    scalar_only_params = []
    vector_only_params = []
    scalar_and_vector_params = []
    block_params = []

    @staticmethod
    def _get_all_combinations():
        # these shapes come from the test file of op `cutlass_scaled_mm` of vLLM
        mnk = [
            (1, 256, 128),
            (1, 16384, 1024),
            (1, 24576, 496),
            (16, 256, 496),
            (16, 16384, 128),
            (16, 24576, 4096),
            (32, 8192, 4096),
            (32, 16384, 4096),
            (33, 1024, 1024),
            (33, 8192, 128),
            (64, 2048, 496),
            (64, 16384, 1024),
            (100, 8192, 496),
            (128, 32768, 4096),
            (256, 4096, 4096),
            (512, 256, 1024),
            (512, 8192, 4096),
            (512, 16384, 128),
            (512, 24576, 128),
        ]
        scale_shape_types = ["scalar", "vector", "matrix"]
        if_use_bias = [True, False]
        dtypes = [(torch.int8, torch.float16), (torch.float8_e4m3fn, torch.bfloat16)]

        combinations = product(
            mnk, scale_shape_types, scale_shape_types, if_use_bias, dtypes
        )
        return combinations

    @classmethod
    def _rand_sample(cls, all_params):
        random.shuffle(all_params)
        count = [0] * 4
        for param in all_params:
            a_scale_category = param["a_scale_category"]
            b_scale_category = param["b_scale_category"]
            if a_scale_category == "matrix" and count[0] < cls.num_perf_cases:
                count[0] += 1
                cls.block_params.append(param)
            elif (
                a_scale_category == "scalar"
                and b_scale_category == "scalar"
                and count[1] < cls.num_perf_cases
            ):
                count[1] += 1
                cls.scalar_only_params.append(param)
            elif (
                a_scale_category == "vector"
                and b_scale_category == "vector"
                and count[2] < cls.num_perf_cases
            ):
                count[2] += 1
                cls.vector_only_params.append(param)
            elif count[3] < cls.num_perf_cases:
                count[3] += 1
                cls.scalar_and_vector_params.append(param)
            else:
                continue

    @classmethod
    def init_perf_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            use_bias,
            (in_dtype, out_dtype),
        ) in combinations:
            is_scalar_or_vector_dequant = a_scale_category in [
                "scalar",
                "vector",
            ] and b_scale_category in ["scalar", "vector"]
            is_block_dequant = (
                a_scale_category == "matrix" and b_scale_category == "matrix"
            )

            if not (is_scalar_or_vector_dequant or is_block_dequant):
                continue

            if is_block_dequant and (use_bias or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": use_bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        elif category == "vector":
            if is_a_scale:
                return (M,)
            else:
                return (N,)
        else:
            if is_a_scale:
                return (M, ceil(K / 128))
            else:
                return (ceil(K / 128), ceil(N / 128))


class CutlassScaledMMBenchmark(Benchmark):
    def __init__(self):
        extended_dtypes = ["scalar_only", "vector_only", "scalar_and_vector", "block"]
        super().__init__(
            "cutlass_scaled_mm", torch.ops._C.cutlass_scaled_mm, extended_dtypes
        )
        self.set_gems(flag_gems.cutlass_scaled_mm)
        self.kit = CutlassScaledMMPerfKit
        self.kit.init_perf_params()

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        params = getattr(self.kit, f"{dtype}_params")

        for p in params:
            M, N, K = p["M"], p["N"], p["K"]
            in_dtype = p["in_dtype"]
            out_dtype = p["out_dtype"]
            a_scale_category = p["a_scale_category"]
            b_scale_category = p["b_scale_category"]

            if in_dtype == torch.int8:
                a = to_int8(torch.randn((M, K), device=flag_gems.device))
                b = to_int8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                    * 5
                )
            else:
                a = to_fp8(torch.randn((M, K), device=flag_gems.device))
                b = to_fp8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                )

            a_scale_shape = self.kit.get_scale_shape(M, N, K, a_scale_category)
            b_scale_shape = self.kit.get_scale_shape(M, N, K, b_scale_category, False)

            scale_a = torch.randn(
                a_scale_shape, device=flag_gems.device, dtype=torch.float32
            )
            scale_b = torch.randn(
                b_scale_shape, device=flag_gems.device, dtype=torch.float32
            )

            scale_a = scale_a.contiguous()
            # convert scale_b to col-major
            # (for scalar/vector scale_b, this's a identical transformation)
            scale_b = scale_b.t().contiguous().t()

            bias = None
            if p["use_bias"]:
                bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

            c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

            yield (c, a, b, scale_a, scale_b, bias)


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and HOPPER_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.performance
def test_cutlass_scaled_mm_benchmark():
    bench = CutlassScaledMMBenchmark()
    bench.run()


# ============ cp_gather_indexer_k_quant_cache benchmark ============
class CpGatherIndexerKQuantCacheBenchmark(Benchmark):
    """Benchmark for cp_gather_indexer_k_quant_cache operator.

    Note: vLLM's C++ kernel requires head_dim % 128 == 0.
    FlagGems implementation is more flexible.
    This benchmark tests FlagGems implementation only (latency measurement).
    """

    DEFAULT_METRICS = ["latency"]

    def __init__(self):
        from flag_gems.fused.cp_gather_indexer_k_quant_cache import (
            cp_gather_indexer_k_quant_cache,
        )

        super().__init__(
            "cp_gather_indexer_k_quant_cache",
            cp_gather_indexer_k_quant_cache,  # Use gems op as torch_op for latency
            dtypes=["fp8"],
        )

    def set_shapes(self, shape_file_path=None):
        # (batch_size, max_seq_len, head_dim, block_size)
        self.shapes = [
            (1, 128, 64, 16),
            (2, 256, 64, 16),
            (4, 512, 128, 32),
            (8, 256, 64, 16),
            (16, 128, 64, 32),
            (32, 64, 128, 16),
            (4, 1024, 64, 64),
            (8, 512, 128, 32),
            (16, 1024, 128, 32),
            (32, 512, 128, 64),
        ]

    def get_input_iter(self, dtype):
        for batch_size, max_seq_len, head_dim, block_size in self.shapes:
            device = flag_gems.device

            # Generate deterministic sequence lengths (must be divisible by 4 for fp8->fp32 view)
            # Use max_seq_len for all batches to ensure num_tokens % 4 == 0
            seq_lens = torch.full(
                (batch_size,),
                max_seq_len,
                device=device,
                dtype=torch.int32,
            )

            # Create cumulative sequence lengths
            cu_seq_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
            cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
            num_tokens = int(cu_seq_lens[-1].item())
            # Ensure num_tokens is divisible by 4
            num_tokens = (num_tokens // 4) * 4
            if num_tokens < 4:
                num_tokens = 4
            cu_seq_lens[-1] = num_tokens

            # Calculate number of blocks needed
            num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
            total_blocks = batch_size * num_blocks_per_seq

            # cache_stride = head_dim + 4 (4 bytes for float32 scale per token)
            cache_stride = head_dim + 4

            # Create kv_cache: [num_blocks, block_size, cache_stride]
            kv_cache = torch.zeros(
                total_blocks,
                block_size,
                cache_stride,
                dtype=torch.float8_e4m3fn,
                device=device,
            )

            # Fill with random data
            for b in range(batch_size):
                seq_len = int(seq_lens[b].item())
                for pos in range(seq_len):
                    block_idx = b * num_blocks_per_seq + pos // block_size
                    block_offset = pos % block_size
                    kv_cache[block_idx, block_offset, :head_dim] = to_fp8(
                        torch.randn(head_dim, device=device) * 0.1
                    )
                    scale_val = torch.tensor(
                        [0.5 + pos * 0.01], dtype=torch.float32, device=device
                    )
                    kv_cache[block_idx, block_offset, head_dim:] = scale_val.view(
                        torch.float8_e4m3fn
                    )

            # Create block_table: [batch_size, num_blocks_per_seq]
            block_table = torch.arange(
                total_blocks, device=device, dtype=torch.int32
            ).view(batch_size, num_blocks_per_seq)

            # Create output tensors
            dst_k = torch.empty(
                num_tokens, head_dim, dtype=torch.float8_e4m3fn, device=device
            )
            dst_scale = torch.empty(
                num_tokens, 1, dtype=torch.float8_e4m3fn, device=device
            )

            yield (kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.performance
def test_cp_gather_indexer_k_quant_cache_benchmark():
    bench = CpGatherIndexerKQuantCacheBenchmark()
    bench.run()


# ============ top_k_per_row_decode benchmark ============
class TopKPerRowDecodeBenchmark(Benchmark):
    """Benchmark for top_k_per_row_decode operator.

    Compares FlagGems Triton implementation with vLLM CUDA implementation.
    Uses histogram-based radix select O(n) algorithm.
    """

    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def __init__(self):
        from flag_gems.fused.top_k_per_row_decode import top_k_per_row_decode

        # vLLM reference function
        # Must import vllm._C first to register torch.ops._C.top_k_per_row_decode
        import vllm._C  # noqa: F401

        def vllm_top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        ):
            torch.ops._C.top_k_per_row_decode(
                logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
            )

        super().__init__(
            "top_k_per_row_decode",
            vllm_top_k_per_row_decode,
            dtypes=["small_rows", "medium_rows", "large_rows"],
        )
        self.set_gems(top_k_per_row_decode)

    def set_shapes(self, shape_file_path=None):
        # (num_rows, vocab_size, stride, top_k)
        self.shapes = {
            "small_rows": [
                (32, 20000, 1, 10),
                (32, 128256, 1, 10),
            ],
            "medium_rows": [
                (128, 20000, 1, 10),
                (128, 128256, 1, 10),
            ],
            "large_rows": [
                (1024, 20000, 1, 10),
                (1024, 128256, 1, 10),
                (2048, 20000, 1, 10),
                (2048, 128256, 1, 10),
                (4096, 20000, 1, 10),
                (4096, 128256, 1, 10),
                (8192, 20000, 1, 10),
                (8192, 128256, 1, 10),
            ],
        }

    def get_input_iter(self, dtype):
        shapes = self.shapes.get(dtype, [])
        for num_rows, vocab_size, stride, top_k in shapes:
            device = flag_gems.device
            batch_size = num_rows
            next_n = 1

            # Create logits tensor
            logits = torch.randn(
                num_rows, vocab_size, dtype=torch.float32, device=device
            )

            # Create seq_lens - each sequence has vocab_size valid elements
            seq_lens = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=device
            )

            # Create output indices tensor
            indices = torch.empty(num_rows, top_k, dtype=torch.int32, device=device)

            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

            yield (logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="requires vLLM")
@pytest.mark.top_k_per_row_decode
@pytest.mark.performance
def test_top_k_per_row_decode_benchmark():
    bench = TopKPerRowDecodeBenchmark()
    bench.run()
