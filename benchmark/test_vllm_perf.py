import dataclasses
import random
from itertools import product
from math import ceil
from typing import List

import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark


def is_vllm_available():
    try:
        import vllm._custom_ops as ops  # noqa: F401

        return True
    except ImportError:
        return False


try:
    from vllm.utils.deep_gemm import fp8_paged_mqa_logits as vllm_fp8_paged_mqa_logits
    from vllm.utils.deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
        has_deep_gemm,
    )

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except ImportError:
    DEEPGEMM_AVAILABLE = False
    vllm_fp8_paged_mqa_logits = None
    get_num_sms = None
    get_paged_mqa_logits_metadata = None

VLLM_AVAILABLE = is_vllm_available()


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
DEFAULT_BLOCK_SHAPE = [128, 128]


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
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
def test_cutlass_scaled_mm_benchmark():
    bench = CutlassScaledMMBenchmark()
    bench.run()


# ---------------------- fused_moe op test ----------------------
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


class FusedMoEBenchmark(Benchmark):
    """
    Benchmark for fused_experts_impl comparing FlagGems Triton kernel vs vLLM.

    Measures latency of the full fused MoE pipeline:
      moe_align_block_size → GEMM1(up+gate) → SiLU+Mul → GEMM2(down) → moe_sum
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._fused_moe_input_fn(config, cur_dtype)

    def _fused_moe_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (hidden_states, w1, w2, topk_weights, topk_ids)


def _vllm_fused_moe_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call vllm fused_experts_impl."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )


def _gems_fused_moe_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call FlagGems fused_experts_impl."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_perf_fused_moe_gems_vs_vllm():
    """
    Benchmark FlagGems fused_experts_impl vs vLLM fused_experts_impl (bf16/fp16).
    """
    bench = FusedMoEBenchmark(
        op_name="fused_moe_gems_vs_vllm",
        torch_op=_vllm_fused_moe_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(_gems_fused_moe_wrapper)
    bench.run()


class FusedMoEFP8Benchmark(Benchmark):
    """
    Benchmark for fused_experts_impl with FP8 W8A8 quantization.

    Weights are pre-quantized to FP8 E4M3 with per-expert scales.
    Activations are dynamically quantized per-tensor inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._fp8_input_fn(config, cur_dtype)

    def _fp8_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device
        fp8_dtype = torch.float8_e4m3fn

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate FP8 weights one expert at a time to avoid OOM on large E.
        w1_fp8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=fp8_dtype,
        )
        w2_fp8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=fp8_dtype,
        )
        for e in range(num_experts):
            w1_fp8[e] = to_fp8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
            )
            w2_fp8[e] = to_fp8(
                torch.randn(
                    hidden_size, intermediate_size, device=device, dtype=torch.float16
                )
            )

        # Synthetic per-expert scales (representative of real quantization)
        w1_scale = (
            torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_fp8,
            w2_fp8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


class FusedMoEFP8BlockwiseBenchmark(Benchmark):
    """
    Benchmark for fused_experts_impl with FP8 W8A8 block-wise quantization.

    Weights are stored in FP8 E4M3 and accompanied by block scales.
    Activations are dynamically quantized per-token per-group inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = DEFAULT_BLOCK_SHAPE

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3.5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
        ]

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for config in self.shapes:
            yield from self._fp8_blockwise_input_fn(config)

    def _fp8_blockwise_input_fn(self, config):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        block_n, block_k = self.block_shape
        device = flag_gems.device
        dtype = torch.bfloat16

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1_fp8 = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * (1.0 / hidden_size**0.5)
        ).to(torch.float8_e4m3fn)
        w2_fp8 = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * (1.0 / intermediate_size**0.5)
        ).to(torch.float8_e4m3fn)

        w1_scale = (
            torch.rand(
                num_experts,
                ceil(intermediate_size * 2 / block_n),
                ceil(hidden_size / block_k),
                device=device,
                dtype=torch.float32,
            )
            + 0.01
        )
        w2_scale = (
            torch.rand(
                num_experts,
                ceil(hidden_size / block_n),
                ceil(intermediate_size / block_k),
                device=device,
                dtype=torch.float32,
            )
            + 0.01
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)

        yield (
            hidden_states,
            w1_fp8,
            w2_fp8,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )


def _vllm_fused_moe_fp8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call vllm fused_experts_impl with FP8."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_fp8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with FP8."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _vllm_fused_moe_fp8_blockwise_wrapper(
    hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
):
    """Wrapper to call vllm fused_experts_impl with block-wise FP8."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=DEFAULT_BLOCK_SHAPE,
    )


def _gems_fused_moe_fp8_blockwise_wrapper(
    hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
):
    """Wrapper to call FlagGems fused_experts_impl with block-wise FP8."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=DEFAULT_BLOCK_SHAPE,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(
    not (HAS_VLLM_FUSED_MOE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture for FP8",
)
def test_perf_fused_moe_fp8_gems_vs_vllm():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with FP8 W8A8 quantization.
    """
    bench = FusedMoEFP8Benchmark(
        op_name="fused_moe_fp8_gems_vs_vllm",
        torch_op=_vllm_fused_moe_fp8_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_fp8_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(
    not (HAS_VLLM_FUSED_MOE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture for FP8 blockwise",
)
def test_perf_fused_moe_fp8_blockwise_gems_vs_vllm():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with FP8 W8A8 block-wise quantization.
    """
    bench = FusedMoEFP8BlockwiseBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_vllm",
        torch_op=_vllm_fused_moe_fp8_blockwise_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_fp8_blockwise_wrapper)
    bench.run()


class FusedMoEINT8Benchmark(Benchmark):
    """
    Benchmark for fused_experts_impl with INT8 W8A8 quantization.

    Weights are pre-quantized to INT8 with per-channel (per output-dim) scales.
    Activations are dynamically quantized per-token inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int8_input_fn(config, cur_dtype)

    def _int8_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights one expert at a time to avoid OOM on large E.
        w1_int8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int8[e] = to_int8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )
            w2_int8[e] = to_int8(
                torch.randn(
                    hidden_size, intermediate_size, device=device, dtype=torch.float16
                )
                * 50
            )

        # Synthetic per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts, intermediate_size * 2, device=device, dtype=torch.float32
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int8,
            w2_int8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call vllm fused_experts_impl with INT8."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT8."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_perf_fused_moe_int8_gems_vs_vllm():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with INT8 W8A8 quantization.
    """
    bench = FusedMoEINT8Benchmark(
        op_name="fused_moe_int8_gems_vs_vllm",
        torch_op=_vllm_fused_moe_int8_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int8_wrapper)
    bench.run()


class FusedMoEINT8W8A16Benchmark(Benchmark):
    """
    Benchmark for fused_experts_impl with INT8 W8A16 weight-only quantization.

    Weights are pre-quantized to INT8 with per-channel scales.
    Activations remain in FP16/BF16 (no activation quantization).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int8_w8a16_input_fn(config, cur_dtype)

    def _int8_w8a16_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights one expert at a time to avoid OOM on large E.
        w1_int8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int8[e] = to_int8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )
            w2_int8[e] = to_int8(
                torch.randn(
                    hidden_size, intermediate_size, device=device, dtype=torch.float16
                )
                * 50
            )

        # Per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts, intermediate_size * 2, device=device, dtype=torch.float32
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int8,
            w2_int8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int8_w8a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Baseline: dequantize INT8 weights to bf16, then run FlagGems bf16
    fused_moe.  Measures the overhead of the dequant + bf16 path.

    NOTE: vLLM's INT8 W8A16 relies on specialised WNA16 kernels (CUDA or
    GPTQ/AWQ Triton) that are not directly comparable to the generic
    dequantize-then-GEMM approach, so we use a bf16 dequant baseline.
    """
    w1_deq = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
    w2_deq = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
    return flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
    )


def _gems_fused_moe_int8_w8a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT8 W8A16."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_int8_w8a16_gems_vs_vllm():
    """
    Benchmark FlagGems fused_experts_impl with INT8 W8A16 quantization.

    Baseline is manual dequant + bf16 FlagGems (vLLM's INT8 W8A16 uses
    specialised WNA16 kernels not available via the generic Triton path).
    """
    bench = FusedMoEINT8W8A16Benchmark(
        op_name="fused_moe_int8_w8a16_gems_vs_bf16_deq",
        torch_op=_vllm_fused_moe_int8_w8a16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int8_w8a16_wrapper)
    bench.run()


class FusedMoEINT4W4A16Benchmark(Benchmark):
    """
    Benchmark for fused_experts_impl with INT4 W4A16 weight-only quantization.

    Weights are pre-quantized to INT4 (stored in INT8 containers) with
    per-channel scales.  Activations remain in FP16/BF16.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int4_w4a16_input_fn(config, cur_dtype)

    def _int4_w4a16_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT4 weights (stored in INT8) one expert at a time.
        w1_int4 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int4 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int4[e] = torch.randint(
                -8,
                8,
                (intermediate_size * 2, hidden_size),
                device=device,
                dtype=torch.int8,
            )
            w2_int4[e] = torch.randint(
                -8,
                8,
                (hidden_size, intermediate_size),
                device=device,
                dtype=torch.int8,
            )

        # Per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts, intermediate_size * 2, device=device, dtype=torch.float32
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int4,
            w2_int4,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int4_w4a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper baseline: dequantize INT4 weights to bf16, then run FlagGems
    bf16 fused_moe.  This measures the overhead of the dequant + bf16 path so
    we can compare it against the dedicated INT4 dispatch path.

    NOTE: vLLM's INT4 W4A16 relies on a specialised WNA16 CUDA kernel that
    is not available via the generic Triton path, so we cannot use vLLM as
    baseline here.
    """
    # Dequantize to bf16 and run standard bf16 path as baseline
    w1_deq = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
    w2_deq = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
    return flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
    )


def _gems_fused_moe_int4_w4a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT4 W4A16."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int4_w4a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_int4_w4a16_gems_vs_vllm():
    """
    Benchmark FlagGems fused_experts_impl with INT4 W4A16 quantization.

    Baseline is manual dequant + bf16 FlagGems (vLLM's INT4 uses a
    specialised WNA16 CUDA kernel not available via the generic Triton path).
    """
    bench = FusedMoEINT4W4A16Benchmark(
        op_name="fused_moe_int4_w4a16_gems_vs_bf16_deq",
        torch_op=_vllm_fused_moe_int4_w4a16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int4_w4a16_wrapper)
    bench.run()


class FusedMoEMXQW8A16Benchmark(Benchmark):
    """
    Benchmark for flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.

    Uses QuantMode.W8A16: INT8 weights, FP16 activations.
    Tests SwiGLU MoE: y = W2 @ (silu(W1 @ x) * (W3 @ x))
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (smaller E to avoid OOM)
            (1, 64, 7168, 2048, 8),
            (4, 64, 7168, 2048, 8),
            (16, 64, 7168, 2048, 8),
            (64, 64, 7168, 2048, 8),
            (128, 64, 7168, 2048, 8),
            (256, 64, 7168, 2048, 8),
            # Qwen3.5-397B-A17B (smaller E to avoid OOM)
            (1, 128, 4096, 1024, 10),
            (4, 128, 4096, 1024, 10),
            (16, 128, 4096, 1024, 10),
            (64, 128, 4096, 1024, 10),
            (128, 128, 4096, 1024, 10),
            (256, 128, 4096, 1024, 10),
        ]

    def set_more_metrics(self):
        # Display both QC TFLOPS (gems latency) and FP16 ref TFLOPS (torch latency_base).
        return ["tflops", "tflops_base"]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._w8a16_mxq_input_fn(config)

    def _w8a16_mxq_input_fn(self, config):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device
        dtype = torch.bfloat16

        from flag_gems.fused_moe_mxq import QuantConfig, QuantMode, quantize_weights_moe

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights with scales (group-wise quantization)
        w1_fp16 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / hidden_size**0.5)
        w2_fp16 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / intermediate_size**0.5)
        w3_fp16 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / hidden_size**0.5)

        # Quantize to W8A16
        quant_config = QuantConfig(mode=QuantMode.W8A16, has_zero_point=False)
        w1_q, w1_scale, _ = quantize_weights_moe(w1_fp16, num_experts, quant_config)
        w2_q, w2_scale, _ = quantize_weights_moe(w2_fp16, num_experts, quant_config)
        w3_q, w3_scale, _ = quantize_weights_moe(w3_fp16, num_experts, quant_config)

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            # FP16/BF16 reference weights (pure fused_experts_impl path)
            w1_fp16,
            w2_fp16,
            # Pre-quantized weights for QC W8A16 (fused_moe_mxq path)
            w1_q,
            w2_q,
            w3_q,
            w1_scale,
            w2_scale,
            w3_scale,
            topk_weights,
            topk_ids,
            num_experts,
            topk,
        )

    def get_tflops(
        self,
        op,
        hidden_states,
        w1_fp16,
        w2_fp16,
        w1_q,
        w2_q,
        w3_q,
        w1_scale,
        w2_scale,
        w3_scale,
        topk_weights,
        topk_ids,
        num_experts,
        topk,
    ):
        """
        Proxy FLOPs estimate for SwiGLU MoE.

        This is an algorithmic FLOPs estimate (not hardware-instruction FLOPs).
        It is derived strictly from tensor shapes to avoid hard-coded constants.

        For each (token, expert) dispatch, we approximate:
          - W1 projection: (H) x (Nw1)  => 2 * H * Nw1
          - W3 projection: (H) x (Nw1)  => 2 * H * Nw1
          - W2 projection: (I) x (H)    => 2 * H * I

        Total FLOPs:
          num_tokens * topk * (2*H*Nw1 + 2*H*Nw1 + 2*H*I)
        """
        # hidden_states: [num_tokens, H]
        num_tokens = int(hidden_states.shape[0])
        hidden_size = int(hidden_states.shape[1])
        # w1_fp16: [E, Nw1, H] where Nw1 is typically 2*I (gated)
        n_w1 = int(w1_fp16.shape[1])
        # w2_fp16: [E, H, I]
        intermediate_size = int(w2_fp16.shape[2])
        topk = int(topk)
        per_dispatch_flops = (
            2.0 * hidden_size * n_w1
            + 2.0 * hidden_size * n_w1
            + 2.0 * hidden_size * intermediate_size
        )
        total_flops = num_tokens * topk * per_dispatch_flops
        return total_flops


def _baseline_w8a16_mxq_wrapper(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """FP16/BF16 reference: run flag_gems.fused_experts_impl pure FP16 path."""
    del w1_q, w2_q, w3_q, w1_scale, w2_scale, w3_scale, num_experts, topk
    return flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1_fp16,
        w2_fp16,
        topk_weights,
        topk_ids,
    )


def _baseline_w8a16_mxq_wrapper_vllm(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """Wrapper to call vllm fused_experts_impl with W8A16 quantized weights."""
    del w1_fp16, w2_fp16, w3_q, w3_scale, num_experts, topk
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_int8_w8a16=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_mxq_w8a16_wrapper(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """Test flag_gems.fused_moe_mxq.fused_moe with W8A16."""
    del w1_fp16, w2_fp16
    from flag_gems.fused_moe_mxq import QuantConfig, QuantMode, fused_moe

    quant_config = QuantConfig(mode=QuantMode.W8A16, has_zero_point=False)
    return fused_moe(
        hidden_states,
        w1=None,
        w2=None,
        w3=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
        num_experts=num_experts,
        top_k=topk,
        w1_q=w1_q,
        w1_scales=w1_scale,
        w1_zeros=None,
        w2_q=w2_q,
        w2_scales=w2_scale,
        w2_zeros=None,
        w3_q=w3_q,
        w3_scales=w3_scale,
        w3_zeros=None,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_w8a16_mxq():
    """
    Benchmark flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.
    """
    bench = FusedMoEMXQW8A16Benchmark(
        op_name="fused_moe_w8a16_mxq_gems_vs_bf16_deq",
        torch_op=_baseline_w8a16_mxq_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_mxq_w8a16_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_w8a16_mxq_gems_vs_vllm():
    """
    Benchmark flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.
    """
    bench = FusedMoEMXQW8A16Benchmark(
        op_name="fused_moe_w8a16_mxq_gems_vs_vllm",
        torch_op=_baseline_w8a16_mxq_wrapper_vllm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_mxq_w8a16_wrapper)
    bench.run()


try:
    from vllm.utils.deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
        has_deep_gemm,
    )

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except ImportError:
    DEEPGEMM_AVAILABLE = False


class GetPagedMqaLogitsMetadataBenchmark(Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (batch_size, next_n)
        self.shapes = [
            (4, 1),
            (8, 1),
            (16, 1),
            (32, 1),
            (64, 1),
            (128, 1),
            (256, 1),
            (512, 1),
            (4, 2),
            (8, 2),
            (16, 2),
            (32, 2),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._input_fn(config, cur_dtype)

    def _input_fn(self, config, dtype):
        batch_size, next_n = config
        device = flag_gems.device

        avg_ctx_len = 2048
        context_lens = torch.randint(
            int(0.8 * avg_ctx_len),
            int(1.2 * avg_ctx_len),
            (batch_size,),
            device=device,
            dtype=torch.int32,
        )

        if next_n > 1:
            context_lens = (
                context_lens.unsqueeze(1).expand(batch_size, next_n).contiguous()
            )

        num_sms = get_num_sms()

        yield (context_lens, 64, num_sms)


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.skipif(
    not DEEPGEMM_AVAILABLE,
    reason="requires vLLM with DeepGEMM and NVIDIA Hopper architecture or newer",
)
def test_get_paged_mqa_logits_metadata_benchmark():
    bench = GetPagedMqaLogitsMetadataBenchmark(
        op_name="get_paged_mqa_logits_metadata",
        torch_op=get_paged_mqa_logits_metadata,
        dtypes=[torch.int32],
    )
    bench.set_gems(flag_gems.get_paged_mqa_logits_metadata)
    bench.run()


def kv_cache_cast_to_fp8_deepgemm(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf_bytes
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def kv_cache_cast_to_fp8_triton(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size, num_heads, head_dim + 4),
        device=x.device,
        dtype=torch.uint8,
    )
    out[..., :head_dim] = x_scaled.view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    out[..., head_dim:] = sf_bytes.view(num_blocks, block_size, num_heads, 4)
    return out


def _build_case(
    batch_size, next_n, heads, head_dim, avg_kv, blocksize, q_dtype, max_model_len=4096
):
    num_blocks = max_model_len * 20

    q = torch.randn(
        (batch_size, next_n, heads, head_dim),
        device=flag_gems.device,
        dtype=q_dtype,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, head_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )

    weights = torch.randn(
        (batch_size * next_n, heads),
        device=flag_gems.device,
        dtype=torch.float32,
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv),
        int(1.2 * avg_kv),
        (batch_size,),
        device=flag_gems.device,
        dtype=torch.int32,
    )

    max_num_blocks_per_seq = (
        int(context_lens.max().item()) + blocksize - 1
    ) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        device=flag_gems.device,
        dtype=torch.int32,
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i, j] = block_idx_pool[counter]
            counter += 1

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8_deepgemm(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    return (
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )


class FP8PagedMQACompareBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            "fp8_paged_mqa_logits_gems_vs_deepgemm",
            self._vllm_wrapper,
            [torch.bfloat16],
        )
        self.set_gems(self._gems_wrapper)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, _dtype):
        compare_shapes = [
            (1, 1, 16, 64, 1024),
            (2, 1, 32, 128, 2048),
            (4, 1, 32, 128, 2048),
            (2, 2, 32, 128, 2048),
            (8, 1, 32, 128, 3072),
        ]
        q_dtypes = [torch.bfloat16, torch.float16]
        blocksize = 64

        for (bs, nn, h, d, avg_kv), q_dtype in product(compare_shapes, q_dtypes):
            case = _build_case(bs, nn, h, d, avg_kv, blocksize, q_dtype)
            (
                q_fp8,
                kv_dg,
                kv_tr,
                weights,
                context_lens,
                block_tables,
                max_model_len,
            ) = case
            schedule_metadata = get_paged_mqa_logits_metadata(
                context_lens, blocksize, get_num_sms()
            )
            yield (
                q_fp8,
                kv_dg,
                kv_tr,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
                q_dtype,
                blocksize,
            )

    @staticmethod
    def _vllm_wrapper(
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        q_dtype,
        blocksize,
    ):
        return vllm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_deepgemm,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=True,
        )

    @staticmethod
    def _gems_wrapper(
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        q_dtype,
        blocksize,
    ):
        return flag_gems.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and VLLM_AVAILABLE
        and DEEPGEMM_AVAILABLE
        and CUDA_AVAILABLE
    ),
    reason="requires CUDA + vLLM + Hopper",
)
@pytest.mark.performance
@pytest.mark.fp8_paged_mqa_logits
def test_perf_fp8_paged_mqa_logits_gems_vs_deepgemm():
    bench = FP8PagedMQACompareBenchmark()
    bench.run()


# ---------------------- flashmla_sparse op test ----------------------
try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )

    HAS_VLLM_FLASHMLA_SPARSE = True
except ImportError:
    HAS_VLLM_FLASHMLA_SPARSE = False


@dataclasses.dataclass
class Flashmla_Sparse_Test_Param:
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd", vllm_flash_mla_sparse_fwd, [torch.bfloat16]
        )
        self.set_gems(flag_gems.flash_mla_sparse_fwd)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, cur_dtype):
        for param in FlashmlaSparseBenchmark.get_performance_test_params_flashmla():
            yield from FlashmlaSparseBenchmark.make_input_flashmla(param)

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def get_performance_test_params_flashmla():
        cases = (
            [
                Flashmla_Sparse_Test_Param(
                    4096, s_kv, 2048, h_q=128, d_qk=576, have_attn_sink=True
                )
                for s_kv in [8192, 32768, 65536, 98304, 131072]
            ]
            + [
                Flashmla_Sparse_Test_Param(
                    4096, s_kv, 512, h_q=64, d_qk=512, have_attn_sink=True
                )
                for s_kv in [8192, 32768, 49152, 65536]
            ]
            + [
                Flashmla_Sparse_Test_Param(
                    4096, s_kv, 1024, h_q=128, d_qk=512, have_attn_sink=True
                )
                for s_kv in [8192, 32768, 49152, 65536]
            ]
        )
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]
    ) -> torch.Tensor:
        assert not torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        perm_range_max = max(int(torch.max(perm_range.item())), perm_size)
        rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
        rand[
            torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max)
            >= perm_range.view(batch_size, 1)
        ] = float("-inf")
        res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        if len(paddings) == 1:
            res[res >= perm_range.view(batch_size, 1)] = paddings[0]
        else:
            fillers = torch.tensor(paddings, dtype=torch.int32).index_select(
                0, torch.randint(0, len(paddings), (res.numel(),), dtype=torch.int32)
            )
            res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
        torch.use_deterministic_algorithms(False)
        return res

    @staticmethod
    def make_input_flashmla(param: Flashmla_Sparse_Test_Param):
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseBenchmark._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]
        indices = FlashmlaSparseBenchmark._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)
        yield (q, kv, indices, 0.5, param.d_v, attn_sink, topk_length)


@pytest.mark.flashmla_sparse
@pytest.mark.skipif(not HAS_VLLM_FLASHMLA_SPARSE, reason="vllm not installed")
def test_perf_flashmla_sparse_gems_vs_vllm():
    bench = FlashmlaSparseBenchmark()
    bench.run()
