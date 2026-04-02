import random
from itertools import product
from math import ceil

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


VLLM_AVAILABLE = is_vllm_available()


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


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
@pytest.mark.performance
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
