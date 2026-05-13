"""
Benchmark tests for FP8 MQA (Multi-Query Attention) Logits operation.

This module provides performance benchmarks comparing FlagGems' Triton-based
FP8 MQA logits kernel against vLLM's DeepGEMM implementation on NVIDIA
Hopper architecture (SM90+).

The FP8 MQA logits operation computes weighted attention logits with FP8
quantization, commonly used in LLM inference for efficient attention computation.
"""

import random
from typing import Optional

import pytest
import torch

import flag_gems

from . import base

random.seed(42)


def is_vllm_available() -> bool:
    """Check if vLLM is available."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def is_hopper_available() -> bool:
    """Check if the current device is NVIDIA Hopper architecture (SM90+)."""
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


def has_deep_gemm() -> bool:
    """Check if vLLM's DeepGEMM is available."""
    try:
        from vllm.utils.import_utils import has_deep_gemm as check_deep_gemm

        return check_deep_gemm()
    except ImportError:
        return False


VLLM_AVAILABLE = is_vllm_available()
DEEPGEMM_AVAILABLE = has_deep_gemm()
HOPPER_AVAILABLE = is_hopper_available()


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to FP8 E4M3 format with proper clamping."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return tensor.clamp(min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)


def _build_case(
    M: int, H: int, D: int, N: int, q_dtype: torch.dtype, device: str
) -> tuple:
    """
    Build a test case for FP8 MQA logits.

    Args:
        M: Number of query sequences
        H: Number of attention heads
        D: Head dimension
        N: Number of key-value tokens
        q_dtype: Original query dtype (before FP8 conversion)
        device: Device to create tensors on

    Returns:
        Tuple of (q_fp8, k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke)
    """
    q = torch.randn((M, H, D), device=device, dtype=q_dtype)
    q_fp8 = to_fp8(q)

    k = torch.randn((N, D), device=device, dtype=torch.bfloat16)
    k_fp8 = to_fp8(k)

    k_scales = torch.rand((N,), device=device, dtype=torch.float32) * 0.01 + 0.001

    weights = torch.randn((M, H), device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.zeros((M,), device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.full((M,), N, device=device, dtype=torch.int32)

    return (q_fp8, k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke)


class FP8MQALogitsCompareBenchmark(base.Benchmark):
    """
    Benchmark comparing FlagGems vs vLLM DeepGEMM for FP8 MQA logits.

    Note: vLLM DeepGEMM requires N to be aligned to block_q (typically 128).
    We use shapes that satisfy this alignment requirement.
    """

    def set_shapes(self, shape_file_path: Optional[str] = None):
        """Define test shapes that satisfy vLLM DeepGEMM alignment requirements."""
        self.shapes = [
            # D=128 cases (most common)
            (32, 32, 128, 1024),
            (32, 32, 128, 2048),
            (32, 32, 128, 4096),
            (32, 32, 128, 8192),
            (64, 32, 128, 1024),
            (64, 32, 128, 2048),
            (64, 32, 128, 4096),
            (128, 32, 128, 1024),
            (128, 32, 128, 2048),
            # D=64 cases
            (32, 32, 64, 1024),
            (32, 32, 64, 2048),
            (64, 32, 64, 1024),
            # D=32 cases
            (32, 32, 32, 1024),
            (64, 32, 32, 2048),
            # Different H values
            (32, 64, 128, 1024),
            (32, 64, 128, 2048),
            (64, 64, 128, 4096),
        ]

    def get_input_iter(self, dtype):
        """Generate input iterator for the benchmark."""
        for M, H, D, N in self.shapes:
            case = _build_case(M, H, D, N, dtype, self.device)
            q_fp8, k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke = case
            yield (
                q_fp8,
                k_fp8,
                k_scales,
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                dtype,
            )

    def __init__(self):
        super().__init__(
            op_name="fp8_mqa_logits_gems_vs_deepgemm",
            torch_op=self._vllm_wrapper,
            dtypes=[torch.bfloat16, torch.float16],
        )
        self.set_gems(self._gems_wrapper)

    @staticmethod
    def _vllm_wrapper(
        q_fp8,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_dtype,
    ):
        """Wrapper for vLLM DeepGEMM FP8 MQA logits."""
        from vllm.utils.deep_gemm import fp8_mqa_logits as vllm_fp8_mqa_logits

        return vllm_fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scales),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
        )

    @staticmethod
    def _gems_wrapper(
        q_fp8,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_dtype,
    ):
        """Wrapper for FlagGems FP8 MQA logits."""
        from flag_gems.ops.fp8_mqa_logits import fp8_mqa_logits as gems_fp8_mqa_logits

        return gems_fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scales),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
        )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HOPPER_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not (VLLM_AVAILABLE and DEEPGEMM_AVAILABLE),
    reason="requires vLLM with DeepGEMM support",
)
@pytest.mark.fp8_mqa_logits
@pytest.mark.performance
def test_perf_fp8_mqa_logits_gems_vs_deepgemm():
    """
    Benchmark FlagGems FP8 MQA logits vs vLLM DeepGEMM implementation.

    This test compares the performance of FlagGems' Triton-based FP8 MQA logits
    kernel against vLLM's DeepGEMM implementation on NVIDIA Hopper GPUs.
    """
    bench = FP8MQALogitsCompareBenchmark()
    bench.run()
