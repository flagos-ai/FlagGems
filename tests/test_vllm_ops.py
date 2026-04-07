import dataclasses
import random
from itertools import product
from math import ceil
from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

random.seed(42)


try:
    import vllm  # noqa: 401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


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


class CutlassScaledMMTestKit:
    num_test_cases = 16 if QUICK_MODE else 32

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
        return all_params[: cls.num_test_cases]

    @classmethod
    def get_test_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            bias,
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

            if is_block_dequant and (bias is not None or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        return cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        if category == "vector":
            if is_a_scale:
                return (M,)
            return (N,)
        # a matrix
        if is_a_scale:
            return (M, ceil(K / 128))
        return (ceil(K / 128), ceil(N / 128))

    @staticmethod
    def baseline_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ):
        def group_broadcast(t: torch.Tensor, shape):
            for i, s in enumerate(shape):
                if t.shape[i] != s and t.shape[i] != 1:
                    assert s % t.shape[i] == 0
                    t = (
                        t.unsqueeze(i + 1)
                        .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                        .flatten(i, i + 1)
                    )
            return t

        scale_a_full = group_broadcast(scale_a, a.shape)
        scale_b_full = group_broadcast(scale_b, b.shape)

        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)

        lhs = scale_a_full * a_f32
        rhs = scale_b_full * b_f32

        output = torch.mm(lhs, rhs).to(out_dtype)

        if bias is not None:
            output = output + bias

        return output


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.parametrize("p", CutlassScaledMMTestKit.get_test_params())
def test_cutlass_scaled_mm(p):
    kit = CutlassScaledMMTestKit

    M, N, K = p["M"], p["N"], p["K"]
    in_dtype = p["in_dtype"]
    out_dtype = p["out_dtype"]
    a_scale_category = p["a_scale_category"]
    b_scale_category = p["b_scale_category"]

    if in_dtype == torch.int8:
        a = to_int8(torch.randn((M, K), device=flag_gems.device))
        b = to_int8(
            torch.randn((K, N), device=flag_gems.device).t().contiguous().t() * 5
        )
    else:
        a = to_fp8(torch.randn((M, K), device=flag_gems.device))
        b = to_fp8(torch.randn((K, N), device=flag_gems.device).t().contiguous().t())

    a_scale_shape = kit.get_scale_shape(M, N, K, a_scale_category)
    b_scale_shape = kit.get_scale_shape(M, N, K, b_scale_category, False)

    scale_a = torch.randn(a_scale_shape, device=flag_gems.device, dtype=torch.float32)
    scale_b = torch.randn(b_scale_shape, device=flag_gems.device, dtype=torch.float32)

    scale_a = scale_a.contiguous()
    # convert scale_b to col-major
    # (for scalar/vector scale_b, this's a identical transformation)
    scale_b = scale_b.t().contiguous().t()

    bias = None
    if p["use_bias"]:
        bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

    c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

    flag_gems.cutlass_scaled_mm(c, a, b, scale_a, scale_b, bias)

    output_ref = kit.baseline_scaled_mm(
        a, b, scale_a.view(-1, 1), scale_b.view(1, -1), out_dtype, bias
    )

    if in_dtype == torch.int8:
        rtol, atol = 1e-1, 1.0
    else:
        rtol, atol = 5e-1, 1.5e-1

    torch.testing.assert_close(c, output_ref, rtol=rtol, atol=atol)


# ---------------------- fused_moe op test ----------------------
FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
    # Qwen3.5 shapes (TP=4)
    (10, 256, 2048, 128, 8),
    (256, 256, 2048, 128, 8),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
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


def torch_fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of fused MoE (no vLLM dependency).

    Computes:
        Y_m = sum_j  A_mj * W2[e_mj] @ SiLU(W1[e_mj] @ H_m)_{:D} ) * (W1[e_mj] @ H_m)_{D:})

    Args:
        hidden_states: (M, K)
        w1: (E, 2D, K)  -- gate + up projection concatenated
        w2: (E, K, D)   -- down projection
        topk_weights: (M, topk)
        topk_ids: (M, topk)

    Returns:
        output: (M, K)
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # GEMM1: up-projection  (1, K) @ (K, 2D) -> (1, 2D)
            z = hidden_states[m].to(torch.float32) @ w1[e].T.to(torch.float32)
            # SiLU-and-Mul: split into gate and up, apply SwiGLU
            D = z.shape[-1] // 2
            gate = z[:D]
            up = z[D:]
            s = (gate * torch.sigmoid(gate)) * up  # SiLU(gate) * up
            # GEMM2: down-projection  (1, D) @ (D, K) -> (1, K)
            r = s @ w2[e].T.to(torch.float32)
            # Weighted accumulation
            output[m] += (weight.to(torch.float32) * r).to(output.dtype)

    return output


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_accuracy_fused_moe_vs_ref(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Pure PyTorch reference (no vLLM dependency)
    ref = torch_fused_moe_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize() if flag_gems.vendor_name != "ascend" else torch.npu.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_accuracy_fused_moe_vs_vllm(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Reference result
    ref = vllm_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
    )

    torch.cuda.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


FUSED_MOE_QUANT_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_QUANT_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
    ]

FUSED_MOE_FP8_BLOCKWISE_CONFIGS = list(FUSED_MOE_QUANT_CONFIGS)

if not QUICK_MODE:
    FUSED_MOE_FP8_BLOCKWISE_CONFIGS += [
        # Qwen3.5-397B-A17B
        (1, 512, 4096, 1024, 10),
        (4, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        (128, 512, 4096, 1024, 10),
        (256, 512, 4096, 1024, 10),
    ]


FUSED_MOE_BLOCK_SHAPES = [None, [128, 128], [64, 128], [128, 64]]


def _fake_quantize_fp8(tensor: torch.Tensor):
    """Simulate FP8 E4M3 quantization round-trip for reference computation."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10
    # Per-tensor quantization
    amax = tensor.abs().amax().clamp(min=eps).float()
    scale = amax / fp8_max
    q = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return q.float() * scale  # dequantized


def _fake_quantize_int8(tensor: torch.Tensor):
    """Simulate INT8 quantization round-trip for reference computation."""
    eps = 1e-10
    # Per-token quantization
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=eps).float()
    scale = amax / 127.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q.float() * scale  # dequantized


def _fake_quantize_fp8_block(tensor: torch.Tensor, group_size: int):
    """Simulate FP8 group-wise quantization on the last dimension."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10
    original_shape = tensor.shape
    assert original_shape[-1] % group_size == 0
    grouped = tensor.float().reshape(-1, original_shape[-1] // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / fp8_max
    q = (grouped / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape(original_shape)


def _fake_quantize_int8_block(tensor: torch.Tensor, group_size: int):
    """Simulate INT8 group-wise quantization on the last dimension."""
    eps = 1e-10
    original_shape = tensor.shape
    assert original_shape[-1] % group_size == 0
    grouped = tensor.float().reshape(-1, original_shape[-1] // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / 127.0
    q = (grouped / scale).clamp(-128, 127).to(torch.int8)
    return (q.float() * scale).reshape(original_shape)


def _quantize_fp8_blockwise(
    tensor: torch.Tensor,
    block_n: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights block-wise and return quantized, scales, dequantized tensors."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10

    num_experts, n_dim, k_dim = tensor.shape
    assert n_dim % block_n == 0
    assert k_dim % block_k == 0

    grouped = tensor.reshape(
        num_experts,
        n_dim // block_n,
        block_n,
        k_dim // block_k,
        block_k,
    ).permute(0, 1, 3, 2, 4)
    amax = grouped.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=eps)
    scale = amax / fp8_max
    q = (grouped / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    deq = q.float() * scale

    return (
        q.permute(0, 1, 3, 2, 4).reshape_as(tensor),
        scale.squeeze(-1).squeeze(-1),
        deq.permute(0, 1, 3, 2, 4).reshape_as(tensor),
    )


def _quantize_int8_blockwise(
    tensor: torch.Tensor,
    block_n: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights block-wise and return quantized, scales, dequantized tensors."""
    eps = 1e-10

    num_experts, n_dim, k_dim = tensor.shape
    assert n_dim % block_n == 0
    assert k_dim % block_k == 0

    grouped = tensor.reshape(
        num_experts,
        n_dim // block_n,
        block_n,
        k_dim // block_k,
        block_k,
    ).permute(0, 1, 3, 2, 4)
    amax = grouped.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=eps)
    scale = amax / 127.0
    q = (grouped / scale).round().clamp(-128, 127).to(torch.int8)
    deq = q.float() * scale

    return (
        q.permute(0, 1, 3, 2, 4).reshape_as(tensor),
        scale.squeeze(-1).squeeze(-1),
        deq.permute(0, 1, 3, 2, 4).reshape_as(tensor),
    )


def torch_fused_moe_quantized_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_mode: str = "fp8",
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    """Reference fused MoE with simulated quantization noise.

    Simulates the quantization → dequantization round-trip on activations
    to model the same numerical behavior as the quantized kernel path.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    if block_shape is not None:
        group_size = block_shape[1]
        if quant_mode == "fp8":
            fake_quant = lambda tensor: _fake_quantize_fp8_block(tensor, group_size)
        else:
            fake_quant = lambda tensor: _fake_quantize_int8_block(tensor, group_size)
    else:
        fake_quant = _fake_quantize_fp8 if quant_mode == "fp8" else _fake_quantize_int8

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Quantize activation before GEMM1
            h_q = fake_quant(hidden_states[m].unsqueeze(0)).squeeze(0)
            # GEMM1
            z = h_q.float() @ w1[e].T.float()
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # Quantize intermediate before GEMM2
            s_q = fake_quant(s.unsqueeze(0)).squeeze(0)
            # GEMM2
            r = s_q.float() @ w2[e].T.float()
            output[m] += (weight.float() * r).to(output.dtype)

    return output


def torch_w8a8_block_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    a = a.to(compute_type)
    b = b.to(compute_type)
    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous() and b_scales.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size
    assert (a.shape[-1] + block_k - 1) // block_k == a_scales.shape[-1]
    assert a.shape[:-1] == a_scales.shape[:-1]

    m = a.numel() // a.shape[-1]
    n, k = b.shape
    origin_c_shape = a.shape[:-1] + (n,)
    a = a.reshape(m, a.shape[-1])
    a_scales = a_scales.reshape(m, a_scales.shape[-1])
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == b_scales.shape[0]
    assert k_tiles == b_scales.shape[1]

    c = torch.zeros((m, n), dtype=compute_type, device=a.device)
    a_tiles = [a[:, i * block_k : min((i + 1) * block_k, k)] for i in range(k_tiles)]
    b_tiles = [
        [
            b[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    c_tiles = [c[:, j * block_n : min((j + 1) * block_n, n)] for j in range(n_tiles)]
    a_scale_tiles = [a_scales[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            scale = a_scale_tiles[i] * b_scales[j][i]
            c_tiles[j][:, :] += torch.matmul(a_tiles[i], b_tiles[j][i].t()) * scale

    return c.reshape(origin_c_shape).to(output_dtype)


def torch_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    finfo = torch.finfo(dtype)
    x_reshaped = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_reshaped.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    )
    x_scales = amax / finfo.max
    x_quant = (x_reshaped / x_scales).clamp(min=finfo.min, max=finfo.max).to(dtype)
    x_quant = x_quant.reshape(x.shape)
    x_scales = x_scales.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_quant, x_scales


def torch_w8a8_block_fp8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: list[int],
):
    batch_size, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    expanded_hidden = hidden_states.view(batch_size, -1, hidden_size).repeat(1, topk, 1)
    expanded_hidden = expanded_hidden.reshape(-1, hidden_size)
    out = torch.zeros(
        batch_size * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    flat_weights = topk_weights.view(-1)
    flat_ids = topk_ids.view(-1)
    _, block_k = block_shape
    hidden_q, hidden_scale = torch_per_token_group_quant_fp8(expanded_hidden, block_k)
    hidden_q = hidden_q.to(torch.float32)

    def silu_and_mul(x):
        import torch.nn.functional as F

        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    for expert_idx in range(w1.shape[0]):
        mask = flat_ids == expert_idx
        if mask.sum():
            inter = torch_w8a8_block_matmul(
                hidden_q[mask],
                w1[expert_idx],
                hidden_scale[mask],
                w1_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )
            act = silu_and_mul(inter)
            act_q, act_scale = torch_per_token_group_quant_fp8(act, block_k)
            out[mask] = torch_w8a8_block_matmul(
                act_q,
                w2[expert_idx],
                act_scale,
                w2_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )

    return (
        out.view(batch_size, -1, w2.shape[1])
        * flat_weights.view(batch_size, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.parametrize("block_shape", FUSED_MOE_BLOCK_SHAPES)
@pytest.mark.skipif(
    not is_cuda_available(),
    reason="FP8 quantization requires NVIDIA Hopper architecture",
)
def test_accuracy_fused_moe_fp8(config, block_shape):
    """Test FlagGems fused_moe with FP8 W8A8 quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create FP8 weights: quantize and store scale
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    if block_shape is None:
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_max = finfo.max
        eps = 1e-10

        w1_scales = []
        w1_fp8_list = []
        for e in range(num_experts):
            amax = w1_fp32[e].abs().amax().clamp(min=eps)
            scale = amax / fp8_max
            w1_q = (
                (w1_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
            )
            w1_fp8_list.append(w1_q)
            w1_scales.append(scale)
        w1_fp8 = torch.stack(w1_fp8_list)
        w1_scale = torch.tensor(w1_scales, device=device, dtype=torch.float32)

        w2_scales = []
        w2_fp8_list = []
        for e in range(num_experts):
            amax = w2_fp32[e].abs().amax().clamp(min=eps)
            scale = amax / fp8_max
            w2_q = (
                (w2_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
            )
            w2_fp8_list.append(w2_q)
            w2_scales.append(scale)
        w2_fp8 = torch.stack(w2_fp8_list)
        w2_scale = torch.tensor(w2_scales, device=device, dtype=torch.float32)

        w1_deq = torch.zeros_like(w1_fp32).to(dtype)
        for e in range(num_experts):
            w1_deq[e] = (w1_fp8[e].float() * w1_scales[e]).to(dtype)
        w2_deq = torch.zeros_like(w2_fp32).to(dtype)
        for e in range(num_experts):
            w2_deq[e] = (w2_fp8[e].float() * w2_scales[e]).to(dtype)
    else:
        block_n, block_k = block_shape
        w1_fp8, w1_scale, w1_deq = _quantize_fp8_blockwise(w1_fp32, block_n, block_k)
        w2_fp8, w2_scale, w2_deq = _quantize_fp8_blockwise(w2_fp32, block_n, block_k)
        w1_deq = w1_deq.to(dtype)
        w2_deq = w2_deq.to(dtype)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems FP8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_fused_moe_quantized_reference(
        hidden_states,
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
        quant_mode="fp8",
        block_shape=block_shape,
    )

    torch.cuda.synchronize()

    # FP8 quantization introduces more error than bf16, use wider tolerances.
    # Two quantized GEMMs + activation create cumulative rounding error.
    rtol = 5e-1
    atol = max(2e-1, ref.abs().max().item() * 1e-1)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_FP8_BLOCKWISE_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(
    not is_cuda_available(),
    reason="FP8 blockwise quantization requires NVIDIA Hopper architecture",
)
def test_accuracy_fused_moe_fp8_blockwise(config, block_shape):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    if hidden_size % block_shape[1] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    if intermediate_size % block_shape[0] != 0:
        pytest.skip("Invalid shape for block-wise quantization")

    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_fp8 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / hidden_size**0.5)
    ).to(torch.float8_e4m3fn)
    w2_fp8 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / intermediate_size**0.5)
    ).to(torch.float8_e4m3fn)

    w1_scale = torch.randn(
        num_experts,
        ceil(intermediate_size * 2 / block_shape[0]),
        ceil(hidden_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.randn(
        num_experts,
        ceil(hidden_size / block_shape[0]),
        ceil(intermediate_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_w8a8_block_fp8_moe(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        block_shape,
    )

    torch.cuda.synchronize()

    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.parametrize("block_shape", FUSED_MOE_BLOCK_SHAPES)
def test_accuracy_fused_moe_int8(config, block_shape):
    """Test FlagGems fused_moe with INT8 W8A8 per-channel quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights: quantize per-channel (per output column of each expert)
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    if block_shape is None:
        eps = 1e-10

        w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
        w1_scale_full = w1_amax / 127.0
        w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
        w1_scale = w1_scale_full.squeeze(-1)

        w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
        w2_scale_full = w2_amax / 127.0
        w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
        w2_scale = w2_scale_full.squeeze(-1)

        w1_deq = (w1_int8.float() * w1_scale_full).to(dtype)
        w2_deq = (w2_int8.float() * w2_scale_full).to(dtype)
    else:
        block_n, block_k = block_shape
        w1_int8, w1_scale, w1_deq = _quantize_int8_blockwise(w1_fp32, block_n, block_k)
        w2_int8, w2_scale, w2_deq = _quantize_int8_blockwise(w2_fp32, block_n, block_k)
        w1_deq = w1_deq.to(dtype)
        w2_deq = w2_deq.to(dtype)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=block_shape is None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_fused_moe_quantized_reference(
        hidden_states,
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
        quant_mode="int8",
        block_shape=block_shape,
    )

    torch.cuda.synchronize()

    # INT8 quantization introduces more error, use wider tolerances
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def torch_fused_moe_weight_only_reference(
    hidden_states: torch.Tensor,
    w1_int: torch.Tensor,
    w2_int: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference fused MoE for weight-only quantization.

    Weights are dequantized (w_int * scale) then used in FP computation.
    Activations remain in original precision (no activation quantization).
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Dequantize weights
            w1_deq = w1_int[e].float() * w1_scale[e].unsqueeze(-1).float()
            w2_deq = w2_int[e].float() * w2_scale[e].unsqueeze(-1).float()
            # GEMM1
            z = hidden_states[m].float() @ w1_deq.T
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # GEMM2
            r = s @ w2_deq.T
            output[m] += (weight.float() * r).to(output.dtype)

    return output


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int8_w8a16(config):
    """Test FlagGems fused_moe with INT8 W8A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    # Per-channel quantization
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)  # [E, 2D]

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)  # [E, K]

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 W8A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int8,
        w2_int8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize()

    # Weight-only quantization has less error than W8A8 since activations
    # are full precision, but still has weight quantization rounding error.
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int4_w4a16(config):
    """Test FlagGems fused_moe with INT4 W4A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT4 weights stored in INT8 containers, per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    int4_max = 7
    int4_min = -8

    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / int4_max
    w1_int4 = (w1_fp32 / w1_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / int4_max
    w2_int4 = (w2_fp32 / w2_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT4 W4A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int4,
        w2_int4,
        topk_weights,
        topk_ids,
        use_int4_w4a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int4,
        w2_int4,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize()

    # INT4 has coarser quantization → wider tolerance
    rtol = 3e-1
    atol = max(1e-1, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_inplace(config, dtype):
    """Test that inplace=True writes output into hidden_states."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # Non-inplace reference
    ref = flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
    )

    # Inplace result
    hidden_copy = hidden_states.clone()
    result = flag_gems.fused_experts_impl(
        hidden_copy,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
    )

    torch.cuda.synchronize()

    # Result should be the same tensor as input
    assert result.data_ptr() == hidden_copy.data_ptr(), "inplace should reuse input"
    torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_apply_router_weight_on_input(config, dtype):
    """Test apply_router_weight_on_input vs default (weight on output)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # Default (weight on GEMM2 output)
    result_default = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )

    # Weight on GEMM1 input
    result_on_input = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )

    torch.cuda.synchronize()

    # Due to SiLU nonlinearity, these will differ, but both should be
    # close to the reference with weight on the respective path.
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    # The default (weight on output) should match our standard reference
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(result_default, ref, rtol=rtol, atol=atol)

    # The apply_on_input result will differ but should be finite and nonzero
    assert torch.isfinite(
        result_on_input
    ).all(), "result_on_input has non-finite values"
    assert result_on_input.abs().sum() > 0, "result_on_input is all zeros"


try:
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except Exception:
    DEEPGEMM_AVAILABLE = False


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="vllm with deep_gemm is required.")
@pytest.mark.parametrize("batch_size, next_n", [(4, 1), (2, 2)])
@pytest.mark.parametrize("avg_ctx_len", [1024, 2048])
def test_get_paged_mqa_logits_metadata(batch_size, next_n, avg_ctx_len):
    context_lens_2d = (
        torch.randint(
            int(0.8 * avg_ctx_len), int(1.2 * avg_ctx_len), (batch_size, next_n)
        )
        .cuda()
        .to(torch.int32)
    )

    ref = get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())
    res = flag_gems.get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())

    assert torch.equal(ref, res)


# ---------------------- flashmla_sparse op test ----------------------
try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )

    HAS_VLLM_FLASHMLA_SPARSE = True
except ImportError:
    HAS_VLLM_FLASHMLA_SPARSE = False
    print(
        "Since vllm not installed, we adopt the native pytorch implementation of FlashMLA for comparison"
    )
    torch.set_float32_matmul_precision("high")


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


class FlashmlaSparseTestKit:
    # used by torch vertion flashmla_sparse
    @staticmethod
    def _merge_two_lse(
        lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
    ) -> torch.Tensor:
        if lse1 is None:
            return lse0
        else:
            return torch.logsumexp(
                torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
                dim=0,
            )

    # torch version flashmla_sparse
    @staticmethod
    def torch_flash_mla_sparse_fwd(
        s_q: int,
        s_kv: int,
        h_q: int,
        h_kv: int,
        d_qk: int,
        topk: int,
        q: torch.Tensor,  # [s_q, h_q, d_qk]
        kv: torch.Tensor,  # [s_q, 1, d_qk]
        indices: torch.Tensor,  # [s_q, 1, topk]
        sm_scale: float,
        d_v: int,
        attn_sink: Optional[torch.Tensor],  # [h_q]
        topk_length: Optional[torch.Tensor],  # [s_q]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - o: [s_q, h_q, dv]
        - o_fp32: [s_q, h_q, dv]
        - max_logits: [s_q, h_q]
        - lse: [s_q, h_q]
        """
        indices = indices.clone().squeeze(1)
        if topk_length is not None:
            mask = torch.arange(topk, device=topk_length.device).unsqueeze(
                0
            ).broadcast_to(s_q, topk) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0
        q = q.float()
        gathered_kv = (
            kv.index_select(dim=0, index=indices.flatten())
            .reshape(s_q, topk, d_qk)
            .float()
        )
        P = q @ gathered_kv.transpose(1, 2)
        P *= sm_scale
        P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

        orig_lse = torch.logsumexp(P, dim=-1)
        max_logits = P.max(dim=-1).values

        lse_for_o = FlashmlaSparseTestKit._merge_two_lse(orig_lse, attn_sink, s_q, h_q)
        if not torch.is_inference_mode_enabled():
            lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float(
            "+inf"
        )  # So that corresponding O will be 0
        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        out = s_for_o @ gathered_kv[..., :d_v]

        lonely_q_mask = orig_lse == float("-inf")
        orig_lse[lonely_q_mask] = float("+inf")
        return (out.to(torch.bfloat16), max_logits, orig_lse)

    @staticmethod
    def get_correctness_test_params():
        cases = [
            Flashmla_Sparse_Test_Param(s_q, s_kv, topk, h_q, h_kv, d_qk, d_v)
            for s_q in [64, 128, 512]
            for s_kv in [1024, 2048, 4096]
            for h_q in [64, 128, 256]
            for h_kv in [1]
            for d_qk in [576]
            for d_v in [512]
            for topk in [64, 128, 256]
        ]
        return cases

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_input(param: Flashmla_Sparse_Test_Param):
        """Create input data for sparse MLA operator"""
        S = param.s_q
        H = param.h_q
        DQK = param.d_qk
        SKV = param.s_kv
        HKV = param.h_kv
        topk = param.topk
        dtype = param.dtype
        device = param.device
        requires_grad = False

        FlashmlaSparseTestKit._init_seed(42)

        q = torch.randn((S, H, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )
        kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )

        indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32, device=device)
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[t, h, : len(i_i)] = i_i

        return q, kv, indices

    @staticmethod
    def get_correctness_test_params_flashmla():
        cases = [
            Flashmla_Sparse_Test_Param(
                s_q,
                s_kv,
                topk,
                h_q,
                d_qk=d_qk,
                have_attn_sink=have_attn_sink,
                have_topk_length=have_topk_length,
            )
            for s_q in [1, 62, 213]
            for h_q in [128, 64]
            for d_qk in [512, 576]
            for s_kv, topk in [
                (592, 128),
                (1840, 256),
                (1592, 384),
                (1521, 512),
                (95, 128),
                (153, 256),
                (114, 384),
            ]
            for have_attn_sink in [True, False]
            for have_topk_length in [True, False]
        ]
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]`
        holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])`
        as many as possible, and the rest are filled with `padding`.
        """
        assert not torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
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
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
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
        FlashmlaSparseTestKit._init_seed(_flashmla_sparse_counter)
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
        indices = FlashmlaSparseTestKit._randperm_batch(
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
        return q, kv, indices, attn_sink, topk_length


@pytest.mark.flashmla_sparse
@pytest.mark.parametrize("param", FlashmlaSparseTestKit.get_correctness_test_params())
def test_flashmla_sparse_correctness(param: Flashmla_Sparse_Test_Param):
    """Sparse MLA forward propagation test"""
    # Skip FlashMLA unsupported cases
    if param.h_q != 64 and param.h_q != 128:
        # RuntimeError: Unsupported h_q: 256
        # FlashMLA csrc/api/sparse_fwd.h:197
        pytest.skip("h_q unsupported by FlashMLA")
    if param.topk % 128 != 0:
        # Assertion `params.topk % (2*B_TOPK) == 0` failed
        # FlashMLA csrc/sm90/prefill/sparse/phase1.cuh:577
        # FlashMLA csrc/sm90/prefill/sparse/config.h:27 "B_TOPK = 64"
        pytest.skip("topk unsupported by FlashMLA")

    # Create input
    q, kv, indices = FlashmlaSparseTestKit.make_input(param)
    sm_scale = param.d_qk**-0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            None,
            None,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
    )

    # Accuracy comparison
    flag_gems.testing.assert_close(your_output, ref_output, param.dtype, atol=1e-2)
    flag_gems.testing.assert_close(
        your_max_logbits, ref_max_logbits, torch.float32, atol=1e-4
    )
    flag_gems.testing.assert_close(your_lse, ref_lse, torch.float32, atol=1e-4)


@pytest.mark.flashmla_sparse
@pytest.mark.parametrize(
    "param", FlashmlaSparseTestKit.get_correctness_test_params_flashmla()
)
def test_flashmla_sparse_correctness_flashmla(param: Flashmla_Sparse_Test_Param):
    """Sparse MLA forward propagation test from FlashMLA"""
    # Create input
    q, kv, indices, attn_sink, topk_length = FlashmlaSparseTestKit.make_input_flashmla(
        param
    )
    sm_scale = 0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            attn_sink,
            topk_length,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
    )

    # Accuracy comparison
    torch.testing.assert_close(
        your_output, ref_output, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )  # cos_diff_tol=7e-6
    torch.testing.assert_close(
        your_max_logbits, ref_max_logbits, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
    torch.testing.assert_close(
        your_lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
