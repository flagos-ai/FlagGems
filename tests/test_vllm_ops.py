import random
from itertools import product
from math import ceil
from typing import Optional

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


@pytest.mark.fused_moe
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

    torch.cuda.synchronize()

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


@pytest.mark.fused_moe
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


def torch_fused_moe_quantized_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_mode: str = "fp8",
) -> torch.Tensor:
    """Reference fused MoE with simulated quantization noise.

    Simulates the quantization → dequantization round-trip on activations
    to model the same numerical behavior as the quantized kernel path.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

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


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not is_cuda_available(),
    reason="FP8 quantization requires NVIDIA Hopper architecture",
)
def test_accuracy_fused_moe_fp8(config):
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

    # Per-tensor quantization of weights
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10

    # Quantize w1 per-expert
    w1_scales = []
    w1_fp8_list = []
    for e in range(num_experts):
        amax = w1_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w1_q = (w1_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w1_fp8_list.append(w1_q)
        w1_scales.append(scale)
    w1_fp8 = torch.stack(w1_fp8_list)
    w1_scale = torch.tensor(w1_scales, device=device, dtype=torch.float32)

    # Quantize w2 per-expert
    w2_scales = []
    w2_fp8_list = []
    for e in range(num_experts):
        amax = w2_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w2_q = (w2_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w2_fp8_list.append(w2_q)
        w2_scales.append(scale)
    w2_fp8 = torch.stack(w2_fp8_list)
    w2_scale = torch.tensor(w2_scales, device=device, dtype=torch.float32)

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
    )

    # Reference: use the dequantized weights (fp8 → float) for reference
    w1_deq = torch.zeros_like(w1_fp32).to(dtype)
    for e in range(num_experts):
        w1_deq[e] = (w1_fp8[e].float() * w1_scales[e]).to(dtype)
    w2_deq = torch.zeros_like(w2_fp32).to(dtype)
    for e in range(num_experts):
        w2_deq[e] = (w2_fp8[e].float() * w2_scales[e]).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="fp8"
    )

    torch.cuda.synchronize()

    # FP8 quantization introduces more error than bf16, use wider tolerances.
    # Two quantized GEMMs + activation create cumulative rounding error.
    rtol = 5e-1
    atol = max(2e-1, ref.abs().max().item() * 1e-1)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_moe
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


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int8(config):
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

    eps = 1e-10

    # Per-channel quantization of weights: scale per [expert, output_dim]
    # w1 shape: [E, 2D, K] → scale shape: [E, 2D, 1]
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    # For the kernel: w1_scale shape [E, 2D] (per-channel: one scale per output dim)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

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
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use dequantized weights
    w1_deq = (w1_int8.float() * w1_scale_full).to(dtype)
    w2_deq = (w2_int8.float() * w2_scale_full).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="int8"
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


@pytest.mark.fused_moe
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


@pytest.mark.fused_moe
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


@pytest.mark.fused_moe
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


@pytest.mark.fused_moe
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
