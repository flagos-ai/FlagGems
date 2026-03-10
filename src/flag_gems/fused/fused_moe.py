import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel

logger = logging.getLogger(__name__)


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    per_channel_quant: tl.constexpr,
):
    """
    Fused MoE GEMM kernel with expert-based indirect addressing.

    Computes: C[t, :] = A[t // topk, :] @ B[expert(t), :, :] [* topk_weight[t]]

    Key Parameters:
    - A: Input activations [M, K] (or quantized)
    - B: Stacked expert weights [E, N, K]
    - C: Output [num_sorted_tokens, N]  (indexed by sorted_token_ids)
    - sorted_token_ids: Per-expert sorted token indices (from moe_align_block_size)
    - expert_ids: Expert index for each M-block
    """
    # Map program id to the block of C it should compute.
    # Grouped ordering promotes L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load sorted token indices for this M-block
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # Determine which expert this block belongs to
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # Set up A and B pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Load quantization scales based on mode
    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # block-wise quantization
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            # per-channel quantization
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            # per-tensor quantization
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # Main GEMM loop: accumulate in float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            # Fused dot-accumulate: on SM90 this maps to WGMMA with
            # in-place accumulation, avoiding a separate add instruction.
            accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Post-loop dequantization
    if (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Router weight multiplication (in float32 for numerical stability)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    """Return a reasonable default Triton config for the fused MoE kernel."""
    if dtype == "fp8_w8a8" and block_shape is not None:
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128

        # --- Tile sizing optimised for H100/H800 SM90 GPUs ---
        # Larger N/K tiles improve compute intensity and reduce grid
        # launches for the common case where N is large (e.g. 14336).
        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        elif N >= 1024:
            block_n = 64 if M <= 64 else 128
        else:
            block_n = 64 if M <= 64 else 128

        # K-tile: 128 gives better arithmetic intensity.
        if dtype == "fp8_w8a8":
            block_k = 128
        elif K >= 4096 or M <= 64:
            block_k = 128
        else:
            block_k = 64

        # Group-M: promotes L2 reuse across M-blocks.
        tokens_per_expert = (M * topk) // max(E, 1)
        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        num_warps = 4 if block_m * block_n < 8192 else 8
        num_stages = 3

        # Shared-memory guard (~232 KB on H100/H800).
        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
) -> None:
    """
    Launch the fused_moe_kernel Triton kernel.

    Args:
        A: Input activations [M, K]
        B: Expert weight matrices [E, N, K]
        C: Output buffer [M, topk, N]
        A_scale: Activation quantization scale (or None)
        B_scale: Weight quantization scale (or None)
        topk_weights: Router weights [M, topk] (or None)
        sorted_token_ids: From moe_align_block_size
        expert_ids: From moe_align_block_size
        num_tokens_post_padded: From moe_align_block_size
        mul_routed_weight: Whether to multiply router weights in-kernel
        top_k: Number of top experts per token
        config: Triton config dict with BLOCK_SIZE_M/N/K, GROUP_SIZE_M, etc.
        compute_type: Triton dtype for compute (tl.bfloat16, tl.float16, etc.)
        use_fp8_w8a8: FP8 weight+activation quantization
        use_int8_w8a8: INT8 weight+activation quantization
        per_channel_quant: Per-channel quantization mode
        block_shape: [block_n, block_k] for block-wise quantization
    """
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    config = config.copy()
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),  # N
        B.size(2),  # K
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        per_channel_quant=per_channel_quant,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
    )


def _apply_silu_and_mul(out: torch.Tensor, inp: torch.Tensor) -> None:
    """Apply SiLU-and-Mul activation: out = SiLU(inp[:, :N]) * inp[:, N:]."""
    N = inp.shape[-1] // 2
    x, y = inp[:, :N], inp[:, N:]
    silu_and_mul_kernel(x, y, out0=out)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int = -1,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Complete fused MoE forward pass (bf16/fp16, no quantization).

    Pipeline:
        moe_align_block_size → GEMM1(up+gate) → SiLU+Mul → GEMM2(down) → moe_sum

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: [E, intermediate_size * 2, hidden_size]  (gate + up projection)
        w2: [E, hidden_size, intermediate_size]       (down projection)
        topk_weights: [num_tokens, topk]
        topk_ids: [num_tokens, topk]
        num_experts: Total number of experts (default: inferred from w1)
        activation: Activation function name ("silu")

    Returns:
        output: [num_tokens, hidden_size]
    """
    logger.debug("GEMS FUSED MOE")
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w1.shape[1]  # intermediate_size * 2
    top_k = topk_ids.shape[1]

    if num_experts <= 0:
        num_experts = E

    # Determine compute type
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")

    # Get kernel config
    config = get_default_config(M, E, w2.shape[1], K, top_k, None)

    # Step 1: Align tokens to experts
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    # Allocate intermediate buffers
    # GEMM1 output: [M, topk, N]
    intermediate_cache1 = torch.empty(
        (M, top_k, N), dtype=hidden_states.dtype, device=hidden_states.device
    )
    # After activation (SiLU+Mul): [M * topk, N // 2]
    intermediate_cache2 = torch.empty(
        (M * top_k, N // 2), dtype=hidden_states.dtype, device=hidden_states.device
    )
    # GEMM2 output: [M, topk, K]
    intermediate_cache3 = torch.empty(
        (M, top_k, K), dtype=hidden_states.dtype, device=hidden_states.device
    )
    # Final output: [M, K]
    output = torch.zeros((M, K), dtype=hidden_states.dtype, device=hidden_states.device)

    # Step 2: GEMM1 — hidden_states @ W1 → intermediate_cache1
    invoke_fused_moe_triton_kernel(
        A=hidden_states,
        B=w1,
        C=intermediate_cache1,
        A_scale=None,
        B_scale=None,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=compute_type,
    )

    # Step 3: Activation — SiLU(gate) * up
    _apply_silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    # Step 4: GEMM2 — intermediate @ W2 → intermediate_cache3
    #         Multiply router weights here
    invoke_fused_moe_triton_kernel(
        A=intermediate_cache2,
        B=w2,
        C=intermediate_cache3,
        A_scale=None,
        B_scale=None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,  # After activation, each token-expert pair is independent
        config=config,
        compute_type=compute_type,
    )

    # Step 5: Reduce — sum over topK experts
    moe_sum(intermediate_cache3, output)

    return output
