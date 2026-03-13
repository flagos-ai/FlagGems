import logging
from math import ceil
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation quantization helpers (pure PyTorch, no vLLM / custom-C++ dependency)
# ---------------------------------------------------------------------------

# Default chunk size for processing tokens in chunks to avoid memory issues
# with very large batch sizes (mirroring vLLM's VLLM_FUSED_MOE_CHUNK_SIZE).
_FUSED_MOE_CHUNK_SIZE = 64 * 1024


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 (E4M3) quantization of activations.

    Supports three modes:
      - per-tensor (A_scale is scalar or None, per_act_token=False, block_shape=None)
      - per-token  (per_act_token=True, block_shape=None)
      - block-wise (block_shape=[block_n, block_k])
    """
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = finfo.min
    eps = 1e-10

    if block_shape is not None:
        # Block-wise quantization
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        # Reshape into groups
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / fp8_max
        A_q = (A_groups.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        # Per-token quantization
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / fp8_max
        # Apply minimum scaling factor for numerical stability
        min_scale = torch.tensor(1.0 / (fp8_max * 512.0), dtype=torch.float32,
                                 device=A.device)
        scale = scale.clamp(min=min_scale)
        A_q = (A_flat.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        # Per-tensor quantization (static if A_scale provided, dynamic otherwise)
        if A_scale is not None:
            scale = A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
            A_q = (A.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, A_scale
        else:
            amax = A.abs().amax().clamp(min=eps).to(torch.float32)
            scale = amax / fp8_max
            iscale = 1.0 / scale
            A_q = (A.float() * iscale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, scale.view(1)


def _int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 quantization of activations.

    Supports three modes:
      - per-token  (per_act_token=True, block_shape=None)
      - block-wise (block_shape=[block_n, block_k])
      - per-tensor (static: A_scale provided)
    """
    iinfo = torch.iinfo(torch.int8)
    int8_max = iinfo.max
    int8_min = iinfo.min
    eps = 1e-10

    if block_shape is not None:
        # Block-wise quantization
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / int8_max
        A_q = (A_groups.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        # Per-token quantization
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / int8_max
        A_q = (A_flat.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        # Per-tensor (static scale only for int8)
        assert A_scale is not None, "int8 per-tensor quantization requires A_scale"
        scale = A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
        A_q = (A.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        return A_q, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: Optional[torch.dtype],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the MoE kernel input activations before GEMM.

    Maps the quantization dtype to the appropriate quantizer.
    Returns (quantized_A, A_scale) if quantization is applied,
    or (A, A_scale) unchanged when quant_dtype is None.
    """
    if quant_dtype is None:
        return A, A_scale
    elif quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


def _get_quant_dtype(
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
) -> Optional[torch.dtype]:
    """Map quantization flags to torch dtype for activation quantization."""
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    else:
        return None


def _get_config_dtype_str(
    use_fp8_w8a8: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Optional[str]:
    """Return dtype string used for kernel config lookup."""
    if use_fp8_w8a8:
        return "fp8_w8a8"
    return None


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


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
            elif use_fp8_w8a8:
                # FP8 dot returns float32 natively, can use acc= for fusion
                accumulator = tl.dot(a, b, acc=accumulator)
            else:
                # INT8 dot returns int32; use += to trigger implicit int32→float32 cast
                accumulator += tl.dot(a, b)
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
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: torch.Tensor | None = None,
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


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Dispatch layer for the fused MoE kernel.

    Routes to the appropriate kernel implementation based on quantization flags.
    Currently only the generic Triton kernel path is implemented; WNA16 and
    other specialised paths can be added here in the future.

    Args:
        A: Input activations [M, K] (possibly quantized)
        B: Expert weight matrices [E, N, K] (possibly quantized)
        C: Output buffer [M, topk, N]
        A_scale: Activation quantization scale (or None)
        B_scale: Weight quantization scale (or None)
        B_zp: Weight zero-point (or None, reserved for WNA16)
        topk_weights: Router weights [M, topk] (or None)
        sorted_token_ids: From moe_align_block_size
        expert_ids: From moe_align_block_size
        num_tokens_post_padded: From moe_align_block_size
        mul_routed_weight: Whether to multiply router weights in-kernel
        top_k: Number of top experts per token
        config: Triton config dict
        compute_type: Triton dtype for compute
        use_fp8_w8a8: FP8 weight+activation quantization
        use_int8_w8a8: INT8 weight+activation quantization
        use_int8_w8a16: INT8 weight, FP16 activation (reserved)
        use_int4_w4a16: INT4 weight, FP16 activation (reserved)
        per_channel_quant: Per-channel quantization mode
        block_shape: [block_n, block_k] for block-wise quantization
        B_bias: Bias tensor (or None, reserved)
    """
    if False:
        # TODO: Other precision-specific implementations
        pass
    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
        )


def _apply_silu_and_mul(out: torch.Tensor, inp: torch.Tensor) -> None:
    """Apply SiLU-and-Mul activation: out = SiLU(inp[:, :N]) * inp[:, N:]."""
    N = inp.shape[-1] // 2
    x, y = inp[:, :N], inp[:, N:]
    _silu_and_mul_kernel(x, y, out0=out)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    # Legacy alias (kept for backward compatibility)
    num_experts: int = -1,
) -> torch.Tensor:
    """
    Complete fused MoE forward pass with optional quantization support.

    Pipeline:
        [quantize input] → moe_align_block_size → GEMM1(up+gate) → SiLU+Mul
        → [quantize intermediate] → GEMM2(down) → moe_sum

    Supports:
      - bf16 / fp16 (default, no quantization)
      - FP8 W8A8 (use_fp8_w8a8=True): weights and activations in FP8 E4M3
      - INT8 W8A8 (use_int8_w8a8=True): weights and activations in INT8
      - Per-tensor, per-token (per_channel_quant), or block-wise (block_shape)
        quantization scales
      - apply_router_weight_on_input: multiply router weight on GEMM1 input
        instead of GEMM2 output
      - inplace: write output into hidden_states tensor
      - Chunked processing for large batch sizes

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: [E, intermediate_size * 2, hidden_size]  (gate + up projection)
        w2: [E, hidden_size, intermediate_size]       (down projection)
        topk_weights: [num_tokens, topk]
        topk_ids: [num_tokens, topk]
        inplace: If True, write output into hidden_states
        activation: Activation function name ("silu")
        apply_router_weight_on_input: Multiply router weights on GEMM1 (True)
            or GEMM2 (False, default)
        use_fp8_w8a8: Enable FP8 weight+activation quantization
        use_int8_w8a8: Enable INT8 weight+activation quantization
        per_channel_quant: Use per-token activation quantization (paired with
            per-channel weight quantization)
        global_num_experts: Total number of experts (default: inferred from w1)
        w1_scale: Weight scale for w1 [E, 1, 1] or [E, N//gn, K//gk]
        w2_scale: Weight scale for w2 [E, 1, 1] or [E, K//gn, D//gk]
        a1_scale: Activation scale for GEMM1 input (or None for dynamic)
        a2_scale: Activation scale for GEMM2 input (or None for dynamic)
        block_shape: [block_n, block_k] for block-wise quantization
        w1_bias: Bias for w1 (currently unused, reserved)
        w2_bias: Bias for w2 (currently unused, reserved)
        num_experts: Legacy alias for global_num_experts

    Returns:
        output: [num_tokens, hidden_size]
    """
    logger.debug("GEMS FUSED MOE")
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"

    # Resolve num_experts (legacy alias vs new name)
    if global_num_experts <= 0:
        global_num_experts = num_experts

    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    assert hidden_states.size(1) == w1.size(2), (
        f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    )

    num_tokens_total, K = hidden_states.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]

    if global_num_experts <= 0:
        global_num_experts = E

    # Determine compute type
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")

    # Determine quantization dtype (None means no quantization)
    quant_dtype = _get_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    # Config dtype string for kernel config lookup
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        dtype=hidden_states.dtype,
    )

    # Chunk size: process tokens in chunks to avoid memory issues
    CHUNK_SIZE = _FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens_total, CHUNK_SIZE)

    # Get kernel config
    config = get_default_config(M, E, w2.shape[1], K, top_k, config_dtype, block_shape)

    # Allocate intermediate buffers
    # Memory optimization: cache1 and cache3 can share storage because
    # cache3 is only needed after cache1 is consumed by the activation.
    cache13_size = M * top_k * max(N, K)
    cache13 = torch.empty(
        cache13_size, device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache1 = cache13[: M * top_k * N].view(M, top_k, N)
    intermediate_cache3 = cache13[: M * top_k * K].view(M, top_k, K)

    # Activation output: SiLU+Mul halves the dimension
    activation_out_dim = N // 2
    intermediate_cache2 = torch.empty(
        (M * top_k, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # Output buffer
    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    # Process in chunks
    for chunk in range((num_tokens_total // CHUNK_SIZE) + 1):
        begin_chunk_idx = chunk * CHUNK_SIZE
        end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens_total)
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk = curr_hidden_states.size(0)

        if tokens_in_chunk == 0:
            break

        # Adjust caches for last (possibly smaller) chunk
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * top_k]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_default_config(
                tokens_in_chunk, E, w2.shape[1], K, top_k, config_dtype, block_shape
            )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        # Step 1: Quantize input activations (no-op if quant_dtype is None)
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # Step 2: Align tokens to experts
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], global_num_experts
        )

        # Step 3: GEMM1 — hidden_states @ W1 → intermediate_cache1
        dispatch_fused_moe_kernel(
            A=qcurr_hidden_states,
            B=w1,
            C=intermediate_cache1,
            A_scale=a1q_scale,
            B_scale=w1_scale,
            B_zp=None,
            topk_weights=curr_topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k,
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        # Step 4: Activation — SiLU(gate) * up
        _apply_silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        # Step 5: Quantize intermediate activations for GEMM2
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
        )

        # Step 6: GEMM2 — intermediate @ W2 → intermediate_cache3
        #         Multiply router weights here (unless applied on input)
        dispatch_fused_moe_kernel(
            A=qintermediate_cache2,
            B=w2,
            C=intermediate_cache3,
            A_scale=a2q_scale,
            B_scale=w2_scale,
            B_zp=None,
            topk_weights=curr_topk_weights if not apply_router_weight_on_input else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,  # After activation, each token-expert pair is independent
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        # Step 7: Reduce — sum over topK experts
        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place fused MoE: writes output directly into ``hidden_states``.

    Same semantics as ``fused_experts_impl(..., inplace=True)``.
    Returns None (the result is stored in ``hidden_states``).
    """
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Out-of-place fused MoE: allocates and returns a new output tensor.

    Same semantics as ``fused_experts_impl(..., inplace=False)``.
    """
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
