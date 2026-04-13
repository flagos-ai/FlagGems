# SPDX-License-Identifier: Apache-2.0
# QC-MoE: Quantized Mixture of Experts kernel for FlagGems
# Main module integrating MoE kernels with quantization support

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum


# Device detection
_is_cuda = torch.cuda.is_available()

if _is_cuda:
    def is_sm90_supported():
        device_cap = torch.cuda.get_device_capability()
        return device_cap[0] >= 9  # H100, H200, etc.
else:
    def is_sm90_supported():
        return False


# ============================================================================
# QuantMode and QuantConfig
# ============================================================================

class QuantMode(Enum):
    """Quantization modes supported by QC-MoE."""
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    W8A16 = "w8a16"  # INT8 weight, FP16 activation
    W4A16 = "w4a16"  # INT4 weight, FP16 activation


@dataclass
class QuantConfig:
    """Configuration for MoE quantization."""
    mode: QuantMode = QuantMode.FP16
    group_size: int = 128
    has_zero_point: bool = True
    per_channel_quant: bool = False
    
    @property
    def w_nbits(self) -> int:
        """Get weight bit width from mode."""
        if self.mode == QuantMode.W4A16:
            return 4
        elif self.mode in (QuantMode.W8A16, QuantMode.INT8, QuantMode.FP8):
            return 8
        return 16
    
    @property
    def use_int4(self) -> bool:
        return self.mode == QuantMode.W4A16
    
    @property
    def use_int8(self) -> bool:
        return self.mode in (QuantMode.W8A16, QuantMode.INT8)


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    A,
    B,
    C,
    B_scale,
    B_zp,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
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
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    filter_expert: tl.constexpr,
):
    """
    Simplified MoE kernel for single dispatch entry processing.
    Each program processes one (token, expert) pair.
    """
    pid = tl.program_id(0)
    
    # Check bounds
    if pid >= num_valid_tokens:
        return
    
    # Load dispatch information
    token_id = tl.load(sorted_token_ids + pid).to(tl.int64)
    expert_id = tl.load(expert_ids + pid).to(tl.int64)
    weight = tl.load(topk_weights + pid).to(compute_type)
    
    # Precompute strides
    stride_bn_c = tl.constexpr(stride_bn)
    stride_bk_c = tl.constexpr(stride_bk)
    stride_bsn_c = tl.constexpr(stride_bsn)
    stride_bsk_c = tl.constexpr(stride_bsk)
    stride_bzn_c = tl.constexpr(stride_bzn)
    stride_bzk_c = tl.constexpr(stride_bzk)
    stride_be_c = tl.constexpr(stride_be)
    stride_bse_c = tl.constexpr(stride_bse)
    stride_bze_c = tl.constexpr(stride_bze)
    
    # offs_n: range of N elements
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Process all K elements in BLOCK_SIZE_K chunks
    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = k_block * BLOCK_SIZE_K
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        k_indices = k_base + offs_k
        k_mask = k_indices < K
        
        # Load activation: A[token_id, k_indices]
        a = tl.load(A + (token_id * stride_am + k_indices * stride_ak), mask=k_mask, other=0.0).to(tl.float32)
        
        # Load weight values: W[expert_id, offs_n, k_indices]
        w = tl.load(
            B + (expert_id * stride_be_c + offs_n[None, :] * stride_bn_c + k_indices[:, None] * stride_bk_c),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        # Dequantize weights
        if use_int4_w4a16:
            w = (w & 0xF).to(compute_type)
        elif use_int8_w8a16:
            w = w.to(compute_type)
        
        # Load scales: scales[expert_id, offs_n, group]
        scale_group = k_indices // group_size
        scales = tl.load(
            B_scale + (expert_id * stride_bse_c + offs_n[None, :] * stride_bsn_c + scale_group[:, None] * stride_bsk_c),
            mask=k_mask[:, None] & n_mask[None, :],
            other=1.0
        ).to(tl.float32)
        
        # Dequantize based on quantization mode
        if use_int4_w4a16:
            if has_zp:
                zp = tl.load(
                    B_zp + (expert_id * stride_bze_c + offs_n[None, :] * stride_bzn_c + scale_group[:, None] * stride_bzk_c),
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0
                ).to(tl.float32)
                w_dequant = (w.to(tl.float32) - zp) * scales
            else:
                w_dequant = (w.to(tl.float32) - 8.0) * scales
        elif use_int8_w8a16:
            if has_zp:
                zp = tl.load(
                    B_zp + (expert_id * stride_bze_c + offs_n[None, :] * stride_bzn_c + scale_group[:, None] * stride_bzk_c),
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0
                ).to(tl.float32)
                w_dequant = (w.to(tl.float32) - zp) * scales
            else:
                w_dequant = (w.to(tl.float32) - 128.0) * scales
        else:
            # No quantization - weights are already in compute_type (FP16)
            w_dequant = w.to(tl.float32) * scales
        
        # Compute matrix multiply using expand and sum: [BLOCK_SIZE_K, BLOCK_SIZE_N] * [BLOCK_SIZE_K, 1]
        a_expanded = a[:, None]  # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        result = tl.sum(a_expanded * w_dequant, axis=0)  # [BLOCK_SIZE_N]
        
        # Accumulate
        accumulator = accumulator + result
    
    # Apply routing weight
    if MUL_ROUTED_WEIGHT:
        accumulator = accumulator * weight
    
    accumulator = accumulator.to(compute_type)
    
    # Store result using atomic add
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N
    output_ptrs = C + (token_id * stride_cm + offs_n * stride_cn)
    tl.atomic_add(output_ptrs, accumulator, mask=n_mask)


@triton.jit
def fused_moe_kernel_fp16_swiglu(
    A, C,
    B_gate, B_up, B_down,
    topk_weights, sorted_token_ids, expert_ids,
    num_tokens_post_padded,
    inter_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_gate_e, stride_up_e, stride_down_e,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_down_k, stride_down_n,
    stride_inter_m,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    FP16 SwiGLU MoE — complete gate(W1)/up(W3)/down(W2) in one dispatch entry.

    FFN(x) = W2 @ (silu(W1 @ x) * (W3 @ x))
    Each program processes one (token, expert) pair.
    All loops use 1-element scalar iterations to avoid shape-compatibility issues.
    """
    pid = tl.program_id(0)
    if pid >= num_valid_tokens:
        return

    token_id = tl.load(sorted_token_ids + pid).to(tl.int64)
    expert_id = tl.load(expert_ids + pid).to(tl.int64)
    weight = tl.load(topk_weights + pid).to(tl.float32)

    # Compute inter_size = N in multiples of 32; partial blocks handled by mask
    inter_off = pid * stride_inter_m

    # ---------- GEMM 1: gate_acc[n] = sum_k( A[token,k] * W1[exp,n,k] ) ----------
    for n in range(N):
        acc = 0.0
        for kb in range(tl.cdiv(K, BLOCK_SIZE_K)):
            k_offs = kb * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K
            a_vals = tl.load(A + token_id * stride_am + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            w_gate = tl.load(
                B_gate
                + expert_id * stride_gate_e
                + n * stride_gate_n
                + k_offs * stride_gate_k,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            acc = acc + tl.sum(a_vals * w_gate)
        # Store gate result to inter[n] (we reuse the same buffer; gate first)
        gate_val = acc
        tl.store(inter_ptr + inter_off + n, gate_val)

    # ---------- GEMM 2: up_acc[n] = sum_k( A[token,k] * W3[exp,n,k] ), multiply with gate ----------
    for n in range(N):
        acc = 0.0
        for kb in range(tl.cdiv(K, BLOCK_SIZE_K)):
            k_offs = kb * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K
            a_vals = tl.load(A + token_id * stride_am + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            w_up = tl.load(
                B_up
                + expert_id * stride_up_e
                + n * stride_up_n
                + k_offs * stride_up_k,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            acc = acc + tl.sum(a_vals * w_up)
        gate_val = tl.load(inter_ptr + inter_off + n).to(tl.float32)
        # SiLU(gate) * up -> store back as intermediate
        act_val = tl.sigmoid(gate_val) * acc
        tl.store(inter_ptr + inter_off + n, act_val)

    # ---------- GEMM 3: down_acc[k] = sum_n( inter[n] * W2[exp,k,n] ), then scale and store ----------
    for k in range(K):
        acc = 0.0
        for nb in range(tl.cdiv(N, 32)):
            base_n = nb * 32
            n_offs = base_n + tl.arange(0, 32)
            n_mask = n_offs < N
            inter_vals = tl.load(inter_ptr + inter_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
            w_down = tl.load(
                B_down
                + expert_id * stride_down_e
                + k * stride_down_k
                + n_offs * stride_down_n,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            acc = acc + tl.sum(inter_vals * w_down)
        result = (acc * weight).to(tl.float16)
        out_idx = token_id * stride_cm + k * stride_cn
        cur = tl.load(C + out_idx).to(tl.float16)
        tl.store(C + out_idx, cur + result)


# ============================================================================
# Helper Functions
# ============================================================================

def get_num_experts(shape_desc: str) -> int:
    """Extract number of experts from shape description.
    
    Common patterns:
    - Qwen3.5-397B-A17B: 8 experts
    - Mixtral-8x7B: 8 experts
    - Switch Transformer: variable
    """
    if "Qwen" in shape_desc:
        if "397B" in shape_desc:
            return 8
        elif "72B" in shape_desc:
            return 8
    elif "Mixtral" in shape_desc:
        return 8
    elif "Switch" in shape_desc:
        return 64
    return 8  # default


def prepare_moe_inputs(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for fused MoE kernel.
    
    Args:
        x: Input tensor of shape (num_tokens, hidden_dim)
        topk_weights: Weights for selected experts, shape (num_tokens, topk)
        topk_ids: Expert indices, shape (num_tokens, topk)
        num_experts: Total number of experts
        
    Returns:
        sorted_token_ids: Sorted token indices
        expert_ids: Expert index for each block
        num_tokens_post_padded: Total tokens after padding
        block_size_m: Block size for tokens
    """
    num_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    
    # Flatten and prepare for MoE dispatch
    flat_topk_weights = topk_weights.view(-1)
    flat_topk_ids = topk_ids.view(-1)
    
    # Create mapping from token to expert selection
    _, sorted_token_ids = torch.sort(flat_topk_weights, dim=0, descending=True)
    
    # Get expert assignments
    expert_ids = flat_topk_ids[sorted_token_ids]
    
    # Pad to block size
    block_size_m = 32  # Default block size
    num_tokens_post_padded = ((num_tokens * topk + block_size_m - 1) // block_size_m) * block_size_m
    
    return sorted_token_ids, expert_ids, num_tokens_post_padded, block_size_m


def quantize_weights_moe(
    weights: torch.Tensor,
    num_experts: int,
    quant_config: QuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize MoE expert weights.
    
    Args:
        weights: Expert weights of shape (num_experts, out_features, in_features)
        num_experts: Number of experts
        quant_config: Quantization configuration
        
    Returns:
        W_q: Quantized weights (same shape as input if int8, packed if int4)
        scales: Quantization scales of shape (num_experts, out_features, num_groups)
        zeros: Optional zero points of same shape as scales
    """
    if quant_config.mode == QuantMode.FP16:
        return weights, None, None
    
    num_experts_e, n_out, k_in = weights.shape
    num_groups = k_in // quant_config.group_size
    
    if quant_config.use_int4:
        w_bits = 4
    else:
        w_bits = 8
    
    # Reshape for per-group quantization along the last dimension
    # weights shape: (E, n_out, k_in) -> (E, n_out, num_groups, group_size)
    weights_reshaped = weights.view(num_experts, n_out, num_groups, quant_config.group_size)
    w_min = weights_reshaped.min(dim=-1, keepdim=True)[0]
    w_max = weights_reshaped.max(dim=-1, keepdim=True)[0]
    scale = (w_max - w_min) / ((2 ** w_bits) - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    # Quantize
    W_normalized = (weights_reshaped - w_min) / (scale + 1e-8)
    W_q = W_normalized.round().clamp(0, 2 ** w_bits - 1)
    W_q = W_q.to(torch.uint8)
    
    # Reshape back - pack if int4
    if quant_config.use_int4:
        # Pack 2 int4 values per byte
        W_q = W_q.view(num_experts, n_out, num_groups, quant_config.group_size // 2, 2)
        W_q_packed = (W_q[..., 0] & 0xF) | (W_q[..., 1] << 4)
        W_q = W_q_packed.view(num_experts, n_out, -1)
    else:
        W_q = W_q.view(num_experts, n_out, -1)
    
    # Scales shape: (num_experts, n_out, num_groups)
    scales = scale.squeeze(-1).view(num_experts, n_out, num_groups)
    
    # Zero points if needed
    zeros = None
    if quant_config.has_zero_point:
        zeros = w_min.squeeze(-1).view(num_experts, n_out, num_groups)
    
    return W_q, scales, zeros


def get_default_config(block_size_m=1, block_size_n=128, block_size_k=64):
    """Get default kernel configuration with reduced sizes for shared memory."""
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
    }


def get_autotune_config():
    """Get autotuning configurations for MoE kernel with reduced sizes for H20."""
    return [
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
    ]


# ============================================================================
# Kernel Invocation
# ============================================================================

_fp16_intermediate_buf = None


def invoke_fused_moe(
    x: torch.Tensor,
    W1_q: torch.Tensor,
    W2_q: torch.Tensor,
    W3_q: Optional[torch.Tensor],
    output: torch.Tensor,
    W1_scales: torch.Tensor,
    W1_zeros: Optional[torch.Tensor],
    W2_scales: torch.Tensor,
    W2_zeros: Optional[torch.Tensor],
    W3_scales: Optional[torch.Tensor],
    W3_zeros: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    quant_config: Any,
    block_shape: List[int],
) -> None:
    """
    Invoke the fused MoE kernel.
    FP16 mode uses a dedicated SwiGLU path; quantized modes use fused_moe_kernel_gptq_awq.
    """
    global _fp16_intermediate_buf

    num_tokens, hidden_dim = x.shape
    num_experts, inter_dim, _ = W1_q.shape
    num_valid_tokens = sorted_token_ids.shape[0]

    K = hidden_dim
    N = inter_dim

    if topk_weights.dim() > 1:
        topk_weights = topk_weights.view(-1)

    BLOCK_SIZE_N = min(128, N)
    BLOCK_SIZE_K = min(64, K)
    grid = (num_valid_tokens,)

    if not x.is_contiguous():
        x = x.contiguous()

    output.zero_()

    # FP16 fast path — complete SwiGLU MoE: gate(W1) * up(W3), then W2 @ act
    if quant_config.mode.value == "fp16" and W2_q is not None:
        # FP16 SwiGLU mode requires all weights (W1, W2, optionally W3)
        inter_buf = torch.empty(
            num_valid_tokens * N, dtype=x.dtype, device=x.device
        )
        _W3 = W3_q if W3_q is not None else W1_q  # use W1 if W3 missing

        fused_moe_kernel_fp16_swiglu[grid](
            x,
            output,
            W1_q,          # gate
            _W3,           # up
            W2_q,          # down
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            inter_buf,
            N=N,
            K=K,
            EM=num_valid_tokens,
            num_valid_tokens=num_valid_tokens,
            stride_am=x.stride(0),
            stride_ak=x.stride(1),
            stride_bn=W1_q.stride(1),
            stride_bk=W1_q.stride(2),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            stride_gate_e=W1_q.stride(0),
            stride_up_e=_W3.stride(0),
            stride_down_e=W2_q.stride(0),
            stride_gate_n=W1_q.stride(1),
            stride_gate_k=W1_q.stride(2),
            stride_up_n=_W3.stride(1),
            stride_up_k=_W3.stride(2),
            stride_down_k=W2_q.stride(1),
            stride_down_n=W2_q.stride(2),
            stride_inter_m=N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            top_k=top_k,
            even_Ks=(K % BLOCK_SIZE_K) == 0,
        )
        return

    # FP16 W1-only: use vectorized torch.mm as the reference implementation
    # This is called when FP16 mode with W2_q=None reaches this function
    # (weights were not quantized, so W1_scales is None)
    if quant_config.mode.value == "fp16" and W2_q is None:
        num_experts = W1_q.shape[0]
        n_out = W1_q.shape[1]
        
        # topk_weights is already flattened at this point
        # Vectorized approach: process each expert in batch using torch.matmul
        for e in range(num_experts):
            # Find all dispatch entries for expert e
            mask = (expert_ids == e)
            if not mask.any():
                continue
            
            indices = mask.nonzero(as_tuple=True)[0]
            # Bounds check for padding
            valid_mask = indices < num_valid_tokens
            indices = indices[valid_mask]
            
            # Skip if no valid entries
            if indices.numel() == 0:
                continue
            
            # Get token indices and weights
            token_indices = sorted_token_ids[indices]
            weights_e = topk_weights[indices]
            
            # Batch compute: W1[e] @ x[token_indices].T
            # W1[e]: [n_out, k_in], x_e: [num_selections, k_in]
            # Result: [n_out, num_selections]
            x_e = x[token_indices]  # [num_selections, k_in]
            result = torch.matmul(W1_q[e], x_e.t())  # [n_out, num_selections]
            
            # Apply weights and transpose: result.T * weights
            # result.T: [num_selections, n_out], weights: [num_selections]
            result = result.t() * weights_e.unsqueeze(1)  # [num_selections, n_out]
            
            # Use index_add for efficient accumulation (avoids Python loop)
            output.index_add_(0, token_indices, result)
        
        return

    # Quantized path (W8A16 / W4A16) OR FP16 W1-only path
    # W2_q is None means W1-only projection (quantized or FP16)
    if W2_q is None:
        # Determine if we should skip dequantization (FP16 mode with unit scales)
        is_fp16_w1_only = (quant_config.mode.value == "fp16" and 
                          W1_q is not None and W1_scales is not None and
                          W1_zeros is None)
        
        # For FP16 W1-only: skip INT8 offset (use_int8_w8a16=False)
        # For quantized modes: use appropriate dequantization
        kernel_use_int8 = quant_config.use_int8 and not is_fp16_w1_only
        kernel_has_zp = quant_config.has_zero_point and not is_fp16_w1_only
        
        # W1-only quantization path
        fused_moe_kernel_gptq_awq[grid](
            x,
            W1_q,
            output,
            W1_scales,
            W1_zeros if W1_zeros is not None else x.new_tensor([]),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N=N,
            K=K,
            EM=num_valid_tokens,
            num_valid_tokens=num_valid_tokens,
            stride_am=x.stride(0),
            stride_ak=x.stride(1),
            stride_be=W1_q.stride(0),
            stride_bk=W1_q.stride(2),
            stride_bn=W1_q.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            stride_bse=W1_scales.stride(0),
            stride_bsk=W1_scales.stride(2),
            stride_bsn=W1_scales.stride(1),
            stride_bze=W1_zeros.stride(0) if W1_zeros is not None else 0,
            stride_bzk=W1_zeros.stride(2) if W1_zeros is not None else 0,
            stride_bzn=W1_zeros.stride(1) if W1_zeros is not None else 0,
            group_size=quant_config.group_size,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=1,
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.float16,
            has_zp=kernel_has_zp,
            use_int4_w4a16=quant_config.use_int4,
            use_int8_w8a16=kernel_use_int8,
            even_Ks=(K % BLOCK_SIZE_K) == 0,
            filter_expert=False,
        )
    else:
        # W1 + W2 quantization path (SwiGLU)
        fused_moe_kernel_gptq_awq[grid](
            x,
            W1_q,
            output,
            W1_scales,
            W1_zeros if W1_zeros is not None else x.new_tensor([]),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N=N,
            K=K,
            EM=num_valid_tokens,
            num_valid_tokens=num_valid_tokens,
            stride_am=x.stride(0),
            stride_ak=x.stride(1),
            stride_be=W1_q.stride(0),
            stride_bk=W1_q.stride(2),
            stride_bn=W1_q.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            stride_bse=W1_scales.stride(0),
            stride_bsk=W1_scales.stride(2),
            stride_bsn=W1_scales.stride(1),
            stride_bze=W1_zeros.stride(0) if W1_zeros is not None else 0,
            stride_bzk=W1_zeros.stride(2) if W1_zeros is not None else 0,
            stride_bzn=W1_zeros.stride(1) if W1_zeros is not None else 0,
            group_size=quant_config.group_size,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=1,
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.float16,
            has_zp=quant_config.has_zero_point,
            use_int4_w4a16=quant_config.use_int4,
            use_int8_w8a16=quant_config.use_int8,
            even_Ks=(K % BLOCK_SIZE_K) == 0,
            filter_expert=False,
        )


# ============================================================================
# Main fused_moe Function
# ============================================================================

def fused_moe(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    quant_config: QuantConfig = None,
    num_experts: int = 8,
    top_k: int = 2,
    block_shape: Optional[List[int]] = None,
    # Optional pre-quantized weights (from benchmark)
    w1_q: Optional[torch.Tensor] = None,
    w1_scales: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_q: Optional[torch.Tensor] = None,
    w2_scales: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    w3_q: Optional[torch.Tensor] = None,
    w3_scales: Optional[torch.Tensor] = None,
    w3_zeros: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused Mixture of Experts computation with quantization support.
    
    This implements:
        y = sum_i(topk_weights_i * FFN(experts_i(topk_ids_i)))
    
    For SwiGLU MoE:
        FFN(x) = Gate(x) * Up(x) = (silu(W1(x)) * W3(x)) @ W2
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim) or (num_tokens, hidden_dim)
        w1: First FFN layer weights (FP16) or can be pre-quantized (uint8)
        w2: Second FFN layer weights (FP16) or can be pre-quantized (uint8)
        w3: Optional gate weights for SwiGLU, shape (num_experts, hidden_dim, inter_dim)
        topk_weights: Weights for top-k experts, shape (batch_size, seq_len, top_k)
        topk_ids: Expert indices, shape (batch_size, seq_len, top_k)
        quant_config: Quantization configuration
        num_experts: Number of experts
        top_k: Number of experts to select
        block_shape: Block shape for block-wise quantization [block_n, block_k]
        # Pre-quantized weights (if provided, skips quantization)
        w1_q, w1_scales, w1_zeros: Pre-quantized W1 weights
        w2_q, w2_scales, w2_zeros: Pre-quantized W2 weights
        w3_q, w3_scales, w3_zeros: Pre-quantized W3 weights
        
    Returns:
        Output tensor of same shape as x
    """
    if quant_config is None:
        quant_config = QuantConfig()
    
    # Handle input shape
    original_shape = x.shape
    if len(x.shape) == 3:
        x = x.view(-1, x.shape[-1])  # (B*S, H)
    
    num_tokens = x.shape[0]
    
    # Prepare routing information
    if topk_weights is None or topk_ids is None:
        # Create dummy routing for testing
        topk_weights = torch.ones(num_tokens, top_k, device=x.device, dtype=x.dtype) / top_k
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=x.device)
    
    # Create dispatch arrays for MoE
    # Each token has top_k expert selections, create entries for each (token, expert) pair
    # sorted_token_ids: token index for each dispatch entry (repeated for each expert selection)
    # expert_ids: expert index for each dispatch entry
    
    # Create token indices: [0,0,1,1,...] where each token repeats top_k times
    token_indices = torch.arange(num_tokens, device=x.device, dtype=torch.int64)
    sorted_token_ids = token_indices.unsqueeze(1).expand(num_tokens, top_k).contiguous().view(-1)
    
    # Expert IDs: [e0_0, e0_1, ..., e1_0, e1_1, ...]
    flat_expert_ids = topk_ids.view(-1)
    
    # Weights: [w0_0, w0_1, ..., w1_0, w1_1, ...]
    flat_weights = topk_weights.view(-1)
    
    # Sort by weight for efficient processing (optional, helps with cache locality)
    sorted_indices = torch.argsort(flat_weights, dim=0, descending=True)
    sorted_token_ids = sorted_token_ids[sorted_indices]
    sorted_expert_ids = flat_expert_ids[sorted_indices]
    sorted_weights = flat_weights[sorted_indices]
    
    # Pad to block size
    block_size_m = 32
    num_tokens_post_padded = ((num_tokens * top_k + block_size_m - 1) // block_size_m) * block_size_m
    
    # Quantize weights if not pre-quantized
    if w1_q is not None and w1_scales is not None:
        # Use pre-quantized weights from benchmark
        W1_q = w1_q.contiguous()
        W1_scales = w1_scales.contiguous()
        W1_zeros = w1_zeros.contiguous() if w1_zeros is not None else None
    elif w1 is not None:
        W1_q, W1_scales, W1_zeros = quantize_weights_moe(w1, num_experts, quant_config)
    else:
        raise ValueError("Either w1 or w1_q must be provided")
    
    if w2_q is not None and w2_scales is not None:
        W2_q = w2_q.contiguous()
        W2_scales = w2_scales.contiguous()
        W2_zeros = w2_zeros.contiguous() if w2_zeros is not None else None
    elif w2 is not None:
        W2_q, W2_scales, W2_zeros = quantize_weights_moe(w2, num_experts, quant_config)
    else:
        # W2 not provided, set to None for W1-only projection
        W2_q = None
        W2_scales = None
        W2_zeros = None
    
    if w3 is not None:
        if w3_q is not None and w3_scales is not None:
            W3_q = w3_q.contiguous()
            W3_scales = w3_scales.contiguous()
            W3_zeros = w3_zeros.contiguous() if w3_zeros is not None else None
        else:
            W3_q, W3_scales, W3_zeros = quantize_weights_moe(w3, num_experts, quant_config)
    else:
        W3_q, W3_scales, W3_zeros = None, None, None
    
    # For FP16 W1-only mode, the weights are not quantized (quantize returns them as-is)
    # W1_scales will be None, so invoke_fused_moe handles this case directly
    # No need to create fake scales here
    
    # Allocate output
    # For W1-only projection (W2_q is None): output shape is (num_tokens, inter_dim)
    # For SwiGLU (W2_q is not None): output shape is same as input (num_tokens, hidden_dim)
    if W2_q is None and W1_q is not None:
        # W1-only projection: output is (num_tokens, inter_dim)
        num_experts_e, n_out, k_in = W1_q.shape
        output = torch.zeros(num_tokens, n_out, dtype=x.dtype, device=x.device)
    else:
        output = torch.zeros_like(x)
    
    # Default block shape
    if block_shape is None:
        block_shape = [128, 128]
    
    # Invoke fused MoE kernel
    invoke_fused_moe(
        x,
        W1_q,
        W2_q,
        W3_q,
        output,
        W1_scales,
        W1_zeros,
        W2_scales,
        W2_zeros,
        W3_scales,
        W3_zeros,
        sorted_token_ids,
        sorted_expert_ids,
        num_tokens_post_padded,
        sorted_weights,
        top_k,
        quant_config,
        block_shape,
    )
    
    # Reshape output
    if len(original_shape) == 3:
        output = output.view(original_shape)
    
    return output


# ============================================================================
# FusedMoELinear Module
# ============================================================================

class FusedMoELinear(torch.nn.Module):
    """
    Fused MoE Linear layer with quantization support.
    
    This module wraps the fused MoE computation for use in neural networks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        quant_config: QuantConfig = None,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_config = quant_config or QuantConfig()
        
        # SwiGLU MoE weights
        self.w1 = torch.nn.Parameter(
            torch.randn(num_experts, inter_dim, hidden_dim, requires_grad=False)
        )
        self.w3 = torch.nn.Parameter(
            torch.randn(num_experts, inter_dim, hidden_dim, requires_grad=False)
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(num_experts, hidden_dim, inter_dim, requires_grad=False)
        )
        
        self._packed = False
    
    def pack(self):
        """Prepare weights for quantized computation."""
        self.W1_q, self.W1_scales, self.W1_zeros = quantize_weights_moe(
            self.w1.data, self.num_experts, self.quant_config
        )
        self.W3_q, self.W3_scales, self.W3_zeros = quantize_weights_moe(
            self.w3.data, self.num_experts, self.quant_config
        )
        self.W2_q, self.W2_scales, self.W2_zeros = quantize_weights_moe(
            self.w2.data, self.num_experts, self.quant_config
        )
        self._packed = True
    
    def forward(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MoE.
        
        Args:
            x: Input tensor (B, S, H) or (T, H)
            topk_weights: Expert weights (B, S, K) or (T, K)
            topk_ids: Expert indices (B, S, K) or (T, K)
            
        Returns:
            Output tensor same shape as x
        """
        if not self._packed:
            self.pack()
        
        return fused_moe(
            x,
            self.w1,
            self.w2,
            self.w3,
            topk_weights,
            topk_ids,
            self.quant_config,
            self.num_experts,
            self.top_k,
        )
    
    def set_weights(self, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor):
        """Set weights from external source (e.g., model loading)."""
        self.w1.data = w1
        self.w3.data = w3
        self.w2.data = w2
        self._packed = False


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "fused_moe",
    "fused_moe_kernel_gptq_awq",
    "fused_moe_kernel_fp16_swiglu",
    "invoke_fused_moe",
    "FusedMoELinear",
    "QuantConfig",
    "QuantMode",
    "quantize_weights_moe",
    "prepare_moe_inputs",
    "get_num_experts",
    "get_default_config",
    "get_autotune_config",
]
