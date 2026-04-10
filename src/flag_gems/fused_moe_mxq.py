# SPDX-License-Identifier: Apache-2.0
# fused_moe_mxq.py
# QC-MoE W8A16 量化 MoE 算子实现

import torch
import triton
import triton.language as tl
from typing import Optional, List, Tuple, Any


# ============================================================================
# 工具类：QuantConfig, QuantMode
# ============================================================================

class QuantMode:
    FP16 = "fp16"
    W8A16 = "w8a16"
    W4A16 = "w4a16"


class QuantConfig:
    def __init__(
        self,
        mode: str = "w8a16",
        group_size: int = 128,
        w_nbits: int = 8,
        has_zero_point: bool = True,
    ):
        self.mode = QuantMode.FP16 if mode == "fp16" else QuantMode.W8A16 if mode == "w8a16" else QuantMode.W4A16
        self.group_size = group_size
        self.w_nbits = w_nbits
        self.has_zero_point = has_zero_point


def quantize_weights_moe(w: torch.Tensor, w_nbits: int = 8, group_size: int = 128):
    """
    MoE 权重量化 (W8A16)
    输入: w (E, K, H) FP16
    输出: w_q (E, K, H) uint8, scales (E, K, H//group_size) FP16, zeros (E, K, H//group_size) FP16
    """
    E, K, H = w.shape
    num_groups = H // group_size
    
    w = w.float()
    w_q = torch.zeros(E, K, H, dtype=torch.uint8, device=w.device)
    scales = torch.zeros(E, K, num_groups, dtype=torch.float32, device=w.device)
    zeros = torch.zeros(E, K, num_groups, dtype=torch.float32, device=w.device)
    
    for e in range(E):
        for k in range(K):
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                w_block = w[e, k, start:end]
                
                w_min = w_block.min()
                w_max = w_block.max()
                scale = (w_max - w_min) / 255.0
                zero = w_min
                
                w_q[e, k, start:end] = ((w_block - zero) / scale).round().to(torch.uint8)
                scales[e, k, g] = scale
                zeros[e, k, g] = zero
    
    return w_q, scales, zeros


# ============================================================================
# Triton Kernel 实现 - W8A16
# ============================================================================

@triton.jit
def _moe_kernel_fp16(
    A, C,
    B,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_valid_tokens,
    N, K,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """FP16 MoE Kernel (without quantization)"""
    pid = tl.program_id(0)
    
    if pid >= num_valid_tokens:
        return
    
    token_id = tl.load(sorted_token_ids + pid).to(tl.int64)
    expert_id = tl.load(expert_ids + pid).to(tl.int64)
    weight = tl.load(topk_weights + pid).to(tl.float32)
    
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    for k in range(K):
        a = tl.load(A + token_id * stride_am + k * stride_ak).to(tl.float32)
        w_ptr = B + expert_id * stride_be + k * stride_bk
        w = tl.load(w_ptr + offs_n * stride_bn, mask=offs_n < N).to(tl.float32)
        acc += a * w
    
    if BLOCK_SIZE_N > 0:
        acc = acc * weight
        output_ptrs = C + token_id * stride_cm + offs_n * stride_cn
        tl.atomic_add(output_ptrs, acc.to(tl.float16), mask=offs_n < N)


@triton.jit
def _moe_kernel_w8a16(
    A, C,
    B,
    B_scale,
    B_zp,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bse, stride_bsk, stride_bsn,
    stride_bze, stride_bzk, stride_bzn,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    has_zp: tl.constexpr,
):
    """
    W8A16 MoE Kernel
    
    W1 形状: (E, K, H) 其中 K=inter_dim, H=hidden_dim
    量化后:
      W_q: (E, K, H) uint8
      scales: (E, K, H/group_size) FP16
      zeros: (E, K, H/group_size) FP16
    """
    pid = tl.program_id(0)
    
    if pid >= num_valid_tokens:
        return
    
    token_id = tl.load(sorted_token_ids + pid).to(tl.int64)
    expert_id = tl.load(expert_ids + pid).to(tl.int64)
    weight = tl.load(topk_weights + pid).to(tl.float32)
    
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # 按 K 维度循环
    for k in range(K):
        # 加载激活
        a = tl.load(A + token_id * stride_am + k * stride_ak).to(tl.float32)
        
        # 计算 group 索引
        g = k // group_size
        
        # 加载权重: W[expert, k, :]
        w_ptr = B + expert_id * stride_be + k * stride_bk
        w = tl.load(
            w_ptr + offs_n * stride_bn,
            mask=offs_n < N,
            other=0.0
        ).to(tl.float32)
        
        # 加载 scales
        s_ptr = B_scale + expert_id * stride_bse + k * stride_bsk + g * stride_bsn
        scales = tl.load(
            s_ptr + offs_n * 0,
            mask=offs_n < N,
            other=1.0
        ).to(tl.float32)
        
        # 反量化
        if has_zp:
            z_ptr = B_zp + expert_id * stride_bze + k * stride_bzk + g * stride_bzn
            zeros = tl.load(
                z_ptr + offs_n * 0,
                mask=offs_n < N,
                other=0.0
            ).to(tl.float32)
            w_deq = w * scales + zeros
        else:
            w_deq = w * scales
        
        acc += a * w_deq
    
    if MUL_ROUTED_WEIGHT:
        acc = acc * weight
    
    output_ptrs = C + token_id * stride_cm + offs_n * stride_cn
    tl.atomic_add(output_ptrs, acc.to(tl.float16), mask=offs_n < N)


# ============================================================================
# 核心 fused_moe 函数
# ============================================================================

def fused_moe(
    x: torch.Tensor,
    w1: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
    w3: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    quant_config: Optional[Any] = None,
    num_experts: int = 8,
    top_k: int = 2,
    block_shape: Optional[List[int]] = None,
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
    """W8A16 量化 MoE W1 投影"""
    if quant_config is None:
        quant_config = QuantConfig(mode="fp16")
    
    if len(x.shape) == 3:
        x = x.view(-1, x.shape[-1])
    
    num_tokens, hidden_dim = x.shape
    
    if topk_weights is None or topk_ids is None:
        topk_weights = torch.ones(num_tokens, top_k, device=x.device, dtype=x.dtype) / top_k
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=x.device)
    
    token_indices = torch.arange(num_tokens, device=x.device, dtype=torch.int64)
    sorted_token_ids = token_indices.unsqueeze(1).expand(num_tokens, top_k).contiguous().view(-1)
    flat_expert_ids = topk_ids.view(-1)
    flat_weights = topk_weights.view(-1)
    
    sorted_indices = torch.argsort(flat_weights, dim=0, descending=True)
    sorted_token_ids = sorted_token_ids[sorted_indices]
    sorted_expert_ids = flat_expert_ids[sorted_indices]
    sorted_weights = flat_weights[sorted_indices]
    
    block_size_m = 32
    num_tokens_post_padded = ((num_tokens * top_k + block_size_m - 1) // block_size_m) * block_size_m
    num_valid_tokens = sorted_token_ids.shape[0]
    
    if w1_q is not None and w1_scales is not None:
        W1_q = w1_q
        W1_scales = w1_scales
        W1_zeros = w1_zeros
    elif w1 is not None:
        mode_str = quant_config.mode.value if hasattr(quant_config.mode, 'value') else str(quant_config.mode)
        if mode_str == "fp16":
            W1_q = w1
            W1_scales = None
            W1_zeros = None
        else:
            W1_q, W1_scales, W1_zeros = quantize_weights_moe(w1, quant_config.w_nbits, quant_config.group_size)
    else:
        raise ValueError("Either w1 or w1_q must be provided")
    
    if not x.is_contiguous():
        x = x.contiguous()
    
    _, inter_dim, _ = W1_q.shape
    K = hidden_dim
    N = inter_dim
    
    output = torch.zeros(num_tokens, inter_dim, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_N = min(128, N)
    BLOCK_SIZE_K = min(64, K)
    grid = (num_valid_tokens,)
    
    mode_str = quant_config.mode.value if hasattr(quant_config.mode, 'value') else str(quant_config.mode)
    
    if mode_str == "fp16":
        _moe_kernel_fp16[grid](
            x, output,
            W1_q,
            sorted_weights, sorted_token_ids, sorted_expert_ids,
            num_valid_tokens,
            N=N, K=K,
            stride_am=x.stride(0), stride_ak=x.stride(1),
            stride_be=W1_q.stride(0), stride_bk=W1_q.stride(1), stride_bn=W1_q.stride(2),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    elif mode_str == "w8a16":
        _moe_kernel_w8a16[grid](
            x, output,
            W1_q, W1_scales, W1_zeros if W1_zeros is not None else x.new_tensor([]),
            sorted_weights, sorted_token_ids, sorted_expert_ids,
            num_tokens_post_padded,
            N=N, K=K, EM=num_valid_tokens, num_valid_tokens=num_valid_tokens,
            stride_am=x.stride(0), stride_ak=x.stride(1),
            stride_be=W1_q.stride(0), stride_bk=W1_q.stride(1), stride_bn=W1_q.stride(2),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            stride_bse=W1_scales.stride(0), stride_bsk=W1_scales.stride(1), stride_bsn=W1_scales.stride(2),
            stride_bze=W1_zeros.stride(0) if W1_zeros is not None else 0,
            stride_bzk=W1_zeros.stride(1) if W1_zeros is not None else 0,
            stride_bzn=W1_zeros.stride(2) if W1_zeros is not None else 0,
            group_size=quant_config.group_size,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            MUL_ROUTED_WEIGHT=True, top_k=top_k,
            has_zp=quant_config.has_zero_point,
        )
    else:
        raise ValueError(f"Unsupported quantization mode: {mode_str}")
    
    return output


class FusedMoELinear(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        quant_config: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_config = quant_config or QuantConfig()
        self.w1 = torch.nn.Parameter(
            torch.randn(num_experts, inter_dim, hidden_dim, requires_grad=False)
        )
        self._packed = False
    
    def pack(self):
        self.W1_q, self.W1_scales, self.W1_zeros = quantize_weights_moe(
            self.w1.data, self.quant_config.w_nbits, self.quant_config.group_size
        )
        self._packed = True
    
    def forward(
        self,
        x: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._packed:
            self.pack()
        return fused_moe(
            x, w1=None, w2=None, w3=None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=self.quant_config,
            num_experts=self.num_experts, top_k=self.top_k,
            w1_q=self.W1_q, w1_scales=self.W1_scales, w1_zeros=self.W1_zeros,
        )
    
    def set_weights(self, w1: torch.Tensor):
        self.w1.data = w1
        self._packed = False


def create_moe_op(
    mode: str = "w8a16",
    num_experts: int = 8,
    top_k: int = 2,
    group_size: int = 128,
) -> Tuple[Any, Any]:
    quant_config = QuantConfig(mode=mode, group_size=group_size)
    return fused_moe, quant_config