# SPDX-License-Identifier: Apache-2.0
# mxq/ultis.py
# 混合精度相关工具函数

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# 量化模式定义
# ============================================================================

class QuantMode:
    """量化模式枚举"""
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    W8A16 = "w8a16"  # INT8 权重, FP16 激活
    W4A16 = "w4a16"  # INT4 权重, FP16 激活


# ============================================================================
# 量化配置
# ============================================================================

class QuantConfig:
    """MoE 量化配置类"""
    
    def __init__(
        self,
        mode: str = "fp16",
        group_size: int = 128,
        has_zero_point: bool = True,
        per_channel_quant: bool = False,
    ):
        self.mode = QuantMode.FP16 if mode == "fp16" else (
            QuantMode.W8A16 if mode == "w8a16" else (
                QuantMode.W4A16 if mode == "w4a16" else QuantMode.FP16
            )
        ) if isinstance(mode, str) else mode
        
        self.group_size = group_size
        self.has_zero_point = has_zero_point
        self.per_channel_quant = per_channel_quant
    
    @property
    def w_nbits(self) -> int:
        """从模式获取权重位宽"""
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
# 量化工具函数
# ============================================================================

def quantize_weights_moe(
    weights: torch.Tensor,
    w_nbits: int = 8,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对 MoE 专家权重进行逐组量化，支持不均匀分组。
    
    Args:
        weights: 权重张量，形状 (E, n_out, k_in)，例如 W1=(E, K, H), W2=(E, H, K)
        w_nbits: 量化位数，8 或 4
        group_size: 量化组大小，沿 k_in 维度分组
        
    Returns:
        W_q: 量化后的 uint8 权重，形状与输入相同（INT4 时会打包）
        scales: 量化缩放因子，形状 (E, n_out, num_groups)
        zeros: 零点，形状 (E, n_out, num_groups)
    """
    num_experts, n_out, k_in = weights.shape
    num_groups = (k_in + group_size - 1) // group_size  # 向上取整
    
    w_bits = 8 if w_nbits == 8 else 4
    
    # 填充到 group_size 的整数倍
    pad_len = num_groups * group_size - k_in
    
    if pad_len > 0:
        weights_padded = torch.nn.functional.pad(weights, (0, pad_len))
    else:
        weights_padded = weights
    
    # 重塑为分组量化格式: (E, n_out, num_groups, group_size)
    weights_reshaped = weights_padded.view(num_experts, n_out, num_groups, group_size)
    w_min = weights_reshaped.min(dim=-1, keepdim=True)[0]
    w_max = weights_reshaped.max(dim=-1, keepdim=True)[0]
    scale = (w_max - w_min) / ((2 ** w_bits) - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    # 量化
    W_normalized = (weights_reshaped - w_min) / (scale + 1e-8)
    W_q = W_normalized.round().clamp(0, 2 ** w_bits - 1)
    W_q = W_q.to(torch.uint8)
    
    # 恢复形状
    if w_nbits == 4:
        # INT4: 每字节打包 2 个值
        W_q = W_q.view(num_experts, n_out, num_groups, group_size // 2, 2)
        W_q = (W_q[..., 0] & 0xF) | (W_q[..., 1] << 4)
        W_q = W_q.view(num_experts, n_out, -1)
    else:
        W_q = W_q.view(num_experts, n_out, -1)
    
    # scales 和 zeros 形状: (E, n_out, num_groups)
    scales = scale.squeeze(-1).view(num_experts, n_out, num_groups)
    zeros = w_min.squeeze(-1).view(num_experts, n_out, num_groups)
    
    return W_q, scales, zeros


def dequantize_w8a16(
    W_q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    """
    反量化 W8A16 权重到 FP16。
    """
    E, K, H = W_q.shape
    group_size = H // scales.shape[-1]
    W_f = W_q.float()
    S = scales.unsqueeze(-1)
    Z = zeros.unsqueeze(-1)
    W_deq = (W_f.unsqueeze(-1) - Z) * S
    W_deq = W_deq.squeeze(-1).view(E, K, scales.shape[-1], group_size)
    W_deq = W_deq.permute(0, 1, 3, 2).reshape(E, K, H)
    return W_deq


def dequantize_w4a16(
    W_q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    """
    反量化 W4A16 权重到 FP16。
    """
    E, K, Hp2 = W_q.shape
    H = Hp2 * 2
    group_size = H // scales.shape[-1]
    
    W_u8 = W_q.unsqueeze(-1).expand(E, K, Hp2, 2)
    lo = (W_u8 & 0x0F).float()
    hi = ((W_u8 >> 4) & 0x0F).float()
    
    S = scales.unsqueeze(-1)
    Z = zeros.unsqueeze(-1)
    
    def reshape_groups(t):
        t = t.view(E, K, scales.shape[-1], group_size)
        t = t.permute(0, 1, 3, 2).reshape(E, K, H)
        return t
    
    W_lo = reshape_groups(lo)
    W_hi = reshape_groups(hi)
    W_deq = ((W_lo + W_hi) - Z) * S
    return W_deq


# ============================================================================
# FP16 参考实现（PyTorch 原生）
# ============================================================================

def fp16_moe_reference(
    inp: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    纯 PyTorch FP16 SwiGLU MoE 参考实现（vLLM 风格）。
    
    FFN(x) = W2 @ (silu(W1 @ x) * (W3 @ x))
    """
    M, H = inp.shape
    E, K, _ = W1.shape
    output = torch.zeros(M, H, dtype=inp.dtype, device=inp.device)
    
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]
        
        inp_e = inp.index_select(0, tokens_e)
        gate = torch.mm(inp_e, W1[e].T)
        up = torch.mm(inp_e, W3[e].T) if W3 is not None else torch.zeros_like(gate)
        act = torch.nn.functional.silu(gate) * up if W3 is not None else torch.nn.functional.silu(gate)
        down = torch.mm(act, W2[e].T)
        
        down_w = down * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, H), down_w)
    
    return output


def fp16_moe_w1_only_reference(
    inp: torch.Tensor,
    W1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    W1 投影 FP16 参考实现（仅 W1，不含 SwiGLU）。
    
    Args:
        inp: (M, H) 序列长度, 隐藏维度
        W1: (E, K, H) gate 投影
        topk_weights: (M, top_k) top-k 专家权重
        topk_ids: (M, top_k) top-k 专家索引
        
    Returns:
        output: (M, K) W1 投影输出
    """
    M, H = inp.shape
    E, K, _ = W1.shape
    output = torch.zeros(M, K, dtype=inp.dtype, device=inp.device)
    
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]
        
        inp_e = inp.index_select(0, tokens_e)
        result = torch.mm(inp_e, W1[e].T)  # (T, K)
        
        down_w = result * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, K), down_w)
    
    return output


# ============================================================================
# 精度验证工具
# ============================================================================

def verify_moe_accuracy(
    output_qc: torch.Tensor,
    output_ref: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Dict[str, Any]:
    """
    验证 MoE 输出精度。
    """
    diff = (output_qc.float() - output_ref.float()).abs()
    
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "all_close": torch.allclose(output_qc, output_ref, rtol=rtol, atol=atol),
        "rel_err_max": (diff / (output_ref.abs() + 1e-8)).max().item(),
    }


def verify_moe_w1_accuracy(
    output_qc: torch.Tensor,
    output_ref: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Dict[str, Any]:
    """
    验证 MoE W1 投影精度。
    """
    diff = (output_qc.float() - output_ref.float()).abs()
    
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "all_close": torch.allclose(output_qc, output_ref, rtol=rtol, atol=atol),
        "rel_err_max": (diff / (output_ref.abs() + 1e-8)).max().item(),
    }


# ============================================================================
# MoE FLOPs 和带宽计算
# ============================================================================

def calc_moe_tflops(
    seq_len: int,
    hidden_dim: int,
    inter_dim: int,
    top_k: int,
    latency_ms: float,
    w1_only: bool = False,
) -> float:
    """
    计算 MoE 操作的 TFLOPS。
    """
    if w1_only:
        total_flops = seq_len * top_k * 2 * hidden_dim * inter_dim
    else:
        total_flops = seq_len * top_k * 4 * hidden_dim * inter_dim
    return total_flops / latency_ms / 1e9


def calc_moe_gbps(
    seq_len: int,
    hidden_dim: int,
    inter_dim: int,
    num_experts: int,
    w_nbits: int,
    group_size: int,
    latency_ms: float,
    w1_only: bool = False,
) -> float:
    """
    计算 MoE 操作的 GB/s。
    """
    input_bytes = seq_len * hidden_dim * 2
    
    if w1_only:
        weight_bytes = num_experts * inter_dim * hidden_dim * w_nbits / 8
        num_groups = inter_dim * hidden_dim // group_size
        scale_bytes = num_groups * 2 * 2
        output_bytes = seq_len * inter_dim * 2
    else:
        weight_bytes_1 = num_experts * inter_dim * hidden_dim * w_nbits / 8
        weight_bytes_2 = num_experts * hidden_dim * inter_dim * w_nbits / 8
        weight_bytes = weight_bytes_1 + weight_bytes_2
        num_groups_w1 = inter_dim * hidden_dim // group_size
        num_groups_w2 = hidden_dim * inter_dim // group_size
        scale_bytes = (num_groups_w1 + num_groups_w2) * 2 * 2
        output_bytes = seq_len * hidden_dim * 2
    
    total_bytes = input_bytes + weight_bytes + scale_bytes + output_bytes
    return total_bytes / latency_ms / 1e9


# ============================================================================
# Qwen3.5 MoE 形状配置
# ============================================================================

QWEN3_SHAPES = {
    "tiny": (512, 1024, 3584),
    "small": (2048, 1024, 3584),
    "medium": (8192, 1024, 3584),
    "large": (16384, 1024, 3584),
    "xlarge": (32768, 1024, 3584),
}

QWEN3_DEFAULT_CONFIG = {
    "num_experts": 8,
    "top_k": 2,
    "group_size": 128,
    "hidden_dim": 1024,
    "inter_dim": 3584,
}
