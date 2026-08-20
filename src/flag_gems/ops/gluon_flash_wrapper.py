"""gluon FA3 warp-specialization 实现的简单 wrapper,返回 (output, dummy_lse)。

FlagGems 集成使用,LSE 先返回零占位。
"""
import torch
from flag_gems.ops.gluon_attn_mha import attn_ws_mha


def gluon_flash_attn(q, k, v, causal=True):
    """
    输入: q,k,v [B, S, H, D] bf16
    输出: (output, lse)
          output [B, S, H, D] bf16
          lse [B, H, S] float32 (暂时为零)
    """
    out = attn_ws_mha(q, k, v, causal=causal, num_warps=8)
    B, S, H, D = q.shape
    lse = torch.zeros(B, H, S, dtype=torch.float32, device=q.device)
    return out, lse
