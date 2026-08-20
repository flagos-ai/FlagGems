"""gluon FA3 warp-specialization 实现的简单 wrapper,返回 (output, lse)。

FlagGems 集成使用。
"""
from flag_gems.ops.gluon_attn_mha import attn_ws_mha


def gluon_flash_attn(q, k, v, causal=True, softmax_scale=None):
    """
    输入: q,k,v [B, S, H, D] bf16
    输出: (output, lse)
          output [B, S, H, D] bf16
          lse [B, H, S] float32
    softmax_scale 为 None 时默认 1/sqrt(D)。
    """
    out, lse = attn_ws_mha(
        q, k, v, causal=causal, num_warps=8, return_lse=True,
        softmax_scale=softmax_scale,
    )
    return out, lse
