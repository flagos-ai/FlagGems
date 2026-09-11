"""Simple wrapper for gluon FA3 warp-specialization implementation, returns (output, lse).

For FlagGems integration.
"""

from flag_gems.ops.gluon_attn_mha import attn_ws_mha


def gluon_flash_attn(q, k, v, causal=True, softmax_scale=None):
    """
    Inputs: q, k, v [B, S, H, D] bf16
    Outputs: (output, lse)
             output [B, S, H, D] bf16
             lse [B, H, S] float32
    softmax_scale defaults to 1/sqrt(D) if None.
    """
    out, lse = attn_ws_mha(
        q,
        k,
        v,
        causal=causal,
        num_warps=8,
        return_lse=True,
        softmax_scale=softmax_scale,
    )
    return out, lse
