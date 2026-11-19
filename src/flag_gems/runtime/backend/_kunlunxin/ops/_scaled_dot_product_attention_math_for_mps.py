"""Kunlunxin XPU3 override of `aten::_scaled_dot_product_attention_math_for_mps`.

Shares the mask-free kernel of the sibling math override.  The generic kernel
fails to lower here for two reasons: masked 2-D loads feeding `tl.dot`
(`ConvertTritonSDNNToLLVM`) and `tl.where(l_i == 0.0, 1.0, l_i)`
(`LLVM ERROR: Invalid selectType_: cannot invert non-select EW funct`).  Both
are absent from the shared kernel (see
``_scaled_dot_product_attention_math.py``).
"""

import logging

from ._scaled_dot_product_attention_math import _sdpa_math_impl

logger = logging.getLogger(__name__)


def _scaled_dot_product_attention_math_for_mps(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    *,
    scale=None,
):
    """Kunlunxin override: math SDPA returning (output, attention weights)."""
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_ATTENTION_MATH_FOR_MPS")

    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"
    assert attn_mask is None, "attn_mask is not supported"
    assert dropout_mask is None, "dropout_mask is not supported"
    assert query.shape[-1] == key.shape[-1], "query/key head_dim must match"

    # The mps reference left-pads the causal mask when q/kv lengths differ
    # (bottom-right alignment); the math reference uses top-left.  Tests and
    # benchmarks use equal lengths, where the two coincide.
    sq = query.shape[-2]
    sk = key.shape[-2]
    return _sdpa_math_impl(
        query,
        key,
        value,
        attn_mask=None,
        is_causal=is_causal,
        scale=scale,
        causal_diag=sk - sq,
    )
