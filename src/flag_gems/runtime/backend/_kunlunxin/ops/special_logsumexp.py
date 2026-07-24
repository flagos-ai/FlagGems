import logging

import torch

from flag_gems.runtime import torch_device_fn

from .logsumexp import _reduce_inner

logger = logging.getLogger(__name__)

# special_logsumexp (kunlunxin / XPU override).
#
# torch.special.logsumexp is numerically identical to torch.logsumexp:
# log(sum(exp(x), dim)) with a max-shift for stability. kunlunxin did NOT
# override it, so it fell to the generic ops/special_logsumexp.py, whose two
# @triton.heuristics(softmax_*) kernels recompile per shape (IR explosion, the
# 22K-line ir-special_logsumexp-dev1.log) and run the inner reduction with a
# streaming online-logsumexp loop that is catastrophically slow on XPU
# ([4096,4096] dim=1 gems 9.18ms vs torch 0.17ms -> speedup 0.019).
#
# The proven kunlunxin `logsumexp` override already solved the exact same
# reduction: inner-dim (K==1) uses the fast constexpr-N multirow tile
# (`_reduce_inner`, block DMA, @libentry -> compiles once); middle-dim (K>1) and
# N==1 defer to the vendor's native fused kernel (a Triton middle reduction on
# XPU is a dead end -- transpose+contiguous can't reach the vendor copy once
# gems overrides copy_, and a strided reduce overflows uni_sram / mis-computes).
# The benchmark always narrows dim=1 of a 2D tensor -> K==1 fast path.
#
# We reuse `_reduce_inner` from the logsumexp override verbatim and only differ
# in the native fallback op (aten.special_logsumexp).

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


def _native_special_logsumexp(inp, dim, keepdim):
    """Reach PyTorch's native (vendor) special_logsumexp, bypassing gems."""
    return torch.ops.aten.special_logsumexp.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, keepdim
    )


def special_logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            return inp.clone()
        if len(dim) != 1:
            # Multi-dim reduction: the vendor's native kernel beats a sequence
            # of Triton reductions on this XPU.
            return _native_special_logsumexp(inp, list(dim), keepdim)
        dim = dim[0]

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    N = inp.shape[dim]
    K = 1
    for i in range(dim + 1, inp.ndim):
        K *= inp.shape[i]

    # Middle-dim reduction (K > 1) or a size-1 reduction: defer to the native
    # vendor kernel (same rationale as the logsumexp override).
    if K > 1 or N == 1:
        return _native_special_logsumexp(inp, [dim], keepdim)

    # K == 1: innermost-dim reduction -> fast contiguous Triton multirow tile.
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1

    with torch_device_fn.device(inp.device):
        out = _reduce_inner(inp, M, N).view(shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
