import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024
_NUM_WARPS = 4

# Fill-path launch geometry: the >=256 MB band saturates write bandwidth with
# tall tiles, while smaller sizes are wave-quantisation sensitive and prefer
# fine granularity.
_BIG_WORD_BLOCK = 16384
_BIG_WORD_WARPS = 8
_MID_WORD_BLOCK = 1024
_MID_WORD_WARPS = 4
_BIG_WORD_MIN = 1 << 26
_WORD_TRUE = 0x01010101


@triton.jit
def _fill_true_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.full((BLOCK,), 1, tl.uint8), mask=offs < n_elements)


@triton.jit
def _fill_words_kernel(out_ptr, n_words, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    val = tl.full((BLOCK,), 0x01010101, tl.int32)
    if base + BLOCK <= n_words:
        tl.store(out_ptr + offs, val)
    else:
        tl.store(out_ptr + offs, val, mask=offs < n_words)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 8192}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _imag_contig_kernel(xr_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    lanes = 2 * offs[:, None] + tl.arange(0, 2)[None, :]
    pair = tl.load(xr_ptr + lanes, mask=lanes < 2 * n_elements, other=0)
    _re, im = tl.split(pair)
    tl.store(out_ptr + offs, (im == 0).to(tl.uint8), mask=mask)


@triton.jit
def _imag_strided_kernel(
    xr_ptr,
    out_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs
    src = tl.zeros((BLOCK,), tl.int32)
    for d in tl.static_range(RANK - 1, -1, -1):
        sz = SHAPE[d]
        idx = rem % sz
        rem = rem // sz
        src = src + idx * STRIDES[d]
    im = tl.load(xr_ptr + src + 1, mask=mask, other=0)
    tl.store(out_ptr + offs, (im == 0).to(tl.uint8), mask=mask)


def isreal(x):
    logger.debug("GEMS_ILUVATAR ISREAL")
    n = x.numel()
    if n == 0:
        return torch.empty(x.shape, dtype=torch.bool, device=x.device)
    if not x.dtype.is_complex:
        if (n & 3) == 0:
            n_words = n >> 2
            words = torch.empty(n_words, dtype=torch.int32, device=x.device)
            if n_words >= _BIG_WORD_MIN:
                blk = _BIG_WORD_BLOCK
                warps = _BIG_WORD_WARPS
            else:
                blk = _MID_WORD_BLOCK
                warps = _MID_WORD_WARPS
            _fill_words_kernel[((n_words + blk - 1) // blk,)](
                words, n_words, BLOCK=blk, num_warps=warps
            )
            return words.view(torch.bool).view(x.shape)
        out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        _fill_true_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            out, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
        return out.view(torch.bool)

    if x.is_conj():
        # isreal is invariant under conjugation (imag sign flip only), and
        # view_as_real rejects unresolved conj tensors, so clear the bit.
        x = x.conj()
    xr = torch.view_as_real(x)
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    if xr.is_contiguous():
        _imag_contig_kernel[lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)](
            xr, out, n
        )
    else:
        _imag_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            xr,
            out,
            n,
            SHAPE=tuple(x.shape),
            STRIDES=tuple(xr.stride())[:-1],
            RANK=x.dim(),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out.view(torch.bool)
