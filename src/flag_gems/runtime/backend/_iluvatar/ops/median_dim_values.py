"""median_dim_values: Triton implementation of the `out`-variant of median along a dim.

Reference semantics (verified on the target against torch.median / aten
median.dim_values, whose CPU implementation is the pytest reference):
  * value = element of rank k = (R - 1) // 2 in the ascending order of the row
    (lower median for even R).
  * index = stable-sort rank semantics: among the elements equal to the median
    value, the (k - #smaller)-th smallest original position along the dim.
    (The suite compares indices exactly only when they are unique, and always
    requires the reported index to select the reported value.)
  * if the row contains NaN, the value is NaN and the index is the *first* NaN
    position along the dim.
  * `indices` is int64, `values` keeps the input dtype, and the caller's buffers
    are filled, resized the way ATen resizes `out=` tensors, and handed back as
    an object exposing `.values` / `.indices`.

Implementation: one program owns BLOCK_OUT independent output slots and the full
reduction axis in registers.  The median is found with an order-preserving
unsigned integer key and a KEY_BITS-step MSB-first bit selection, which yields
both the selected key and the rank of the target inside the tie class; the tie
index then comes from an exclusive prefix sum over the tie mask.  The value is
re-loaded from the input so it is bitwise an input element.
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_BLOCK_R = 16384
_MAX_TILE = 512

# Key width per input dtype: 2-byte dtypes need only 16 bit-selection passes.
_KEY_BITS = {
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.int16: 16,
    torch.int64: 64,
    torch.float64: 64,
}

MedianResult = namedtuple("median", ["values", "indices"])


# --------------------------------------------------------------------------- #
# device helpers
# --------------------------------------------------------------------------- #
@triton.jit
def _key_of(x, KEY_BITS: tl.constexpr):
    """Order-preserving unsigned integer key of the values in `x`.

    The key width is the number of significant bits, so 2-byte dtypes need only
    a 16-step bit selection: an fp16/bf16 pattern is order-equivalent to its
    fp32 widening (the widened trailing mantissa bits are constant zero) and an
    int16 pattern is order-equivalent to its sign-extended int32 value, so the
    16-bit key selects exactly the same element as the 32-bit one.
    """
    if KEY_BITS == 16:
        if x.dtype.is_floating():
            b = x.to(tl.int16, bitcast=True).to(tl.int32)
            k = tl.where(b < 0, ~b, b ^ 0x8000)
        else:
            k = x.to(tl.int32) ^ 0x8000
        return (k & 0xFFFF).to(tl.uint32, bitcast=True)
    elif KEY_BITS == 32:
        if x.dtype.is_floating():
            b = x.to(tl.float32).to(tl.int32, bitcast=True)
            k = tl.where(b < 0, ~b, b ^ (-2147483648))
        else:
            b = x.to(tl.int32)
            k = b ^ (-2147483648)
        return k.to(tl.uint32, bitcast=True)
    else:
        if x.dtype.is_floating():
            b = x.to(tl.int64, bitcast=True)
            k = tl.where(b < 0, ~b, b ^ (-9223372036854775808))
        else:
            b = x.to(tl.int64)
            k = b ^ (-9223372036854775808)
        return k.to(tl.uint64, bitcast=True)


@triton.jit
def _maxkey(KEY_BITS: tl.constexpr):
    if KEY_BITS == 16:
        return tl.full((1, 1), 0xFFFF, tl.uint32)
    elif KEY_BITS == 32:
        return tl.full((1, 1), 0xFFFFFFFF, tl.uint32)
    else:
        return tl.full((1, 1), 0xFFFFFFFFFFFFFFFF, tl.uint64)


@triton.jit
def _rank_select(key, rem, BLOCK_OUT: tl.constexpr, KEY_BITS: tl.constexpr):
    """MSB-first bit selection of the `rem`-th smallest key (per row).

    Returns (selected_key, rem_within_tie_class).
    """
    if KEY_BITS == 64:
        prefix = tl.zeros((BLOCK_OUT,), dtype=tl.uint64)
    else:
        prefix = tl.zeros((BLOCK_OUT,), dtype=tl.uint32)
    for i in tl.static_range(0, KEY_BITS):
        bit = KEY_BITS - 1 - i
        cand = prefix << 1
        cnt0 = tl.sum(((key >> bit) == cand[:, None]).to(tl.int32), axis=1)
        go1 = rem >= cnt0
        rem = tl.where(go1, rem - cnt0, rem)
        prefix = tl.where(go1, cand | 1, cand)
    return prefix, rem


@triton.jit
def _nan_pick(x, valid, rr, BLOCK_R: tl.constexpr):
    """(has_nan, first_nan_index) for a floating-point tile."""
    nanm = valid & (x != x)
    nan_i = nanm.to(tl.int32)
    has_nan = tl.max(nan_i, axis=1) != 0
    first_nan = tl.min(tl.where(nanm, rr[None, :], BLOCK_R), axis=1)
    return has_nan, first_nan


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _median_contig_kernel(
    inp,
    out_val,
    out_idx,
    INNER,
    R,
    OUT,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    """Canonical layout: inp/out_val/out_idx all row-major contiguous."""
    pid = tl.program_id(0)
    slot = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    smask = slot < OUT
    o = slot // INNER
    i = slot % INNER
    base = o * (R * INNER) + i

    rr = tl.arange(0, BLOCK_R)
    valid = smask[:, None] & (rr[None, :] < R)

    x = tl.load(inp + base[:, None] + rr[None, :] * INNER, mask=valid, other=0)

    if x.dtype.is_floating():
        has_nan, first_nan = _nan_pick(x, valid, rr, BLOCK_R)

    key = tl.where(valid, _key_of(x, KEY_BITS), _maxkey(KEY_BITS))

    rem0 = tl.zeros((BLOCK_OUT,), dtype=tl.int32) + ((R - 1) // 2)
    sel, rem = _rank_select(key, rem0, BLOCK_OUT, KEY_BITS)

    eq = valid & (key == sel[:, None])
    eqi = eq.to(tl.int32)
    pref = tl.cumsum(eqi, axis=1) - eqi
    idx = tl.min(tl.where(eq & (pref == rem[:, None]), rr[None, :], BLOCK_R), axis=1)

    if x.dtype.is_floating():
        idx = tl.where(has_nan, first_nan, idx)

    val = tl.load(inp + base + idx * INNER, mask=smask, other=0)

    tl.store(out_val + slot, val, mask=smask)
    tl.store(out_idx + slot, idx.to(tl.int64), mask=smask)


@triton.jit
def _median_dim_kernel(
    inp,
    out_val,
    out_idx,
    INNER,
    R,
    OUT,
    s_in_o,
    s_in_r,
    s_in_i,
    s_val_o,
    s_val_i,
    s_idx_o,
    s_idx_i,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    KEY_BITS: tl.constexpr,
    WIDE: tl.constexpr,
):
    """General strides: offset = outer_index * s_o + inner_index * s_i."""
    pid = tl.program_id(0)
    slot = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    smask = slot < OUT
    o = slot // INNER
    i = slot % INNER
    base = o * s_in_o + i * s_in_i
    if WIDE:
        base = base.to(tl.int64)

    rr = tl.arange(0, BLOCK_R)
    valid = smask[:, None] & (rr[None, :] < R)

    x = tl.load(inp + base[:, None] + rr[None, :] * s_in_r, mask=valid, other=0)

    if x.dtype.is_floating():
        has_nan, first_nan = _nan_pick(x, valid, rr, BLOCK_R)

    key = tl.where(valid, _key_of(x, KEY_BITS), _maxkey(KEY_BITS))

    rem0 = tl.zeros((BLOCK_OUT,), dtype=tl.int32) + ((R - 1) // 2)
    sel, rem = _rank_select(key, rem0, BLOCK_OUT, KEY_BITS)

    eq = valid & (key == sel[:, None])
    eqi = eq.to(tl.int32)
    pref = tl.cumsum(eqi, axis=1) - eqi
    idx = tl.min(tl.where(eq & (pref == rem[:, None]), rr[None, :], BLOCK_R), axis=1)

    if x.dtype.is_floating():
        idx = tl.where(has_nan, first_nan, idx)

    val = tl.load(inp + base + idx * s_in_r, mask=smask, other=0)

    tl.store(out_val + o * s_val_o + i * s_val_i, val, mask=smask)
    tl.store(out_idx + o * s_idx_o + i * s_idx_i, idx.to(tl.int64), mask=smask)


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return p


def _two_strides(t, shape_list, pre_len, post_start):
    """(S_o, S_i) such that flat offset == outer_index * S_o + inner_index * S_i.

    outer_index enumerates shape_list[:pre_len] row-major, inner_index
    enumerates shape_list[post_start:] row-major.  Returns None when the tensor
    layout cannot be expressed that way (non-collapsible strides).
    """
    st = [int(s) for s in t.stride()]
    nd = len(shape_list)
    if len(st) != nd:
        return None
    ok = True
    s_o = st[pre_len - 1] if pre_len > 0 else 1
    for q in range(pre_len - 2, -1, -1):
        if st[q] != shape_list[q + 1] * st[q + 1]:
            ok = False
    s_i = st[nd - 1] if post_start < nd else 1
    acc = 1
    for q in range(nd - 1, post_start - 1, -1):
        if st[q] != acc * s_i:
            ok = False
        acc *= shape_list[q]
    return (s_o, s_i) if ok else None


def _tile_shape(R, inner):
    block_r = max(1, triton.next_power_of_2(R))
    if block_r > _MAX_BLOCK_R:
        raise NotImplementedError(
            "median_dim_values(): reduction length {} exceeds the supported "
            "limit of {}".format(R, _MAX_BLOCK_R)
        )
    block_out = 8
    while block_out > 1 and block_out * block_r > _MAX_TILE:
        block_out //= 2
    elems = block_out * block_r
    if inner > 1 and elems <= 1024:
        # When the reduction axis is not innermost, the unit-stride direction is
        # the output-slot axis, so the compiler spreads the reduction axis over
        # warps and pays a shared-memory cross-warp reduction (barriers) on
        # every bit-selection pass.  Keeping the block to one warp forces that
        # reduction to stay intra-warp.
        nw = 1
    elif elems <= 64:
        nw = 1
    elif elems <= 512:
        nw = 2
    elif elems <= 2048:
        nw = 4
    else:
        nw = 8
    return block_out, block_r, nw


def median_dim_values(
    inp, dim=0, keepdim=False, values=None, indices=None, out=None, **kwargs
):
    logger.debug("GEMS_ILUVATAR MEDIAN_DIM_VALUES")
    if out is not None and values is None and indices is None:
        values, indices = out[0], out[1]
    if not isinstance(inp, torch.Tensor):
        raise TypeError("median_dim_values(): inp must be a torch.Tensor")

    # These shapes are tiny, so per-call host work is on the critical path:
    # keep the metadata math to tuple slicing and single attribute reads.
    shape = inp.shape
    nd0 = len(shape)
    if nd0 == 0:
        work = inp.unsqueeze(0)
        shape = (1,)
        split = 0
    else:
        work = inp
        split = int(dim)
        if split < 0:
            split += nd0
        if split < 0 or split >= nd0:
            raise IndexError(
                "median_dim_values(): Dimension out of range (expected to be in "
                "range of [{}, {}], but got {})".format(-nd0, nd0 - 1, dim)
            )

    R = shape[split]
    if R == 0:
        raise IndexError(
            "median(): Expected reduction dim {} to have non-zero size.".format(split)
        )
    nelem = work.numel()
    total = nelem // R
    contig = work.is_contiguous()
    inner = work.stride(split) if contig else _prod(shape[split + 1 :])
    out_shape = shape[:split] + ((1,) if keepdim else ()) + shape[split + 1 :]
    dev = work.device
    dtype = work.dtype

    # Out buffers: validate like ATen (device / dtype -> RuntimeError) and resize
    # the way the out= variants do when the caller passed a differently shaped
    # (e.g. single-element) buffer.
    if values is None:
        values = torch.empty(out_shape, dtype=dtype, device=dev)
    else:
        if values.device != dev:
            raise RuntimeError(
                "median_dim_values(): Expected all tensors to be on the same "
                "device, but found at least two devices, {} and {}!".format(
                    values.device, dev
                )
            )
        if values.dtype != dtype:
            raise RuntimeError(
                "median_dim_values(): expected out tensor `values` to have dtype "
                "{}, but got {}".format(dtype, values.dtype)
            )
        if values.shape != out_shape:
            values.resize_(out_shape)
    if indices is None:
        indices = torch.empty(out_shape, dtype=torch.int64, device=dev)
    else:
        if indices.device != dev:
            raise RuntimeError(
                "median_dim_values(): Expected all tensors to be on the same "
                "device, but found at least two devices, {} and {}!".format(
                    indices.device, dev
                )
            )
        if indices.dtype != torch.int64:
            raise RuntimeError(
                "median_dim_values(): expected out tensor `indices` to have dtype "
                "torch.int64, but got {}".format(indices.dtype)
            )
        if indices.shape != out_shape:
            indices.resize_(out_shape)

    if total == 0:
        return MedianResult(values, indices)

    key_bits = _KEY_BITS.get(dtype, 32)
    block_out, block_r, nw = _tile_shape(R, inner)
    grid = ((total + block_out - 1) // block_out,)

    if contig and nelem < 2**31 and values.is_contiguous() and indices.is_contiguous():
        _median_contig_kernel[grid](
            work,
            values,
            indices,
            inner,
            R,
            total,
            BLOCK_OUT=block_out,
            BLOCK_R=block_r,
            KEY_BITS=key_bits,
            num_warps=nw,
        )
        return MedianResult(values, indices)

    in_pair = _two_strides(work, shape, split, split + 1)
    if in_pair is None:
        raise NotImplementedError(
            "median_dim_values(): non-collapsible input strides are not supported"
        )
    post_start = split + 1 if keepdim else split
    val_pair = _two_strides(values, out_shape, split, post_start)
    idx_pair = _two_strides(indices, out_shape, split, post_start)
    if val_pair is None or idx_pair is None:
        raise NotImplementedError(
            "median_dim_values(): non-collapsible output strides are not supported"
        )
    _median_dim_kernel[grid](
        work,
        values,
        indices,
        inner,
        R,
        total,
        in_pair[0],
        int(work.stride(split)),
        in_pair[1],
        val_pair[0],
        val_pair[1],
        idx_pair[0],
        idx_pair[1],
        BLOCK_OUT=block_out,
        BLOCK_R=block_r,
        KEY_BITS=key_bits,
        WIDE=nelem >= 2**31,
        num_warps=nw,
    )
    return MedianResult(values, indices)
