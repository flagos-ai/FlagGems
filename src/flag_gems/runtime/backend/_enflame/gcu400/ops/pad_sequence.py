import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_NUM_CTAS = 96
_MAX_GROUP = 32
_SLAB_SMALL = 8192
_TILE_LANES = 16384


@triton.jit
def _pad_sequence_slab_kernel(
    Out,
    Sources,
    Lengths,
    TilesPerSeq,
    SlabSize,
    Feature,
    GroupCount,
    BatchOffset,
    PaddingValue,
    CHUNK: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    """batch_first=True: copy each sequence slab into Out[b] with flat chunks."""
    pid = tl.program_id(0)
    total = TilesPerSeq * GroupCount
    offs = tl.arange(0, CHUNK)
    for item in range(pid, total, NumCTAs):
        b = item // TilesPerSeq
        tile = item % TilesPerSeq
        src = Sources[0]
        length = Lengths[0]
        for i in tl.static_range(1, len(Sources)):
            src = tl.where(b == i, Sources[i], src)
            length = tl.where(b == i, Lengths[i], length)
        if INDEX64:
            base = (BatchOffset + b).to(tl.int64) * SlabSize.to(tl.int64)
            soffs = tile.to(tl.int64) * CHUNK + offs.to(tl.int64)
            valid_end = length.to(tl.int64) * Feature.to(tl.int64)
            load_end = (tile + 1) * CHUNK <= length * Feature
            slab_end = (tile + 1) * CHUNK <= SlabSize
        else:
            base = (BatchOffset + b) * SlabSize
            soffs = tile * CHUNK + offs
            valid_end = length * Feature
            load_end = (tile + 1) * CHUNK <= length * Feature
            slab_end = (tile + 1) * CHUNK <= SlabSize
        if load_end & slab_end:
            value = tl.load(src + soffs)
            tl.store(Out + base + soffs, value)
        elif (tile * CHUNK >= length * Feature) & slab_end:
            value = tl.full((CHUNK,), PaddingValue, Out.dtype.element_ty)
            tl.store(Out + base + soffs, value)
        else:
            value = tl.load(src + soffs, mask=soffs < valid_end, other=PaddingValue)
            tl.store(Out + base + soffs, value, mask=soffs < SlabSize)


@triton.jit
def _pad_sequence_tiled_kernel(
    Out,
    Sources,
    Lengths,
    RowTiles,
    FTiles,
    MaxLength,
    Feature,
    Batch,
    GroupCount,
    BatchOffset,
    PaddingValue,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    """batch_first=False: 2D row tiles, f-contiguous, scalar row bases."""
    pid = tl.program_id(0)
    tiles_per_seq = RowTiles * FTiles
    total = tiles_per_seq * GroupCount
    r = tl.arange(0, BLOCK_R)
    f = tl.arange(0, BLOCK_F)
    for item in range(pid, total, NumCTAs):
        b = item // tiles_per_seq
        rem = item % tiles_per_seq
        rt = rem // FTiles
        ft = rem % FTiles
        src = Sources[0]
        length = Lengths[0]
        for i in tl.static_range(1, len(Sources)):
            src = tl.where(b == i, Sources[i], src)
            length = tl.where(b == i, Lengths[i], length)
        t = rt * BLOCK_R + r
        ff = ft * BLOCK_F + f
        if INDEX64:
            t64 = t.to(tl.int64)
            f64 = ff.to(tl.int64)
            feature64 = Feature.to(tl.int64)
            sbase = t64[:, None] * feature64 + f64[None, :]
            dst = (
                t64[:, None] * (Batch.to(tl.int64) * feature64)
                + (BatchOffset + b).to(tl.int64) * feature64
                + f64[None, :]
            )
        else:
            sbase = t[:, None] * Feature + ff[None, :]
            dst = (
                t[:, None] * (Batch * Feature)
                + (BatchOffset + b) * Feature
                + ff[None, :]
            )
        row_in = (rt + 1) * BLOCK_R <= length
        f_in = (ft + 1) * BLOCK_F <= Feature
        if row_in & f_in:
            value = tl.load(src + sbase)
            tl.store(Out + dst, value)
        elif rt * BLOCK_R >= length:
            # The whole tile sits in the padding region: no source read is needed, so
            # store the constant directly and save one masked load.
            fill = tl.full((BLOCK_R, BLOCK_F), PaddingValue, Out.dtype.element_ty)
            tl.store(
                Out + dst,
                fill,
                mask=(t < MaxLength)[:, None] & (ff < Feature)[None, :],
            )
        else:
            ff_mask = ff < Feature
            value = tl.load(
                src + sbase,
                mask=(t < length)[:, None] & ff_mask[None, :],
                other=PaddingValue,
            )
            tl.store(
                Out + dst,
                value,
                mask=(t < MaxLength)[:, None] & ff_mask[None, :],
            )


@triton.jit
def _pad_sequence_slab_w4(
    Out,
    S0,
    S1,
    S2,
    S3,
    L0,
    L1,
    L2,
    L3,
    TilesPerSeq,
    SlabSize,
    Feature,
    GroupCount,
    BatchOffset,
    PaddingValue,
    CHUNK: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_slab_kernel(
        Out,
        (S0, S1, S2, S3),
        (L0, L1, L2, L3),
        TilesPerSeq,
        SlabSize,
        Feature,
        GroupCount,
        BatchOffset,
        PaddingValue,
        CHUNK,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_slab_w8(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    TilesPerSeq,
    SlabSize,
    Feature,
    GroupCount,
    BatchOffset,
    PaddingValue,
    CHUNK: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_slab_kernel(
        Out,
        (S0, S1, S2, S3, S4, S5, S6, S7),
        (L0, L1, L2, L3, L4, L5, L6, L7),
        TilesPerSeq,
        SlabSize,
        Feature,
        GroupCount,
        BatchOffset,
        PaddingValue,
        CHUNK,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_slab_w16(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    TilesPerSeq,
    SlabSize,
    Feature,
    GroupCount,
    BatchOffset,
    PaddingValue,
    CHUNK: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_slab_kernel(
        Out,
        (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15),
        (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15),
        TilesPerSeq,
        SlabSize,
        Feature,
        GroupCount,
        BatchOffset,
        PaddingValue,
        CHUNK,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_slab_w32(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
    S17,
    S18,
    S19,
    S20,
    S21,
    S22,
    S23,
    S24,
    S25,
    S26,
    S27,
    S28,
    S29,
    S30,
    S31,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    L16,
    L17,
    L18,
    L19,
    L20,
    L21,
    L22,
    L23,
    L24,
    L25,
    L26,
    L27,
    L28,
    L29,
    L30,
    L31,
    TilesPerSeq,
    SlabSize,
    Feature,
    GroupCount,
    BatchOffset,
    PaddingValue,
    CHUNK: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_slab_kernel(
        Out,
        (
            S0,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S7,
            S8,
            S9,
            S10,
            S11,
            S12,
            S13,
            S14,
            S15,
            S16,
            S17,
            S18,
            S19,
            S20,
            S21,
            S22,
            S23,
            S24,
            S25,
            S26,
            S27,
            S28,
            S29,
            S30,
            S31,
        ),
        (
            L0,
            L1,
            L2,
            L3,
            L4,
            L5,
            L6,
            L7,
            L8,
            L9,
            L10,
            L11,
            L12,
            L13,
            L14,
            L15,
            L16,
            L17,
            L18,
            L19,
            L20,
            L21,
            L22,
            L23,
            L24,
            L25,
            L26,
            L27,
            L28,
            L29,
            L30,
            L31,
        ),
        TilesPerSeq,
        SlabSize,
        Feature,
        GroupCount,
        BatchOffset,
        PaddingValue,
        CHUNK,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_tiled_w4(
    Out,
    S0,
    S1,
    S2,
    S3,
    L0,
    L1,
    L2,
    L3,
    RowTiles,
    FTiles,
    MaxLength,
    Feature,
    Batch,
    GroupCount,
    BatchOffset,
    PaddingValue,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_tiled_kernel(
        Out,
        (S0, S1, S2, S3),
        (L0, L1, L2, L3),
        RowTiles,
        FTiles,
        MaxLength,
        Feature,
        Batch,
        GroupCount,
        BatchOffset,
        PaddingValue,
        BLOCK_R,
        BLOCK_F,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_tiled_w8(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    RowTiles,
    FTiles,
    MaxLength,
    Feature,
    Batch,
    GroupCount,
    BatchOffset,
    PaddingValue,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_tiled_kernel(
        Out,
        (S0, S1, S2, S3, S4, S5, S6, S7),
        (L0, L1, L2, L3, L4, L5, L6, L7),
        RowTiles,
        FTiles,
        MaxLength,
        Feature,
        Batch,
        GroupCount,
        BatchOffset,
        PaddingValue,
        BLOCK_R,
        BLOCK_F,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_tiled_w16(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    RowTiles,
    FTiles,
    MaxLength,
    Feature,
    Batch,
    GroupCount,
    BatchOffset,
    PaddingValue,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_tiled_kernel(
        Out,
        (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15),
        (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15),
        RowTiles,
        FTiles,
        MaxLength,
        Feature,
        Batch,
        GroupCount,
        BatchOffset,
        PaddingValue,
        BLOCK_R,
        BLOCK_F,
        NumCTAs,
        INDEX64,
    )


@triton.jit
def _pad_sequence_tiled_w32(
    Out,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
    S17,
    S18,
    S19,
    S20,
    S21,
    S22,
    S23,
    S24,
    S25,
    S26,
    S27,
    S28,
    S29,
    S30,
    S31,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    L16,
    L17,
    L18,
    L19,
    L20,
    L21,
    L22,
    L23,
    L24,
    L25,
    L26,
    L27,
    L28,
    L29,
    L30,
    L31,
    RowTiles,
    FTiles,
    MaxLength,
    Feature,
    Batch,
    GroupCount,
    BatchOffset,
    PaddingValue,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NumCTAs,
    INDEX64: tl.constexpr,
):
    _pad_sequence_tiled_kernel(
        Out,
        (
            S0,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S7,
            S8,
            S9,
            S10,
            S11,
            S12,
            S13,
            S14,
            S15,
            S16,
            S17,
            S18,
            S19,
            S20,
            S21,
            S22,
            S23,
            S24,
            S25,
            S26,
            S27,
            S28,
            S29,
            S30,
            S31,
        ),
        (
            L0,
            L1,
            L2,
            L3,
            L4,
            L5,
            L6,
            L7,
            L8,
            L9,
            L10,
            L11,
            L12,
            L13,
            L14,
            L15,
            L16,
            L17,
            L18,
            L19,
            L20,
            L21,
            L22,
            L23,
            L24,
            L25,
            L26,
            L27,
            L28,
            L29,
            L30,
            L31,
        ),
        RowTiles,
        FTiles,
        MaxLength,
        Feature,
        Batch,
        GroupCount,
        BatchOffset,
        PaddingValue,
        BLOCK_R,
        BLOCK_F,
        NumCTAs,
        INDEX64,
    )


_SLAB_KERNELS = {
    4: _pad_sequence_slab_w4,
    8: _pad_sequence_slab_w8,
    16: _pad_sequence_slab_w16,
    32: _pad_sequence_slab_w32,
}

_TILED_KERNELS = {
    4: _pad_sequence_tiled_w4,
    8: _pad_sequence_tiled_w8,
    16: _pad_sequence_tiled_w16,
    32: _pad_sequence_tiled_w32,
}


def _group_width(batch):
    width = 4
    while width < batch and width < _MAX_GROUP:
        width *= 2
    return width


def _slab_chunk(slab, item_size):
    if slab <= _SLAB_SMALL:
        return 2048
    if item_size >= 4:
        return 8192
    return 16384


def _tile_shape(feature, item_size):
    block_f = min(4096, triton.next_power_of_2(max(1, feature)))
    block_r = max(4, min(64, _TILE_LANES // block_f))
    if item_size < 4 and block_f >= 4096:
        block_r = min(64, block_r * 2)
    return block_r, block_f


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Pad variable length tensors into a single batch tensor."""
    logger.debug("GEMS_ENFLAME PAD_SEQUENCE")

    batch = len(sequences)
    if batch == 0:
        raise RuntimeError("pad_sequence empty input")

    first = sequences[0]
    first_shape = first.shape
    if len(first_shape) == 0:
        raise RuntimeError("pad_sequence requires at least one dimension")
    device = first.device
    dtype = first.dtype
    trailing_shape = first_shape[1:]
    max_len = first_shape[0]
    seqs = [first if first.is_contiguous() else first.contiguous()]

    for sequence in sequences[1:]:
        shape = sequence.shape
        if len(shape) == 0:
            raise RuntimeError("pad_sequence requires at least one dimension")
        if shape[1:] != trailing_shape:
            raise RuntimeError("pad_sequence expects matching trailing dimensions")
        if sequence.device != device:
            raise RuntimeError(
                "pad_sequence expects all input tensors to be on the same device"
            )
        # Match native output dtype and the Hygon backend. Homogeneous inputs
        # do not need an additional conversion kernel.
        if sequence.dtype != dtype:
            sequence = sequence.to(dtype=dtype)
        if not sequence.is_contiguous():
            sequence = sequence.contiguous()
        seqs.append(sequence)
        max_len = max(max_len, shape[0])

    feature = 1
    for d in trailing_shape:
        feature *= d

    if batch_first:
        out_shape = (batch, max_len, *trailing_shape)
    else:
        out_shape = (max_len, batch, *trailing_shape)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    total_elements = batch * max_len * feature
    if total_elements == 0:
        return out

    item_size = out.element_size()
    index64 = total_elements >= 2**31
    width = _group_width(batch)
    num_warps = 4
    with torch_device_fn.device(device):
        for start in range(0, batch, width):
            group = seqs[start : start + width]
            count = len(group)
            ptrs = list(group)
            lens = [seq.shape[0] for seq in group]
            ptrs += [group[0]] * (width - count)
            lens += [0] * (width - count)
            if batch_first:
                slab = max_len * feature
                chunk = _slab_chunk(slab, item_size)
                tiles_per_seq = triton.cdiv(slab, chunk)
                grid = (min(_NUM_CTAS, tiles_per_seq * count),)
                _SLAB_KERNELS[width][grid](
                    out,
                    *ptrs,
                    *lens,
                    tiles_per_seq,
                    slab,
                    feature,
                    count,
                    start,
                    padding_value,
                    CHUNK=chunk,
                    NumCTAs=grid[0],
                    INDEX64=index64,
                    num_warps=num_warps,
                )
            else:
                block_r, block_f = _tile_shape(feature, item_size)
                row_tiles = triton.cdiv(max_len, block_r)
                ftiles = triton.cdiv(feature, block_f)
                grid = (min(_NUM_CTAS, row_tiles * ftiles * count),)
                _TILED_KERNELS[width][grid](
                    out,
                    *ptrs,
                    *lens,
                    row_tiles,
                    ftiles,
                    max_len,
                    feature,
                    batch,
                    count,
                    start,
                    padding_value,
                    BLOCK_R=block_r,
                    BLOCK_F=block_f,
                    NumCTAs=grid[0],
                    INDEX64=index64,
                    num_warps=num_warps,
                )

    return out
