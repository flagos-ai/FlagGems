import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.fill import fill_scalar_
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_ASCEND = tl.constexpr(runtime_device.vendor_name == "ascend")
_LEGACY_ASCEND = (
    runtime_device.vendor_name == "ascend" and triton.__version__.startswith("3.2.")
)


@triton.jit
def _spatial_offset(
    index,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    return index // (D1 * D2) * S0 + (index // D2 % D1) * S1 + (index % D2) * S2


@libentry()
@triton.jit
def _input_gradient_dot(
    GO,
    W,
    DX,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    PROGRAM0=None,
    PROGRAM1=None,
):
    # A tile reuses grad_output across input channels and weights across space.
    spatial = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0).to(
        INDEX
    ) * BM + tl.arange(0, BM)
    tile = (tl.program_id(1) if PROGRAM1 is None else PROGRAM1).to(INDEX)
    channel_tiles: tl.constexpr = (CI // GROUPS + 31) // 32
    group = tile // channel_tiles % GROUPS
    batch = tile // (channel_tiles * GROUPS)
    ci = (tile % channel_tiles) * 32 + tl.arange(0, 32)
    acc = tl.full((BM, 32), 0, tl.float32)
    for kernel in range(KS):
        spos = spatial
        kpos = kernel
        go_base = tl.full((BM,), batch * GS0, INDEX)
        valid = spatial < IS
        wo = tl.full((), 0, INDEX)
        for axis in tl.static_range(2, -1, -1):
            ip = spos % (I0 if axis == 0 else I1 if axis == 1 else I2)
            kp = kpos % (K0 if axis == 0 else K1 if axis == 1 else K2)
            if TRANSPOSED:
                op = (
                    ip * (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                    - (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                    + kp * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
                )
            else:
                raw = (
                    ip
                    + (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                    - kp * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
                )
                valid &= (
                    raw % (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                    == 0
                )
                op = raw // (
                    STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2
                )
            valid &= (op >= 0) & (
                op
                < (
                    OUTPUT_SHAPE0
                    if axis == 0
                    else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
                )
            )
            go_base += op * (GS2 if axis == 0 else GS3 if axis == 1 else GS4)
            wo += kp * (WS2 if axis == 0 else WS3 if axis == 1 else WS4)
            spos = spos // (I0 if axis == 0 else I1 if axis == 1 else I2)
            kpos = kpos // (K0 if axis == 0 else K1 if axis == 1 else K2)
        for start in range(0, CO // GROUPS, BR):
            co = start + tl.arange(0, BR)
            go_offset = go_base[:, None] + (group * (CO // GROUPS) + co[None, :]) * GS1
            if TRANSPOSED:
                weight_offset = (
                    (group * (CI // GROUPS) + ci[None, :]) * WS0
                    + co[:, None] * WS1
                    + wo
                )
            else:
                weight_offset = (
                    (group * (CO // GROUPS) + co[:, None]) * WS0
                    + ci[None, :] * WS1
                    + wo
                )
            if _ASCEND and (
                (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
                | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
                | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
            ):
                go_offset = tl.where(
                    valid[:, None] & (co[None, :] < CO // GROUPS), go_offset, 0
                )
                weight_offset = tl.where(
                    (co[:, None] < CO // GROUPS) & (ci[None, :] < CI // GROUPS),
                    weight_offset,
                    0,
                )
            go = tl.load(
                GO + go_offset, valid[:, None] & (co[None, :] < CO // GROUPS), 0
            )
            w = tl.load(
                W + weight_offset,
                (co[:, None] < CO // GROUPS) & (ci[None, :] < CI // GROUPS),
                0,
            )
            acc += tl.dot(go, w, allow_tf32=False)
    dest = batch * DS0 + (group * (CI // GROUPS) + ci[None, :]) * DS1
    dest = dest + _spatial_offset(spatial, I0, I1, I2, DS2, DS3, DS4)[:, None]
    tl.store(DX + dest, acc, (spatial[:, None] < IS) & (ci[None, :] < CI // GROUPS))


@triton.jit
def _divide_spatial_index(value, DIVISOR: tl.constexpr):
    # Packing is bounded to 32 MiB, hence every spatial index is below 2**24.
    # FP32 estimates can differ by at most one; integer corrections are exact.
    quotient = (value.to(tl.float32) * (1.0 / DIVISOR)).to(tl.int32)
    quotient -= (quotient * DIVISOR > value).to(tl.int32)
    quotient += ((quotient + 1) * DIVISOR <= value).to(tl.int32)
    return quotient


@libentry()
@triton.jit
def _pad_pack_rows(
    X,
    PAD,
    S: tl.constexpr,
    PITCH: tl.constexpr,
    MARGIN: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    row = tl.program_id(1).to(INDEX)
    start = tl.program_id(0).to(INDEX) * BLOCK
    r = start + tl.arange(0, BLOCK)
    # The data and two zero intervals are disjoint and use nonnegative bases.
    if start < S:
        value = tl.load(X + row * S + r, r < S, 0)
        tl.store(PAD + row * PITCH + MARGIN + r, value, r < S)
    tl.store(PAD + row * PITCH + r, 0, r < MARGIN)
    tail = MARGIN + S + r
    tl.store(PAD + row * PITCH + tail, 0, tail < PITCH)


def _padded_pack_source(tensor, kernel, padding, dilation, index):
    shape = (1,) * (5 - tensor.ndim) + tuple(tensor.shape[2:])
    s = math.prod(shape)
    margin = sum(
        max(p, (k - 1) * d - p) * step
        for k, p, d, step in zip(
            kernel, padding, dilation, (shape[1] * shape[2], shape[2], 1)
        )
    )
    pitch = triton.cdiv(triton.cdiv(s, 1024) * 1024 + 2 * margin, 16) * 16
    padded = torch.empty(
        (tensor.shape[0], tensor.shape[1], pitch),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    _pad_pack_rows[
        (
            triton.cdiv(max(s, margin, pitch - margin - s), 1024),
            tensor.shape[0] * tensor.shape[1],
        )
    ](tensor, padded, s, pitch, margin, index)
    return padded, margin, pitch


@libentry()
@triton.jit
def _pack_spatial_rows(
    SRC,
    OTHER,
    DEST,
    AUX,
    N: tl.constexpr,
    C: tl.constexpr,
    CT: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    INPUT_GRAD: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INDEX: tl.constexpr,
    BCI: tl.constexpr,
    MARGIN: tl.constexpr = 0,
    PITCH: tl.constexpr = 0,
    ROWS: tl.constexpr = 8,
    BLOCK: tl.constexpr = 1024,
):
    spatial_size: tl.constexpr = I0 * I1 * I2
    ks: tl.constexpr = K0 * K1 * K2
    start = tl.program_id(0).to(INDEX) * BLOCK
    kernel = tl.program_id(1).to(INDEX) % ks
    batch = tl.program_id(1).to(INDEX) // ks
    channel_start = tl.program_id(2).to(INDEX) * ROWS
    spatial = start + tl.arange(0, BLOCK)
    k0, k1, k2 = kernel // (K1 * K2), kernel // K2 % K1, kernel % K2
    d0, d1, d2 = k0 * D0 - P0, k1 * D1 - P1, k2 * D2 - P2
    if INPUT_GRAD and not TRANSPOSED:
        d0, d1, d2 = -d0, -d1, -d2
    plane = _divide_spatial_index(spatial, I1 * I2).to(INDEX)
    line = _divide_spatial_index(spatial, I2).to(INDEX)
    q0 = plane + d0
    q1 = line - plane * I1 + d1
    q2 = spatial - line * I2 + d2
    valid = (spatial < spatial_size) & (q0 >= 0) & (q0 < I0)
    valid &= (q1 >= 0) & (q1 < I1) & (q2 >= 0) & (q2 < I2)
    delta = d0 * I1 * I2 + d1 * I2 + d2
    if channel_start < C:
        source = tl.make_block_ptr(
            SRC + batch * C * PITCH,
            shape=(C, PITCH),
            strides=(PITCH, 1),
            offsets=(channel_start, start + MARGIN + delta),
            block_shape=(ROWS, BLOCK),
            order=(1, 0),
        )
        value = tl.load(source, boundary_check=(0,), padding_option="zero")
        value = tl.where(valid[None, :], value, 0)
        destination = tl.make_block_ptr(
            base=DEST + kernel * N * spatial_size + batch * spatial_size,
            shape=(C, spatial_size),
            strides=(ks * N * spatial_size, 1),
            offsets=(channel_start, start),
            block_shape=(ROWS, BLOCK),
            order=(1, 0),
        )
        tl.store(destination, value, boundary_check=(0, 1))


@libentry()
@triton.jit
def _pack_input_weights(
    W,
    PW,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr = 512,
):
    ks: tl.constexpr = K0 * K1 * K2
    reduction: tl.constexpr = CO // GROUPS * ks
    flat = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    ci = _divide_spatial_index(flat, reduction).to(INDEX)
    rem = flat - ci * reduction
    co = _divide_spatial_index(rem, ks).to(INDEX)
    kernel = rem - co * ks
    group = ci // (CI // GROUPS)
    if TRANSPOSED:
        base = ci * WS0 + co * WS1
    else:
        base = (group * (CO // GROUPS) + co) * WS0 + (ci % (CI // GROUPS)) * WS1
    if WS4 == 1 and (K1 == 1 or WS3 == K2) and (K0 == 1 or WS2 == K1 * K2):
        offset = kernel
    else:
        offset = _spatial_offset(kernel, K0, K1, K2, WS2, WS3, WS4)
    value = tl.load(W + base + offset, flat < CI * reduction, 0)
    tl.store(PW + flat, value, flat < CI * reduction)


@libentry()
@triton.jit
def _pack_output_rows(
    Y,
    PY,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    row = tl.program_id(1).to(INDEX)
    batch = row // C
    channel = row % C
    spatial = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Y + row * S + spatial, spatial < S, 0)
    tl.store(PY + channel * N * S + batch * S + spatial, value, spatial < S)


@libentry()
@triton.jit
def _pack_input_operands(
    GO,
    W,
    PATCH,
    PW,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
    BCI: tl.constexpr,
    ROWS: tl.constexpr = 4,
):
    spatial = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    row_tiles: tl.constexpr = (CO * KS + ROWS - 1) // ROWS
    row = (tl.program_id(1).to(INDEX) % row_tiles * ROWS + tl.arange(0, ROWS))[:, None]
    batch = tl.program_id(1).to(INDEX) // row_tiles
    spatial = spatial[None, :]
    group = row // (CO // GROUPS * KS)
    co = row // KS % (CO // GROUPS)
    kernel = row % KS
    go_offset = batch * GS0 + (group * (CO // GROUPS) + co) * GS1
    valid = (spatial < IS) & (row < CO * KS)
    delta = tl.full((), 0, INDEX)
    weight_offset = tl.full((), 0, INDEX)
    spos = spatial
    kpos = kernel
    for axis in tl.static_range(2, -1, -1):
        size = I0 if axis == 0 else I1 if axis == 1 else I2
        osize = O0 if axis == 0 else O1 if axis == 1 else O2
        ksize = K0 if axis == 0 else K1 if axis == 1 else K2
        step = S0 if axis == 0 else S1 if axis == 1 else S2
        pad = P0 if axis == 0 else P1 if axis == 1 else P2
        dil = D0 if axis == 0 else D1 if axis == 1 else D2
        gs = GS2 if axis == 0 else GS3 if axis == 1 else GS4
        ws = WS2 if axis == 0 else WS3 if axis == 1 else WS4
        ip = spos % size
        kp = kpos % ksize
        if TRANSPOSED:
            op = ip * step - pad + kp * dil
            delta += (kp * dil - pad) * gs
        else:
            raw = ip + pad - kp * dil
            valid &= raw % step == 0
            op = raw // step
            delta += (pad - kp * dil) * gs
        valid &= (op >= 0) & (op < osize)
        go_offset += op * gs
        weight_offset += kp * ws
        spos = spos // size
        kpos = kpos // ksize
    if (
        S0 == 1
        and S1 == 1
        and S2 == 1
        and I0 == O0
        and I1 == O1
        and I2 == O2
        and GS4 == 1
        and (O1 == 1 or GS3 == O2)
        and (O0 == 1 or GS2 == O1 * O2)
    ):
        go_offset = batch * GS0 + (group * (CO // GROUPS) + co) * GS1 + spatial + delta
    if (P0 + D0 > I0 + O0) | (P1 + D1 > I1 + O1) | (P2 + D2 > I2 + O2):
        value = tl.load(GO + tl.where(valid, go_offset, 0))
        value = tl.where(valid, value, 0)
    elif (
        GS0 == CO * OS
        and GS1 == OS
        and GS4 == 1
        and (O1 == 1 or GS3 == O2)
        and (O0 == 1 or GS2 == O1 * O2)
    ):
        value = tl.load(GO + go_offset, (go_offset >= 0) & (go_offset < N * CO * OS), 0)
        value = tl.where(valid, value, 0)
    else:
        value = tl.load(GO + go_offset, valid, 0)
    dest_block = tl.make_block_ptr(
        PATCH + batch * IS,
        shape=(CO * KS, IS),
        strides=(N * IS, 1),
        offsets=(tl.program_id(1) % row_tiles * ROWS, tl.program_id(0) * BLOCK),
        block_shape=(ROWS, BLOCK),
        order=(1, 0),
    )
    tl.store(dest_block, value, boundary_check=(0, 1))
    if batch == 0 and tl.program_id(0) == 0:
        ci = tl.arange(0, BCI).to(INDEX)[None, :]
        if TRANSPOSED:
            wo = (group * (CI // GROUPS) + ci) * WS0 + co * WS1 + weight_offset
        else:
            wo = (group * (CO // GROUPS) + co) * WS0 + ci * WS1 + weight_offset
        weights = tl.load(W + wo, (ci < CI // GROUPS) & (row < CO * KS), 0)
        tl.store(
            PW + (group * (CI // GROUPS) + ci) * (CO // GROUPS * KS) + co * KS + kernel,
            weights,
            (ci < CI // GROUPS) & (row < CO * KS),
        )


@libentry()
@triton.jit
def _input_gradient_packed(
    PATCH,
    PW,
    DX,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    IS: tl.constexpr,
    KS: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 64,
):
    spatial = tl.program_id(0).to(INDEX) * BN + tl.arange(0, BN)
    tile = tl.program_id(1).to(INDEX)
    ctiles: tl.constexpr = (CI // GROUPS + BM - 1) // BM
    group = tile // ctiles % GROUPS
    batch = tile // (ctiles * GROUPS)
    ci = tile % ctiles * BM + tl.arange(0, BM)
    reduction: tl.constexpr = CO // GROUPS * KS
    weight_block = tl.make_block_ptr(
        PW + group * (CI // GROUPS) * reduction,
        shape=(CI // GROUPS, reduction),
        strides=(reduction, 1),
        offsets=(tile % ctiles * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    patch_block = tl.make_block_ptr(
        PATCH + group * reduction * N * IS + batch * IS,
        shape=(reduction, IS),
        strides=(N * IS, 1),
        offsets=(0, tl.program_id(0) * BN),
        block_shape=(BK, BN),
        order=(1, 0),
    )
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, reduction, BK):
        w = tl.load(weight_block, boundary_check=(0, 1), padding_option="zero")
        patch = tl.load(patch_block, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(w, patch, acc, allow_tf32=False)
        weight_block = tl.advance(weight_block, (0, BK))
        patch_block = tl.advance(patch_block, (BK, 0))
    dest = batch * DS0 + (group * (CI // GROUPS) + ci[:, None]) * DS1
    if DS4 == 1 and (I1 == 1 or DS3 == I2) and (I0 == 1 or DS2 == I1 * I2):
        dest += spatial[None, :]
    else:
        dest += _spatial_offset(spatial, I0, I1, I2, DS2, DS3, DS4)[None, :]
    tl.store(DX + dest, acc, (ci[:, None] < CI // GROUPS) & (spatial[None, :] < IS))


@libentry()
@triton.jit
def _weight_gradient_dot(
    X,
    Y,
    DW,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    YS2: tl.constexpr,
    YS3: tl.constexpr,
    YS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    SPLIT: tl.constexpr,
    PROGRAM0=None,
    PROGRAM1=None,
    PROGRAM2=None,
):
    co = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0) // (
        ((CI // GROUPS) * KS + BM - 1) // BM
    ) * BM + tl.arange(0, BM)
    column = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0) % (
        ((CI // GROUPS) * KS + BM - 1) // BM
    ) * BM + tl.arange(0, BM)
    group = (tl.program_id(1) if PROGRAM1 is None else PROGRAM1).to(INDEX)
    split = (tl.program_id(2) if PROGRAM2 is None else PROGRAM2).to(INDEX)
    ci = column // KS
    kernel = column % KS
    acc = tl.full((BM, BM), 0, tl.float32)
    chunk: tl.constexpr = ((N * OS + SPLIT * BR - 1) // (SPLIT * BR)) * BR
    for start in range(0, chunk, BR):
        r = split * chunk + start + tl.arange(0, BR).to(INDEX)
        batch = r // OS
        spatial = r % OS
        xo = batch[:, None] * XS0 + (group * (CI // GROUPS) + ci[None, :]) * XS1
        yo = batch[None, :] * YS0 + (group * (CO // GROUPS) + co[:, None]) * YS1
        valid = (r[:, None] < N * OS) & (ci[None, :] < CI // GROUPS)
        spos = spatial
        kpos = kernel
        for axis in tl.static_range(2, -1, -1):
            op = spos % (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            kp = kpos % (K0 if axis == 0 else K1 if axis == 1 else K2)
            ip = (
                op[:, None]
                * (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                - (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                + kp[None, :] * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
            )
            valid &= (ip >= 0) & (ip < (I0 if axis == 0 else I1 if axis == 1 else I2))
            xo += ip * (XS2 if axis == 0 else XS3 if axis == 1 else XS4)
            yo += op[None, :] * (YS2 if axis == 0 else YS3 if axis == 1 else YS4)
            spos = spos // (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            kpos = kpos // (K0 if axis == 0 else K1 if axis == 1 else K2)
        if _ASCEND and (
            (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
            | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
            | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
        ):
            xo = tl.where(valid, xo, 0)
            yo = tl.where((co[:, None] < CO // GROUPS) & (r[None, :] < N * OS), yo, 0)
        x = tl.load(X + xo, valid, 0)
        y = tl.load(Y + yo, (co[:, None] < CO // GROUPS) & (r[None, :] < N * OS), 0)
        acc += tl.dot(y, x, allow_tf32=False)
    dest = split * (CO * (CI // GROUPS) * KS)
    dest += (group * (CO // GROUPS) + co[:, None]) * (CI // GROUPS) * KS + column[
        None, :
    ]
    tl.store(
        DW + dest,
        acc,
        (co[:, None] < CO // GROUPS) & (column[None, :] < (CI // GROUPS) * KS),
    )


@libentry()
@triton.jit
def _merge_weight_gradient(
    PARTIAL,
    DW,
    TOTAL: tl.constexpr,
    SPLIT: tl.constexpr,
    C1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, PARTIAL.dtype.element_ty)
    for split in range(SPLIT):
        acc += tl.load(PARTIAL + m + split * tl.full((), TOTAL, INDEX), m < TOTAL, 0)
    ks: tl.constexpr = K0 * K1 * K2
    dest = m // (C1 * ks) * S0 + (m // ks % C1) * S1
    dest += _spatial_offset(m % ks, K0, K1, K2, S2, S3, S4)
    tl.store(DW + dest, acc, m < TOTAL)


@libentry()
@triton.jit
def _merge_weight_gradient_vector(
    PARTIAL,
    DW,
    TOTAL: tl.constexpr,
    SPLIT: tl.constexpr,
    C1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    tl.static_assert(SPLIT <= 16)
    split = tl.arange(0, 16).to(INDEX)
    values = tl.load(
        PARTIAL + split[:, None] * TOTAL + m[None, :],
        (split[:, None] < SPLIT) & (m[None, :] < TOTAL),
        0,
    )
    acc = tl.sum(values, 0)
    ks: tl.constexpr = K0 * K1 * K2
    if (
        S0 == C1 * ks
        and S1 == ks
        and S4 == 1
        and (K1 == 1 or S3 == K2)
        and (K0 == 1 or S2 == K1 * K2)
    ):
        dest = m
    else:
        dest = m // (C1 * ks) * S0 + (m // ks % C1) * S1
        dest += _spatial_offset(m % ks, K0, K1, K2, S2, S3, S4)
    tl.store(DW + dest, acc, m < TOTAL)


@libentry()
@triton.jit
def _input_gradient_vector(
    GO,
    W,
    DX,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    PROGRAM0=None,
):
    m = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0).to(INDEX) * BM + tl.arange(
        0, BM
    )
    spatial = m % IS
    channel = m // IS % CI
    batch = m // (IS * CI)
    group = channel // (CI // GROUPS)
    dest = (
        batch * DS0
        + channel * DS1
        + _spatial_offset(spatial, I0, I1, I2, DS2, DS3, DS4)
    )
    acc = tl.full((BM,), 0, ACC)
    for r in tl.static_range(CO // GROUPS * KS):
        oc = r // KS
        kernel = r % KS
        go_offset = batch * GS0 + (group * (CO // GROUPS) + oc) * GS1
        if TRANSPOSED:
            weight_offset = channel * WS0 + oc * WS1
        else:
            weight_offset = (group * (CO // GROUPS) + oc) * WS0 + (
                channel % (CI // GROUPS)
            ) * WS1
        valid = m < N * CI * IS
        spos = spatial
        kpos = kernel
        for axis in tl.static_range(2, -1, -1):
            size = I0 if axis == 0 else I1 if axis == 1 else I2
            ksize = K0 if axis == 0 else K1 if axis == 1 else K2
            step = STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2
            pad = PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2
            dil = DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2
            osize = (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            ip = spos % size
            kp = kpos % ksize
            if TRANSPOSED:
                op = ip * step - pad + kp * dil
            else:
                raw = ip + pad - kp * dil
                valid &= raw % step == 0
                op = raw // step
            valid &= (op >= 0) & (op < osize)
            go_offset += op * (GS2 if axis == 0 else GS3 if axis == 1 else GS4)
            weight_offset += kp * (WS2 if axis == 0 else WS3 if axis == 1 else WS4)
            spos = spos // size
            kpos = kpos // ksize
        if _ASCEND and (
            (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
            | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
            | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
        ):
            go_offset = tl.where(valid, go_offset, 0)
            weight_offset = tl.where(valid, weight_offset, 0)
        go = tl.load(GO + go_offset, valid, 0).to(ACC)
        w = tl.load(W + weight_offset, valid, 0).to(ACC)
        acc += go * w
    tl.store(DX + dest, acc, m < N * CI * IS)


@libentry()
@triton.jit
def _input_gradient(
    GO,
    W,
    DX,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    PROGRAM0=None,
):
    # Each program owns its output elements; no floating-point atomics.
    m = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0).to(INDEX) * BM + tl.arange(
        0, BM
    )
    spatial = m % IS
    channel = m // IS % CI
    batch = m // (IS * CI)
    group = channel // (CI // GROUPS)
    dest = batch * DS0 + channel * DS1
    dest += _spatial_offset(spatial, I0, I1, I2, DS2, DS3, DS4)
    acc = tl.full((BM, BR), 0, ACC)
    for start in range(0, CO // GROUPS * KS, BR):
        r = start + tl.arange(0, BR).to(INDEX)
        oc = r // KS
        kernel = r % KS
        go_offset = batch[:, None] * GS0
        go_offset += (group[:, None] * (CO // GROUPS) + oc[None, :]) * GS1
        if TRANSPOSED:
            weight_offset = channel[:, None] * WS0 + oc[None, :] * WS1
        else:
            weight_offset = (group[:, None] * (CO // GROUPS) + oc[None, :]) * WS0
            weight_offset += (channel[:, None] % (CI // GROUPS)) * WS1
        valid = (m[:, None] < N * CI * IS) & (r[None, :] < CO // GROUPS * KS)
        spos = spatial
        kpos = kernel
        for axis in tl.static_range(2, -1, -1):
            ip = spos % (I0 if axis == 0 else I1 if axis == 1 else I2)
            kp = kpos % (K0 if axis == 0 else K1 if axis == 1 else K2)
            if TRANSPOSED:
                op = (
                    ip[:, None]
                    * (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                    - (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                    + kp[None, :] * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
                )
            else:
                raw = (
                    ip[:, None]
                    + (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                    - kp[None, :] * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
                )
                valid &= (
                    raw % (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                    == 0
                )
                op = raw // (
                    STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2
                )
            valid &= (op >= 0) & (
                op
                < (
                    OUTPUT_SHAPE0
                    if axis == 0
                    else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
                )
            )
            go_offset += op * (GS2 if axis == 0 else GS3 if axis == 1 else GS4)
            weight_offset += kp[None, :] * (
                WS2 if axis == 0 else WS3 if axis == 1 else WS4
            )
            spos = spos // (I0 if axis == 0 else I1 if axis == 1 else I2)
            kpos = kpos // (K0 if axis == 0 else K1 if axis == 1 else K2)
        if _ASCEND and (
            (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
            | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
            | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
        ):
            go_offset = tl.where(valid, go_offset, 0)
            weight_offset = tl.where(valid, weight_offset, 0)
        go = tl.load(GO + go_offset, valid, 0).to(ACC)
        w = tl.load(W + weight_offset, valid, 0).to(ACC)
        acc += go * w
    tl.store(DX + dest, tl.sum(acc, 1), m < N * CI * IS)


@libentry()
@triton.jit
def _pack_weight_operands(
    X,
    Y,
    PX,
    PY,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    YS2: tl.constexpr,
    YS3: tl.constexpr,
    YS4: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr = 4,
):
    spatial = tl.program_id(0).to(INDEX) * BLOCK + tl.arange(0, BLOCK)
    # Each program reads one spatial row with fixed channel/kernel coordinates.
    columns: tl.constexpr = CI // GROUPS * KS
    rows: tl.constexpr = CI * KS if CI * KS > CO else CO
    row_tiles: tl.constexpr = (rows + ROWS - 1) // ROWS
    row = (tl.program_id(1).to(INDEX) % row_tiles * ROWS + tl.arange(0, ROWS))[:, None]
    batch = tl.program_id(1).to(INDEX) // row_tiles
    spatial = spatial[None, :]
    group = row // columns
    col = row % columns
    kernel = col % KS
    channel = group * (CI // GROUPS) + col // KS
    valid = (row < CI * KS) & (spatial < OS)
    xo = batch * XS0 + channel * XS1
    delta = tl.full((), 0, INDEX)
    spos = spatial
    kpos = kernel
    for axis in tl.static_range(2, -1, -1):
        ip_size = I0 if axis == 0 else I1 if axis == 1 else I2
        op_size = (
            OUTPUT_SHAPE0
            if axis == 0
            else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
        )
        k_size = K0 if axis == 0 else K1 if axis == 1 else K2
        op = spos % op_size
        kp = kpos % k_size
        step = STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2
        pad = PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2
        dil = DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2
        xs = XS2 if axis == 0 else XS3 if axis == 1 else XS4
        ip = op * step - pad + kp * dil
        valid &= (ip >= 0) & (ip < ip_size)
        xo += ip * xs
        delta += (kp * dil - pad) * xs
        spos = spos // op_size
        kpos = kpos // k_size
    if (
        STRIDE0 == 1
        and STRIDE1 == 1
        and STRIDE2 == 1
        and I0 == OUTPUT_SHAPE0
        and I1 == OUTPUT_SHAPE1
        and I2 == OUTPUT_SHAPE2
        and XS4 == 1
        and (I1 == 1 or XS3 == I2)
        and (I0 == 1 or XS2 == I1 * I2)
    ):
        xo = batch * XS0 + channel * XS1 + spatial + delta
    if (
        (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
        | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
        | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
    ):
        x = tl.load(X + tl.where(valid, xo, 0))
        x = tl.where(valid, x, 0)
    elif (
        XS0 == CI * I0 * I1 * I2
        and XS1 == I0 * I1 * I2
        and XS4 == 1
        and (I1 == 1 or XS3 == I2)
        and (I0 == 1 or XS2 == I1 * I2)
    ):
        x = tl.load(X + xo, (xo >= 0) & (xo < N * CI * I0 * I1 * I2), 0)
        x = tl.where(valid, x, 0)
    else:
        x = tl.load(X + xo, valid, 0)
    if tl.program_id(1) % row_tiles * ROWS < CI * KS:
        dest_block = tl.make_block_ptr(
            PX + batch * OS,
            shape=(CI * KS, OS),
            strides=(N * OS, 1),
            offsets=(tl.program_id(1) % row_tiles * ROWS, tl.program_id(0) * BLOCK),
            block_shape=(ROWS, BLOCK),
            order=(1, 0),
        )
        tl.store(dest_block, x, boundary_check=(0, 1))
    yo = batch * YS0 + row * YS1
    if YS4 == 1 and YS3 == OUTPUT_SHAPE2 and YS2 == OUTPUT_SHAPE1 * OUTPUT_SHAPE2:
        yo = yo + spatial
    else:
        yo = yo + _spatial_offset(
            spatial, OUTPUT_SHAPE0, OUTPUT_SHAPE1, OUTPUT_SHAPE2, YS2, YS3, YS4
        )
    y = tl.load(Y + yo, (row < CO) & (spatial < OS), 0)
    tl.store(PY + row * (N * OS) + batch * OS + spatial, y, (row < CO) & (spatial < OS))


@libentry()
@triton.jit
def _weight_gradient_packed(
    PX,
    PY,
    DW,
    R: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    KS: tl.constexpr,
    SPLIT: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    columns: tl.constexpr = CI // GROUPS * KS
    tiles: tl.constexpr = (columns + BM - 1) // BM
    co = tl.program_id(0) // tiles * BM + tl.arange(0, BM)
    col = tl.program_id(0) % tiles * BM + tl.arange(0, BM)
    group = tl.program_id(1).to(INDEX) // SPLIT
    split = tl.program_id(1).to(INDEX) % SPLIT
    chunk: tl.constexpr = ((R + SPLIT * BK - 1) // (SPLIT * BK)) * BK
    acc = tl.full((BM, BM), 0, tl.float32)
    for start in range(0, chunk, BK):
        r = split * chunk + start + tl.arange(0, BK).to(INDEX)
        a = tl.load(
            PY + (group * (CO // GROUPS) + co[:, None]) * R + r[None, :],
            (co[:, None] < CO // GROUPS) & (r[None, :] < R),
            0,
        )
        b = tl.load(
            PX + (group * columns + col[None, :]) * R + r[:, None],
            (r[:, None] < R) & (col[None, :] < columns),
            0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    dest = (
        split * (CO * columns)
        + (group * (CO // GROUPS) + co[:, None]) * columns
        + col[None, :]
    )
    tl.store(DW + dest, acc, (co[:, None] < CO // GROUPS) & (col[None, :] < columns))


@libentry()
@triton.jit
def _weight_gradient(
    X,
    Y,
    DW,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    YS2: tl.constexpr,
    YS3: tl.constexpr,
    YS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    SPLIT: tl.constexpr = 1,
    PROGRAM0=None,
    PROGRAM1=None,
):
    # For transposed convolutions, X=grad_output and Y=input. This is the
    # same correlation with the original weight channel layout preserved.
    m = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0).to(INDEX) * BM + tl.arange(
        0, BM
    )
    kernel = m % KS
    ci = m // KS % (CI // GROUPS)
    co = m // (KS * (CI // GROUPS))
    group = co // (CO // GROUPS)
    dest = co * WS0 + ci * WS1 + _spatial_offset(kernel, K0, K1, K2, WS2, WS3, WS4)
    acc = tl.full((BM, BR), 0, ACC)
    chunk: tl.constexpr = ((N * OS + SPLIT * BR - 1) // (SPLIT * BR)) * BR
    split = (tl.program_id(1) if PROGRAM1 is None else PROGRAM1).to(INDEX)
    for start in range(0, chunk, BR):
        r = split * chunk + start + tl.arange(0, BR).to(INDEX)
        batch = r // OS
        spatial = r % OS
        xo = batch[None, :] * XS0 + (group * (CI // GROUPS) + ci)[:, None] * XS1
        yo = batch[None, :] * YS0 + co[:, None] * YS1
        valid = (m[:, None] < CO * (CI // GROUPS) * KS) & (r[None, :] < N * OS)
        spos = spatial
        kpos = kernel
        for axis in tl.static_range(2, -1, -1):
            op = spos % (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            kp = kpos % (K0 if axis == 0 else K1 if axis == 1 else K2)
            ip = (
                op[None, :]
                * (STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2)
                - (PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2)
                + kp[:, None] * (DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2)
            )
            valid &= (ip >= 0) & (ip < (I0 if axis == 0 else I1 if axis == 1 else I2))
            xo += ip * (XS2 if axis == 0 else XS3 if axis == 1 else XS4)
            yo += op[None, :] * (YS2 if axis == 0 else YS3 if axis == 1 else YS4)
            spos = spos // (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            kpos = kpos // (K0 if axis == 0 else K1 if axis == 1 else K2)
        if _ASCEND and (
            (PAD0 + DIL0 > I0 + OUTPUT_SHAPE0)
            | (PAD1 + DIL1 > I1 + OUTPUT_SHAPE1)
            | (PAD2 + DIL2 > I2 + OUTPUT_SHAPE2)
        ):
            xo = tl.where(valid, xo, 0)
            yo = tl.where(valid, yo, 0)
        x = tl.load(X + xo, valid, 0).to(ACC)
        y = tl.load(Y + yo, valid, 0).to(ACC)
        acc += x * y
    if SPLIT > 1:
        dest = split * (CO * (CI // GROUPS) * KS) + m
    tl.store(DW + dest, tl.sum(acc, 1), m < CO * (CI // GROUPS) * KS)


@libentry()
@triton.jit
def _bias_gradient(
    GO,
    DB,
    N: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    OS: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    BS: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRAM0=None,
):
    channel = (tl.program_id(0) if PROGRAM0 is None else PROGRAM0).to(INDEX)
    acc = tl.full((BLOCK,), 0, ACC)
    for start in range(0, N * OS, BLOCK):
        r = start + tl.arange(0, BLOCK).to(INDEX)
        offset = (r // OS) * GS0 + channel * GS1
        offset += _spatial_offset(
            r % OS, OUTPUT_SHAPE0, OUTPUT_SHAPE1, OUTPUT_SHAPE2, GS2, GS3, GS4
        )
        acc += tl.load(GO + offset, r < N * OS, 0).to(ACC)
    tl.store(DB + channel * BS, tl.sum(acc, 0))


@libentry()
@triton.jit
def _bias_gradient_spatial(
    GO,
    DB,
    N: tl.constexpr,
    OS: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    BS: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0).to(INDEX)
    r = tl.arange(0, BLOCK).to(INDEX)
    acc = tl.full((BLOCK,), 0, ACC)
    for batch in range(N):
        for start in range(0, OS, BLOCK):
            spatial = start + r
            acc += tl.load(
                GO + batch * GS0 + channel * GS1 + spatial, spatial < OS, 0
            ).to(ACC)
    tl.store(DB + channel * BS, tl.sum(acc, 0))


@libentry()
@triton.jit
def _copy_gradient(
    SRC,
    DST,
    TOTAL: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    T4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    spatial: tl.constexpr = D2 * D3 * D4
    source = m // (D1 * spatial) * S0 + (m // spatial % D1) * S1
    target = m // (D1 * spatial) * T0 + (m // spatial % D1) * T1
    source += _spatial_offset(m % spatial, D2, D3, D4, S2, S3, S4)
    target += _spatial_offset(m % spatial, D2, D3, D4, T2, T3, T4)
    value = tl.load(SRC + source, m < TOTAL, 0)
    tl.store(DST + target, value, m < TOTAL)


# Keep scalar signatures for older compilers. Compact wrappers reduce the
# host dispatch overhead of large constexpr argument lists; nested calls use
# explicit tuple indexing because star-unpack is rejected by some compilers.
_input_gradient_dot_jit = _input_gradient_dot.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["GO", "W", "DX"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _input_gradient_dot_compact(GO, W, DX, META: tl.constexpr):
    _input_gradient_dot_jit(
        GO,
        W,
        DX,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        META[17],
        META[18],
        META[19],
        META[20],
        META[21],
        META[22],
        META[23],
        META[24],
        META[25],
        META[26],
        META[27],
        META[28],
        META[29],
        META[30],
        META[31],
        META[32],
        META[33],
        META[34],
        META[35],
        META[36],
        META[37],
        META[38],
        META[39],
        META[40],
        META[41],
        META[42],
        META[43],
        META[44],
    )


_weight_gradient_dot_jit = _weight_gradient_dot.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["X", "Y", "DW"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _weight_gradient_dot_compact(X, Y, DW, META: tl.constexpr):
    _weight_gradient_dot_jit(
        X,
        Y,
        DW,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        META[17],
        META[18],
        META[19],
        META[20],
        META[21],
        META[22],
        META[23],
        META[24],
        META[25],
        META[26],
        META[27],
        META[28],
        META[29],
        META[30],
        META[31],
        META[32],
        META[33],
        META[34],
        META[35],
        META[36],
        META[37],
        META[38],
        META[39],
        META[40],
        META[41],
        META[42],
        META[43],
    )


_merge_weight_gradient_jit = _merge_weight_gradient.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["PARTIAL", "DW"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _merge_weight_gradient_compact(PARTIAL, DW, META: tl.constexpr):
    _merge_weight_gradient_jit(
        PARTIAL,
        DW,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
    )


_input_gradient_vector_jit = _input_gradient_vector.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["GO", "W", "DX"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _input_gradient_vector_compact(GO, W, DX, META: tl.constexpr):
    _input_gradient_vector_jit(
        GO,
        W,
        DX,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        META[17],
        META[18],
        META[19],
        META[20],
        META[21],
        META[22],
        META[23],
        META[24],
        META[25],
        META[26],
        META[27],
        META[28],
        META[29],
        META[30],
        META[31],
        META[32],
        META[33],
        META[34],
        META[35],
        META[36],
        META[37],
        META[38],
        META[39],
        META[40],
        META[41],
        META[42],
        META[43],
        META[44],
    )


_input_gradient_jit = _input_gradient.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["GO", "W", "DX"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _input_gradient_compact(GO, W, DX, META: tl.constexpr):
    _input_gradient_jit(
        GO,
        W,
        DX,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        META[17],
        META[18],
        META[19],
        META[20],
        META[21],
        META[22],
        META[23],
        META[24],
        META[25],
        META[26],
        META[27],
        META[28],
        META[29],
        META[30],
        META[31],
        META[32],
        META[33],
        META[34],
        META[35],
        META[36],
        META[37],
        META[38],
        META[39],
        META[40],
        META[41],
        META[42],
        META[43],
        META[44],
    )


_weight_gradient_jit = _weight_gradient.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["X", "Y", "DW"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _weight_gradient_compact(X, Y, DW, META: tl.constexpr):
    _weight_gradient_jit(
        X,
        Y,
        DW,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        META[17],
        META[18],
        META[19],
        META[20],
        META[21],
        META[22],
        META[23],
        META[24],
        META[25],
        META[26],
        META[27],
        META[28],
        META[29],
        META[30],
        META[31],
        META[32],
        META[33],
        META[34],
        META[35],
        META[36],
        META[37],
        META[38],
        META[39],
        META[40],
        META[41],
        META[42],
        META[43],
    )


_bias_gradient_jit = _bias_gradient.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["GO", "DB"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _bias_gradient_compact(GO, DB, META: tl.constexpr):
    _bias_gradient_jit(
        GO,
        DB,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
    )


_copy_gradient_jit = _copy_gradient.jit_function


@libentry()
@triton.jit(
    do_not_specialize=(
        ["SRC", "DST"] if runtime_device.vendor_name in ("nvidia", "hygon") else []
    )
)
def _copy_gradient_compact(SRC, DST, META: tl.constexpr):
    _copy_gradient_jit(
        SRC,
        DST,
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
    )


_COMPACT_KERNELS = {
    _input_gradient_dot: (_input_gradient_dot_compact, 3),
    _weight_gradient_dot: (_weight_gradient_dot_compact, 3),
    _merge_weight_gradient: (_merge_weight_gradient_compact, 2),
    _input_gradient_vector: (_input_gradient_vector_compact, 3),
    _input_gradient: (_input_gradient_compact, 3),
    _weight_gradient: (_weight_gradient_compact, 3),
    _bias_gradient: (_bias_gradient_compact, 2),
    _copy_gradient: (_copy_gradient_compact, 2),
}


def _launch_gradient(kernel, grid, *args, pending=None, record=None):
    if pending is not None:
        pending.append((kernel, grid, args))
        return
    if runtime_device.vendor_name in ("nvidia", "hygon"):
        compact, pointers = _COMPACT_KERNELS[kernel]
        call_args = (*args[:pointers], args[pointers:])
        compiled, _ = compact[grid](*call_args)
        if record is not None:
            record.append((compiled, grid, call_args, pointers))
    else:
        kernel[grid](*args)


@libentry()
@triton.jit(
    do_not_specialize=(
        ["GO", "W", "DX", "X", "Y", "DW", "DB"]
        if runtime_device.vendor_name in ("nvidia", "hygon")
        else []
    )
)
def _combined_gradients(
    GO,
    W,
    DX,
    X,
    Y,
    DW,
    DB,
    DX_META: tl.constexpr,
    DW_META: tl.constexpr,
    DB_META: tl.constexpr,
    DX_KIND: tl.constexpr,
    DW_DOT: tl.constexpr,
    DX_COUNT: tl.constexpr,
    DW_COUNT: tl.constexpr,
    DX_GRID0: tl.constexpr,
    DW_GRID0: tl.constexpr,
    DW_GRID1: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < DX_COUNT:
        if DX_KIND == 2:
            _input_gradient_dot_jit(
                GO,
                W,
                DX,
                DX_META[0],
                DX_META[1],
                DX_META[2],
                DX_META[3],
                DX_META[4],
                DX_META[5],
                DX_META[6],
                DX_META[7],
                DX_META[8],
                DX_META[9],
                DX_META[10],
                DX_META[11],
                DX_META[12],
                DX_META[13],
                DX_META[14],
                DX_META[15],
                DX_META[16],
                DX_META[17],
                DX_META[18],
                DX_META[19],
                DX_META[20],
                DX_META[21],
                DX_META[22],
                DX_META[23],
                DX_META[24],
                DX_META[25],
                DX_META[26],
                DX_META[27],
                DX_META[28],
                DX_META[29],
                DX_META[30],
                DX_META[31],
                DX_META[32],
                DX_META[33],
                DX_META[34],
                DX_META[35],
                DX_META[36],
                DX_META[37],
                DX_META[38],
                DX_META[39],
                DX_META[40],
                DX_META[41],
                DX_META[42],
                DX_META[43],
                DX_META[44],
                PROGRAM0=pid % DX_GRID0,
                PROGRAM1=pid // DX_GRID0,
            )
        elif DX_KIND == 1:
            _input_gradient_vector_jit(
                GO,
                W,
                DX,
                DX_META[0],
                DX_META[1],
                DX_META[2],
                DX_META[3],
                DX_META[4],
                DX_META[5],
                DX_META[6],
                DX_META[7],
                DX_META[8],
                DX_META[9],
                DX_META[10],
                DX_META[11],
                DX_META[12],
                DX_META[13],
                DX_META[14],
                DX_META[15],
                DX_META[16],
                DX_META[17],
                DX_META[18],
                DX_META[19],
                DX_META[20],
                DX_META[21],
                DX_META[22],
                DX_META[23],
                DX_META[24],
                DX_META[25],
                DX_META[26],
                DX_META[27],
                DX_META[28],
                DX_META[29],
                DX_META[30],
                DX_META[31],
                DX_META[32],
                DX_META[33],
                DX_META[34],
                DX_META[35],
                DX_META[36],
                DX_META[37],
                DX_META[38],
                DX_META[39],
                DX_META[40],
                DX_META[41],
                DX_META[42],
                DX_META[43],
                DX_META[44],
                PROGRAM0=pid,
            )
        else:
            _input_gradient_jit(
                GO,
                W,
                DX,
                DX_META[0],
                DX_META[1],
                DX_META[2],
                DX_META[3],
                DX_META[4],
                DX_META[5],
                DX_META[6],
                DX_META[7],
                DX_META[8],
                DX_META[9],
                DX_META[10],
                DX_META[11],
                DX_META[12],
                DX_META[13],
                DX_META[14],
                DX_META[15],
                DX_META[16],
                DX_META[17],
                DX_META[18],
                DX_META[19],
                DX_META[20],
                DX_META[21],
                DX_META[22],
                DX_META[23],
                DX_META[24],
                DX_META[25],
                DX_META[26],
                DX_META[27],
                DX_META[28],
                DX_META[29],
                DX_META[30],
                DX_META[31],
                DX_META[32],
                DX_META[33],
                DX_META[34],
                DX_META[35],
                DX_META[36],
                DX_META[37],
                DX_META[38],
                DX_META[39],
                DX_META[40],
                DX_META[41],
                DX_META[42],
                DX_META[43],
                DX_META[44],
                PROGRAM0=pid,
            )
    elif pid < DX_COUNT + DW_COUNT:
        wp = pid - DX_COUNT
        if DW_DOT:
            _weight_gradient_dot_jit(
                X,
                Y,
                DW,
                DW_META[0],
                DW_META[1],
                DW_META[2],
                DW_META[3],
                DW_META[4],
                DW_META[5],
                DW_META[6],
                DW_META[7],
                DW_META[8],
                DW_META[9],
                DW_META[10],
                DW_META[11],
                DW_META[12],
                DW_META[13],
                DW_META[14],
                DW_META[15],
                DW_META[16],
                DW_META[17],
                DW_META[18],
                DW_META[19],
                DW_META[20],
                DW_META[21],
                DW_META[22],
                DW_META[23],
                DW_META[24],
                DW_META[25],
                DW_META[26],
                DW_META[27],
                DW_META[28],
                DW_META[29],
                DW_META[30],
                DW_META[31],
                DW_META[32],
                DW_META[33],
                DW_META[34],
                DW_META[35],
                DW_META[36],
                DW_META[37],
                DW_META[38],
                DW_META[39],
                DW_META[40],
                DW_META[41],
                DW_META[42],
                DW_META[43],
                PROGRAM0=wp % DW_GRID0,
                PROGRAM1=wp // DW_GRID0 % DW_GRID1,
                PROGRAM2=wp // (DW_GRID0 * DW_GRID1),
            )
        else:
            _weight_gradient_jit(
                X,
                Y,
                DW,
                DW_META[0],
                DW_META[1],
                DW_META[2],
                DW_META[3],
                DW_META[4],
                DW_META[5],
                DW_META[6],
                DW_META[7],
                DW_META[8],
                DW_META[9],
                DW_META[10],
                DW_META[11],
                DW_META[12],
                DW_META[13],
                DW_META[14],
                DW_META[15],
                DW_META[16],
                DW_META[17],
                DW_META[18],
                DW_META[19],
                DW_META[20],
                DW_META[21],
                DW_META[22],
                DW_META[23],
                DW_META[24],
                DW_META[25],
                DW_META[26],
                DW_META[27],
                DW_META[28],
                DW_META[29],
                DW_META[30],
                DW_META[31],
                DW_META[32],
                DW_META[33],
                DW_META[34],
                DW_META[35],
                DW_META[36],
                DW_META[37],
                DW_META[38],
                DW_META[39],
                DW_META[40],
                DW_META[41],
                DW_META[42],
                DW_META[43],
                PROGRAM0=wp % DW_GRID0,
                PROGRAM1=wp // DW_GRID0,
            )
    else:
        _bias_gradient_jit(
            GO,
            DB,
            DB_META[0],
            DB_META[1],
            DB_META[2],
            DB_META[3],
            DB_META[4],
            DB_META[5],
            DB_META[6],
            DB_META[7],
            DB_META[8],
            DB_META[9],
            DB_META[10],
            DB_META[11],
            DB_META[12],
            DB_META[13],
            PROGRAM0=pid - DX_COUNT - DW_COUNT,
        )


def _launch_combined(pending, record=None):
    dx_kernel, dx_grid, dx_args = pending[0]
    dw_kernel, dw_grid, dw_args = pending[1]
    _, db_grid, db_args = pending[-1]
    dx_count, dw_count = math.prod(dx_grid), math.prod(dw_grid)
    dx_kind = (
        2
        if dx_kernel is _input_gradient_dot
        else int(dx_kernel is _input_gradient_vector)
    )
    grid = (dx_count + dw_count + math.prod(db_grid),)
    combined_args = (
        *dx_args[:3],
        *dw_args[:3],
        db_args[1],
        dx_args[3:],
        dw_args[3:],
        db_args[2:],
        dx_kind,
        dw_kernel is _weight_gradient_dot,
        dx_count,
        dw_count,
        dx_grid[0],
        dw_grid[0],
        dw_grid[1],
    )
    compiled, _ = _combined_gradients[grid](*combined_args)
    if record is not None:
        record.append((compiled, grid, combined_args, 7))
    if len(pending) == 4:
        kernel, grid, args = pending[2]
        _launch_gradient(kernel, grid, *args, record=record)


def _normalize_parameter(value, dimensions, name, minimum):
    if not isinstance(value, (tuple, list)) or len(value) not in (1, dimensions):
        raise RuntimeError(f"convolution_backward: invalid {name} length")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < minimum for v in value):
        raise RuntimeError(f"convolution_backward: invalid {name}")
    return tuple(value) * dimensions if len(value) == 1 else tuple(value)


def _empty_gradient(tensor):
    dimensions = tensor.ndim - 2
    if dimensions in (2, 3):
        fmt = torch.channels_last if dimensions == 2 else torch.channels_last_3d
        if tensor.is_contiguous(memory_format=fmt):
            # Allocate the physical layout explicitly: some adapters ignore
            # preserve_format in empty_like, even for a dense strided input.
            storage = torch.empty(
                (tensor.shape[0], *tensor.shape[2:], tensor.shape[1]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return storage.permute(0, dimensions + 1, *range(1, dimensions + 1))
    return torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)


def _convolution_backward_uncached(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    _outputs=None,
    _record=None,
):
    dimensions = input.ndim - 2
    if (
        dimensions not in (1, 2, 3)
        or weight.ndim != input.ndim
        or grad_output.ndim != input.ndim
    ):
        raise RuntimeError(
            "convolution_backward: expected matching batched 1D/2D/3D tensors"
        )
    if (
        input.layout != torch.strided
        or weight.layout != torch.strided
        or grad_output.layout != torch.strided
    ):
        raise RuntimeError("convolution_backward: expected strided tensors")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("convolution_backward: expected real floating-point tensors")
    if input.dtype == torch.float64 and not runtime_device.support_fp64:
        raise RuntimeError(
            "convolution_backward: this backend does not support FP64 arithmetic"
        )
    if input.dtype != weight.dtype or input.dtype != grad_output.dtype:
        raise RuntimeError("convolution_backward: dtype mismatch")
    if input.device != weight.device or input.device != grad_output.device:
        raise RuntimeError("convolution_backward: device mismatch")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise RuntimeError("convolution_backward: groups must be positive")
    if len(output_mask) != 3 or any(type(m) is not bool for m in output_mask):
        raise RuntimeError(
            "convolution_backward: output_mask must contain three booleans"
        )
    stride = _normalize_parameter(stride, dimensions, "stride", 1)
    padding = _normalize_parameter(padding, dimensions, "padding", 0)
    dilation = _normalize_parameter(dilation, dimensions, "dilation", 1)
    output_padding = _normalize_parameter(
        output_padding, dimensions, "output_padding", 0
    )
    if any(
        s <= 0 for s in (*input.shape[2:], *weight.shape[2:], *grad_output.shape[2:])
    ):
        raise RuntimeError("convolution_backward: spatial dimensions must be positive")
    if weight.shape[0] < groups or weight.shape[0] % groups:
        raise RuntimeError("convolution_backward: invalid weight channels for groups")
    if transposed:
        channels = weight.shape[1] * groups
        if input.shape[1] != weight.shape[0]:
            raise RuntimeError("convolution_backward: input/weight channel mismatch")
        if any(
            op >= s and op >= d for op, s, d in zip(output_padding, stride, dilation)
        ):
            raise RuntimeError(
                "convolution_backward: output_padding must be smaller than stride or dilation"
            )
    else:
        channels = weight.shape[0]
        if input.shape[1] != weight.shape[1] * groups:
            raise RuntimeError("convolution_backward: input/weight channel mismatch")
        if any(output_padding):
            raise RuntimeError(
                "convolution_backward: ordinary convolution requires zero output_padding"
            )
    spatial = []
    for i, k, s, p, d, op in zip(
        input.shape[2:], weight.shape[2:], stride, padding, dilation, output_padding
    ):
        spatial.append(
            (i - 1) * s - 2 * p + d * (k - 1) + op + 1
            if transposed
            else (i + 2 * p - d * (k - 1) - 1) // s + 1
        )
    # ATen's empty-channel forward keeps the output channel dimension empty.
    if input.shape[1] == 0:
        channels = 0
    if any(s <= 0 for s in spatial) or tuple(grad_output.shape) != (
        input.shape[0],
        channels,
        *spatial,
    ):
        raise RuntimeError(
            "convolution_backward: grad_output shape does not match convolution"
        )
    bias_channels = weight.shape[1] * groups if transposed else weight.shape[0]
    if bias_sizes is not None and tuple(bias_sizes) != (bias_channels,):
        raise RuntimeError("convolution_backward: invalid bias_sizes")
    if _outputs is None:
        dx = _empty_gradient(input) if output_mask[0] else None
        dw = _empty_gradient(weight) if output_mask[1] else None
        db = (
            torch.empty((bias_channels,), dtype=input.dtype, device=input.device)
            if output_mask[2]
            else None
        )
    else:
        for out, shape in zip(_outputs, (input.shape, weight.shape, (bias_channels,))):
            if out.shape != shape:
                out.resize_(shape)
            if any(
                size > 1 and step == 0 for size, step in zip(out.shape, out.stride())
            ):
                raise RuntimeError("convolution_backward.out: overlapping output")
        dx, dw, db = _outputs
    if not any(output_mask):
        return dx, dw, db
    with torch_device_fn.device(input.device):
        if input.numel() == 0 or grad_output.numel() == 0:
            for result in (dx, dw, db):
                if result is not None and result.numel():
                    fill_scalar_(result, 0)
            return dx, dw, db
        acc = tl.float64 if input.dtype == torch.float64 else tl.float32
        # Bound logical indices, strided offsets and the expanded convolution
        # coordinates before choosing the fast 32-bit address arithmetic.
        index_bound = max(
            max(t.numel(), sum((n - 1) * s for n, s in zip(t.shape, t.stride())))
            for t in (input, weight, grad_output, dx, dw, db)
            if t is not None
        )
        coordinate_bound = max(
            i * s + p + k * d + o
            for i, k, s, p, d, o in zip(
                input.shape[2:],
                weight.shape[2:],
                stride,
                padding,
                dilation,
                grad_output.shape[2:],
            )
        )
        index = (
            tl.int32
            if max(index_bound, coordinate_bound, 16 * weight.numel()) < 2**31 - 4096
            else tl.int64
        )
        i, o, k = (
            (1,) * (3 - dimensions) + tuple(input.shape[2:]),
            (1,) * (3 - dimensions) + tuple(grad_output.shape[2:]),
            (1,) * (3 - dimensions) + tuple(weight.shape[2:]),
        )
        stride = (1,) * (3 - dimensions) + stride
        padding = (0,) * (3 - dimensions) + padding
        dilation = (1,) * (3 - dimensions) + dilation
        spatial_pack_safe = False
        if runtime_device.vendor_name == "ascend" and i == o:
            margin = sum(
                max(p, (extent - 1) * d - p) * step
                for extent, p, d, step in zip(
                    k, padding, dilation, (i[1] * i[2], i[2], 1)
                )
            )
            pitch = (
                triton.cdiv(triton.cdiv(math.prod(i), 1024) * 1024 + 2 * margin, 16)
                * 16
            )
            spatial_pack_safe = (
                input.shape[0]
                * max(input.shape[1], channels)
                * pitch
                * input.element_size()
                <= 32 * 1024 * 1024
            )
        # Small NVIDIA convolutions are faster on the direct SIMT path.
        matrix_channels = 4 if runtime_device.vendor_name == "nvidia" else 3
        if (
            runtime_device.vendor_name == "ascend"
            and i == o
            and all(step == 1 for step in stride)
            and input.is_contiguous()
            and grad_output.is_contiguous()
            and ((input.numel() + grad_output.numel()) * math.prod(k) + weight.numel())
            * input.element_size()
            <= 32 * 1024 * 1024
        ):
            # Explicit packing also removes scalar gather overhead for depthwise.
            matrix_channels = 1
        use_dot = (
            input.dtype != torch.float64
            and not _LEGACY_ASCEND
            and (runtime_device.vendor_name != "ascend" or math.prod(i) > 1)
            and input.shape[1] // groups >= matrix_channels
            and (runtime_device.vendor_name != "nvidia" or dimensions > 1)
        )
        pending = (
            []
            if runtime_device.vendor_name in ("nvidia", "hygon")
            and input.dtype != torch.float64
            and all(output_mask)
            else None
        )
        spatial_vector = (
            input.dtype == torch.float32
            and dimensions > 1
            and (
                (
                    runtime_device.vendor_name == "nvidia"
                    and input.shape[1] // groups >= 4
                    and channels // groups * math.prod(k) <= 512
                )
                or (
                    runtime_device.vendor_name == "mthreads"
                    and input.shape[1] // groups <= 3
                    and channels // groups * math.prod(k) <= 128
                )
            )
        )
        use_dot_input = use_dot and not spatial_vector
        pack_input_elements = (
            (input.numel() * channels // input.shape[1] * math.prod(k) + weight.numel())
            if runtime_device.vendor_name == "ascend"
            else 0
        )
        input_patch = None
        if (
            dx is not None
            and use_dot_input
            and runtime_device.vendor_name == "ascend"
            and pack_input_elements * input.element_size() <= 32 * 1024 * 1024
        ):
            patch = torch.empty(
                (channels * math.prod(k), input.shape[0] * math.prod(i)),
                dtype=input.dtype,
                device=input.device,
            )
            packed_weight = torch.empty(
                (channels * math.prod(k), input.shape[1] // groups),
                dtype=weight.dtype,
                device=weight.device,
            )
            if (
                i == o
                and spatial_pack_safe
                and all(step == 1 for step in stride)
                and grad_output.is_contiguous()
            ):
                padded, margin, pitch = _padded_pack_source(
                    grad_output, k, padding, dilation, index
                )
                pack_rows = min(8, channels & -channels)
                _pack_spatial_rows[
                    (
                        triton.cdiv(math.prod(i), 1024),
                        input.shape[0] * math.prod(k),
                        triton.cdiv(channels, pack_rows),
                    )
                ](
                    padded,
                    weight,
                    patch,
                    packed_weight,
                    input.shape[0],
                    channels,
                    input.shape[1],
                    groups,
                    *i,
                    *k,
                    *padding,
                    *dilation,
                    *(
                        weight.stride()[:2]
                        + (0,) * (3 - dimensions)
                        + weight.stride()[2:]
                    ),
                    True,
                    transposed,
                    index,
                    triton.next_power_of_2(input.shape[1] // groups),
                    MARGIN=margin,
                    PITCH=pitch,
                    ROWS=pack_rows,
                    num_stages=1,
                )
                _pack_input_weights[(triton.cdiv(weight.numel(), 512),)](
                    weight,
                    packed_weight,
                    input.shape[1],
                    channels,
                    groups,
                    *k,
                    *(
                        weight.stride()[:2]
                        + (0,) * (3 - dimensions)
                        + weight.stride()[2:]
                    ),
                    transposed,
                    index,
                    512,
                )
            else:
                _pack_input_operands[
                    (
                        triton.cdiv(math.prod(i), 1024),
                        input.shape[0] * triton.cdiv(channels * math.prod(k), 4),
                    )
                ](
                    grad_output,
                    weight,
                    patch,
                    packed_weight,
                    input.shape[0],
                    input.shape[1],
                    channels,
                    groups,
                    *i,
                    *o,
                    *k,
                    math.prod(i),
                    math.prod(o),
                    math.prod(k),
                    *stride,
                    *padding,
                    *dilation,
                    *(
                        grad_output.stride()[:2]
                        + (0,) * (3 - dimensions)
                        + grad_output.stride()[2:]
                    ),
                    *(
                        weight.stride()[:2]
                        + (0,) * (3 - dimensions)
                        + weight.stride()[2:]
                    ),
                    transposed,
                    index,
                    1024,
                    triton.next_power_of_2(input.shape[1] // groups),
                    num_stages=1,
                )
            _input_gradient_packed[
                (
                    triton.cdiv(math.prod(i), 128),
                    input.shape[0] * groups * triton.cdiv(input.shape[1] // groups, 16),
                )
            ](
                patch,
                packed_weight,
                dx,
                input.shape[0],
                input.shape[1],
                channels,
                groups,
                *i,
                math.prod(i),
                math.prod(k),
                *(dx.stride()[:2] + (0,) * (3 - dimensions) + dx.stride()[2:]),
                index,
            )
            input_patch = patch
        elif dx is not None:
            use_vector = not use_dot_input and (
                channels // groups <= 4 or spatial_vector
            )
            dx_block = 128 if use_vector else 16
            dx_kernel = (
                _input_gradient_dot
                if use_dot_input
                else _input_gradient_vector if use_vector else _input_gradient
            )
            dx_grid = (
                (
                    triton.cdiv(math.prod(i), 32),
                    input.shape[0] * groups * triton.cdiv(input.shape[1] // groups, 32),
                )
                if use_dot_input
                else (triton.cdiv(input.numel(), dx_block),)
            )
            _launch_gradient(
                dx_kernel,
                dx_grid,
                grad_output,
                weight,
                dx,
                input.shape[0],
                input.shape[1],
                channels,
                groups,
                *i,
                *o,
                *k,
                math.prod(i),
                math.prod(o),
                math.prod(k),
                *stride,
                *padding,
                *dilation,
                *(
                    grad_output.stride()[:2]
                    + (0,) * (3 - dimensions)
                    + grad_output.stride()[2:]
                ),
                *(weight.stride()[:2] + (0,) * (3 - dimensions) + weight.stride()[2:]),
                *(dx.stride()[:2] + (0,) * (3 - dimensions) + dx.stride()[2:]),
                transposed,
                acc,
                index,
                32 if use_dot_input else dx_block,
                32 if use_dot_input else 128,
                pending=pending,
                record=_record,
            )
        if dw is not None:
            x, y = (grad_output, input) if transposed else (input, grad_output)
            pack_elements = (
                x.shape[0]
                * math.prod(y.shape[2:])
                * (x.shape[1] * math.prod(k) + y.shape[1])
            )
            use_dot_weight = use_dot and (
                runtime_device.vendor_name != "ascend"
                or pack_elements * x.element_size() <= 32 * 1024 * 1024
            )
            if use_dot_weight:
                splits = min(
                    16, max(1, triton.cdiv(x.shape[0] * math.prod(y.shape[2:]), 256))
                )
                partial = torch.empty(
                    (splits, weight.numel()), device=input.device, dtype=torch.float32
                )
                pack_elements = (
                    x.shape[0]
                    * math.prod(y.shape[2:])
                    * (x.shape[1] * math.prod(k) + y.shape[1])
                )
                use_packed = (
                    runtime_device.vendor_name == "ascend"
                    and pack_elements * x.element_size() <= 32 * 1024 * 1024
                )
                if use_packed:
                    reduction = x.shape[0] * math.prod(y.shape[2:])
                    reuse_input_patch = (
                        transposed
                        and input_patch is not None
                        and i == o
                        and all(step == 1 for step in stride)
                        and input.is_contiguous()
                        and grad_output.is_contiguous()
                    )
                    # Transposed dx and dw gather the same grad_output patches.
                    px = (
                        input_patch
                        if reuse_input_patch
                        else torch.empty(
                            (groups, x.shape[1] // groups * math.prod(k), reduction),
                            dtype=x.dtype,
                            device=x.device,
                        )
                    )
                    py = torch.empty(
                        (y.shape[1], reduction), dtype=y.dtype, device=y.device
                    )
                    if (
                        runtime_device.vendor_name == "ascend"
                        and spatial_pack_safe
                        and x.shape[2:] == y.shape[2:]
                        and all(step == 1 for step in stride)
                        and x.is_contiguous()
                        and y.is_contiguous()
                    ):
                        if not reuse_input_patch:
                            padded, margin, pitch = _padded_pack_source(
                                x, k, padding, dilation, index
                            )
                            pack_rows = min(8, x.shape[1] & -x.shape[1])
                            _pack_spatial_rows[
                                (
                                    triton.cdiv(math.prod(y.shape[2:]), 1024),
                                    x.shape[0] * math.prod(k),
                                    triton.cdiv(x.shape[1], pack_rows),
                                )
                            ](
                                padded,
                                y,
                                px,
                                py,
                                x.shape[0],
                                x.shape[1],
                                y.shape[1],
                                groups,
                                *((1,) * (3 - dimensions) + tuple(x.shape[2:])),
                                *k,
                                *padding,
                                *dilation,
                                0,
                                0,
                                0,
                                0,
                                0,
                                False,
                                False,
                                index,
                                1,
                                MARGIN=margin,
                                PITCH=pitch,
                                ROWS=pack_rows,
                                num_stages=1,
                            )
                        _pack_output_rows[
                            (
                                triton.cdiv(math.prod(y.shape[2:]), 1024),
                                y.shape[0] * y.shape[1],
                            )
                        ](
                            y,
                            py,
                            y.shape[0],
                            y.shape[1],
                            math.prod(y.shape[2:]),
                            index,
                        )
                    else:
                        _pack_weight_operands[
                            (
                                triton.cdiv(math.prod(y.shape[2:]), 1024),
                                x.shape[0]
                                * triton.cdiv(
                                    max(x.shape[1] * math.prod(k), y.shape[1]), 4
                                ),
                            )
                        ](
                            x,
                            y,
                            px,
                            py,
                            x.shape[0],
                            x.shape[1],
                            y.shape[1],
                            groups,
                            *((1,) * (3 - dimensions) + tuple(x.shape[2:])),
                            *((1,) * (3 - dimensions) + tuple(y.shape[2:])),
                            *k,
                            math.prod(y.shape[2:]),
                            math.prod(k),
                            *stride,
                            *padding,
                            *dilation,
                            *(
                                x.stride()[:2]
                                + (0,) * (3 - dimensions)
                                + x.stride()[2:]
                            ),
                            *(
                                y.stride()[:2]
                                + (0,) * (3 - dimensions)
                                + y.stride()[2:]
                            ),
                            index,
                            1024,
                            num_stages=1,
                        )
                    _weight_gradient_packed[
                        (
                            triton.cdiv(y.shape[1] // groups, 32)
                            * triton.cdiv(x.shape[1] // groups * math.prod(k), 32),
                            groups * splits,
                        )
                    ](
                        px,
                        py,
                        partial,
                        reduction,
                        x.shape[1],
                        y.shape[1],
                        groups,
                        math.prod(k),
                        splits,
                        index,
                    )
                else:
                    _launch_gradient(
                        _weight_gradient_dot,
                        (
                            triton.cdiv(y.shape[1] // groups, 32)
                            * triton.cdiv(x.shape[1] // groups * math.prod(k), 32),
                            groups,
                            splits,
                        ),
                        x,
                        y,
                        partial,
                        x.shape[0],
                        x.shape[1],
                        y.shape[1],
                        groups,
                        *((1,) * (3 - dimensions) + tuple(x.shape[2:])),
                        *((1,) * (3 - dimensions) + tuple(y.shape[2:])),
                        *k,
                        math.prod(y.shape[2:]),
                        math.prod(k),
                        *stride,
                        *padding,
                        *dilation,
                        *(x.stride()[:2] + (0,) * (3 - dimensions) + x.stride()[2:]),
                        *(y.stride()[:2] + (0,) * (3 - dimensions) + y.stride()[2:]),
                        *(dw.stride()[:2] + (0,) * (3 - dimensions) + dw.stride()[2:]),
                        acc,
                        index,
                        32,
                        32,
                        splits,
                        pending=pending,
                        record=_record,
                    )
                _launch_gradient(
                    (
                        _merge_weight_gradient_vector
                        if runtime_device.vendor_name == "ascend" and not _LEGACY_ASCEND
                        else _merge_weight_gradient
                    ),
                    (triton.cdiv(weight.numel(), 256),),
                    partial,
                    dw,
                    weight.numel(),
                    splits,
                    weight.shape[1],
                    *k,
                    *(dw.stride()[:2] + (0,) * (3 - dimensions) + dw.stride()[2:]),
                    index,
                    256,
                    pending=pending,
                    record=_record,
                )
            else:
                reduction_size = x.shape[0] * math.prod(y.shape[2:])
                single_reduction = (
                    runtime_device.vendor_name == "nvidia"
                    and x.shape[1] // groups <= 2
                    and weight.numel() >= 128
                    and reduction_size <= 8192
                )
                dw_bm = 1 if single_reduction else 8
                dw_br = 1024 if single_reduction else 128
                splits = (
                    1
                    if single_reduction or reduction_size <= 1024
                    else min(16, triton.cdiv(reduction_size, 512))
                )
                partial = (
                    torch.empty(
                        (splits, weight.numel()),
                        device=input.device,
                        dtype=(
                            torch.float64
                            if input.dtype == torch.float64
                            else torch.float32
                        ),
                    )
                    if splits > 1
                    else dw
                )
                _launch_gradient(
                    _weight_gradient,
                    (triton.cdiv(weight.numel(), dw_bm), splits),
                    x,
                    y,
                    partial,
                    x.shape[0],
                    x.shape[1],
                    y.shape[1],
                    groups,
                    *((1,) * (3 - dimensions) + tuple(x.shape[2:])),
                    *((1,) * (3 - dimensions) + tuple(y.shape[2:])),
                    *k,
                    math.prod(y.shape[2:]),
                    math.prod(k),
                    *stride,
                    *padding,
                    *dilation,
                    *(x.stride()[:2] + (0,) * (3 - dimensions) + x.stride()[2:]),
                    *(y.stride()[:2] + (0,) * (3 - dimensions) + y.stride()[2:]),
                    *(dw.stride()[:2] + (0,) * (3 - dimensions) + dw.stride()[2:]),
                    acc,
                    index,
                    dw_bm,
                    dw_br,
                    splits,
                    pending=pending,
                    record=_record,
                )
                if splits > 1:
                    _launch_gradient(
                        (
                            _merge_weight_gradient_vector
                            if runtime_device.vendor_name == "ascend"
                            and not _LEGACY_ASCEND
                            else _merge_weight_gradient
                        ),
                        (triton.cdiv(weight.numel(), 256),),
                        partial,
                        dw,
                        weight.numel(),
                        splits,
                        weight.shape[1],
                        *k,
                        *(dw.stride()[:2] + (0,) * (3 - dimensions) + dw.stride()[2:]),
                        index,
                        256,
                        pending=pending,
                        record=_record,
                    )
        if (
            db is not None
            and runtime_device.vendor_name == "ascend"
            and not _LEGACY_ASCEND
            and grad_output.is_contiguous()
        ):
            _bias_gradient_spatial[(channels,)](
                grad_output,
                db,
                input.shape[0],
                math.prod(o),
                grad_output.stride(0),
                grad_output.stride(1),
                db.stride(0),
                acc,
                index,
                triton.next_power_of_2(min(math.prod(o), 8192)),
            )
        elif db is not None:
            _launch_gradient(
                _bias_gradient,
                (channels,),
                grad_output,
                db,
                input.shape[0],
                *o,
                math.prod(o),
                *(
                    grad_output.stride()[:2]
                    + (0,) * (3 - dimensions)
                    + grad_output.stride()[2:]
                ),
                db.stride(0),
                acc,
                index,
                1024,
                pending=pending,
                record=_record,
            )
        if pending is not None:
            _launch_combined(pending, record=_record)
    return dx, dw, db


# Plans retain only immutable metadata and tensor slot numbers. Every invocation
# supplies fresh pointers and workspace; streams and tensor values are not cached.
# Compact kernels disable pointer specialization, so compiled launches are
# valid for every pointer matching the cached dtype/device/shape/stride contract.
_LAUNCH_PLANS = {}


def _convolution_backward_impl(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    _outputs=None,
):
    arguments = (
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    if (
        runtime_device.vendor_name not in ("nvidia", "hygon")
        or input.numel() == 0
        or grad_output.numel() == 0
    ):
        return _convolution_backward_uncached(*arguments, _outputs=_outputs)
    # Include types so True/1 and 1.0/1 cannot reuse an integer-validated plan.
    try:
        parameter_key = tuple(
            (
                type(values),
                None if values is None else tuple((type(v), v) for v in values),
            )
            for values in (
                bias_sizes,
                stride,
                padding,
                dilation,
                output_padding,
                output_mask,
            )
        )
        tensor_key = tuple(
            (t.shape, t.stride(), t.dtype, t.device, t.layout)
            for t in (grad_output, input, weight)
        )
        output_key = (
            None
            if _outputs is None
            else tuple(
                (t.shape, t.stride(), t.dtype, t.device, t.layout) for t in _outputs
            )
        )
        key = (
            tensor_key,
            (grad_output is input, grad_output is weight, input is weight),
            parameter_key,
            (type(groups), groups),
            (type(transposed), transposed),
            output_key,
        )
        plan = _LAUNCH_PLANS.get(key)
    except (TypeError, AttributeError):
        return _convolution_backward_uncached(*arguments, _outputs=_outputs)
    if plan is not None:
        launches, workspace, bias_shape = plan
        if _outputs is None:
            outputs = (
                _empty_gradient(input) if output_mask[0] else None,
                _empty_gradient(weight) if output_mask[1] else None,
                (
                    torch.empty(bias_shape, dtype=input.dtype, device=input.device)
                    if output_mask[2]
                    else None
                ),
            )
        else:
            outputs = _outputs
        tensors = (grad_output, input, weight, *outputs) + tuple(
            torch.empty(shape, dtype=dtype, device=input.device)
            for shape, dtype in workspace
        )
        with torch_device_fn.device(input.device):
            for kernel, grid, slots, metadata in launches:
                pointers = tuple(tensors[slot] for slot in slots)
                kernel[grid](*pointers, *metadata)
        return outputs
    record = []
    outputs = _convolution_backward_uncached(
        *arguments, _outputs=_outputs, _record=record
    )
    if not record:
        return outputs
    # A resizing .out call cannot reuse the pre-resize output descriptor.
    if _outputs is not None and output_key != tuple(
        (t.shape, t.stride(), t.dtype, t.device, t.layout) for t in _outputs
    ):
        return outputs
    slots = {
        id(t): i
        for i, t in enumerate((grad_output, input, weight, *outputs))
        if t is not None
    }
    launches, workspace = [], []
    for kernel, grid, args, count in record:
        pointer_slots = []
        for tensor in args[:count]:
            slot = slots.get(id(tensor))
            if slot is None:
                slot = 6 + len(workspace)
                slots[id(tensor)] = slot
                workspace.append((tensor.shape, tensor.dtype))
            pointer_slots.append(slot)
        launches.append(
            (kernel, (grid + (1, 1))[:3], tuple(pointer_slots), args[count:])
        )
    if len(_LAUNCH_PLANS) >= 128:
        _LAUNCH_PLANS.pop(next(iter(_LAUNCH_PLANS)))
    _LAUNCH_PLANS[key] = (
        tuple(launches),
        tuple(workspace),
        outputs[2].shape if outputs[2] is not None else (),
    )
    return outputs


def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS CONVOLUTION_BACKWARD")
    return _convolution_backward_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )


def convolution_backward_out(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    *,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS CONVOLUTION_BACKWARD_OUT")
    if not all(output_mask):
        raise RuntimeError(
            "convolution_backward.out: a defined out tensor cannot represent an undefined masked gradient"
        )
    for out in (out0, out1, out2):
        if out.dtype != input.dtype or out.device != input.device:
            raise RuntimeError("convolution_backward.out: output dtype/device mismatch")
    outputs = (out0, out1, out2)
    inputs = (grad_output, input, weight)
    aliases = any(
        torch._C._is_alias_of(out, tensor) for out in outputs for tensor in inputs
    )
    aliases |= any(
        torch._C._is_alias_of(outputs[i], outputs[j])
        for i in range(3)
        for j in range(i)
    )
    if not aliases:
        # Write directly into independent destinations; preserve their strides.
        return _convolution_backward_impl(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
            _outputs=outputs,
        )
    # Aliased outputs retain ATen's compute-first, copy-afterwards semantics.
    results = convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    for out, result in zip((out0, out1, out2), results):
        if result is not None:
            if out.shape != result.shape:
                out.resize_(result.shape)
            if result.numel():
                if any(
                    size > 1 and step == 0
                    for size, step in zip(out.shape, out.stride())
                ):
                    raise RuntimeError("convolution_backward.out: overlapping output")
                with torch_device_fn.device(input.device):
                    _launch_gradient(
                        _copy_gradient,
                        (triton.cdiv(result.numel(), 512),),
                        result,
                        out,
                        result.numel(),
                        *((1,) * (5 - result.ndim) + tuple(result.shape)),
                        *((0,) * (5 - result.ndim) + result.stride()),
                        *((0,) * (5 - out.ndim) + out.stride()),
                        512,
                    )
    return out0, out1, out2
