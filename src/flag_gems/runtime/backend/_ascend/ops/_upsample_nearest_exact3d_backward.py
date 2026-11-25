"""GEMS ``_upsample_nearest_exact3d_backward`` (TLE / Ascend).

Semantics reproduced from PyTorch's
``nearest_neighbor_exact_bw_compute_source_index`` in all three spatial axes:

    src(j) = min(ceil(j * s - 0.5), R)                (output space, fp32)
    grad_input[n, c, i, j, k] = sum grad_output[n, c, od, oh, ow]
        for od in [src_d(i), src_d(i+1)), oh in [...), ow in [...]

with ``s`` the fp32-rounded *backward* scale.  The three per-axis intervals
partition ``[0, R)``, so every ``grad_output`` element is read at most once
and every ``grad_input`` element has exactly one writer: no atomics, no
zero-fill pre-pass, and an empty interval must still store an exact 0.

All index arithmetic is fp32 and index-exact; accumulation is fp32 for
fp16/bf16/fp32 inputs (the reference does the same), int32 for byte input and
fp64 for fp64 input, with the dtype narrowing only on the single store.

Grid / layout design (why this beats a flattened formulation):

* The grid is ``(h tiles, ID, NC slice)`` and every program derives ``n``/``c``
  from a *scalar* ``tl.program_id``.  No ``%`` / ``//`` is ever evaluated on
  lane data -- on this backend ``arith.remsi`` expands to a serial per-lane
  scalar loop, which is exactly what the reference's flattened index pays for.
* A program owns ``BLOCK = TILE_H * BLOCK_W`` flat lanes covering a
  ``TILE_H``-row band of one ``(id)`` input plane, so no tile ever crosses an
  ``id`` boundary.  ``col``/``row`` inside the band come from bit ops on
  ``BLOCK_W`` (a power of two), never from ``remsi``.
* ``batch``/``channel`` are not a serial ``tl.range`` over ``N``: they are the
  third grid axis (the fused ``N*C`` axis), so the whole ``(d, h, w)`` volume is
  covered by independent programs instead of one program walking ``N``.
* The one *eligible* bulk move -- an unmasked, unit-stride, contiguous full
  fp32 tile store -- is issued with ``tle.dsa.to_buffer`` + ``tle.dsa.copy``
  (``FULL`` specialization).  The gathers use per-lane addresses that
  ``tle.dsa.copy`` cannot express, and boundary / narrowing / strided stores
  need a mask that ``tle.dsa.copy`` does not carry, so those keep the plain
  ``triton.language`` path.

Uniform-integer fast paths (two extra kernels, dispatched before the above):

The general kernel pays for the interval search on *every* output element --
profiling showed the gather is a serial per-lane scalar loop (``aiv_scalar``
ratio 0.99) while both DMA pipes sit idle.  But for a uniform integer scale the
three intervals have a closed form, so the search disappears entirely:

* integer ``S >= 1`` with ``O = I*S``  ->  ``src(i) = i*S`` exactly, because
  ``i*S`` is exact in fp32, so ``gi[i] = sum(go[i*S : (i+1)*S])``.  A tile is
  then one *contiguous* slab reshaped ``[TILE_H, S_H, RW, S_W]`` and reduced
  over axis 3 then axis 1: no index math on lane data at all.
* ``s == 1/q`` exactly with ``I = O*q``  ->  ``src(j) = ceil(j/q - 0.5) ->
  floor((2j + q - 1) / 2q)``, whose interval ``[src(i), src(i+1))`` is non-empty
  iff ``(2i + q) mod 2q`` is ``0`` or ``2q-1``, i.e. iff ``i mod q == q//2``.
  So ``gi[i] = go[i//q]`` on a single residue, exactly ``0`` elsewhere -- a
  scatter with no arithmetic.  Gated to ``q in {2, 4}``: only for powers of two
  is ``j/q`` exact in fp32, so only then is the residue the true one.

Both forms are bitwise-identical to this file's general path (verified in a
numerical probe over fp32/fp16/bf16, 8 shape/scale combinations).  For the
downsample the tile is built by DMA rather than in registers -- a full-tile zero
store followed by one bulk copy per surviving row -- because the zero-
interleaved register form silently dropped the duplicate-row mask at ``Q_H = 2``.
"""

import logging
import math
import struct

import torch
import torch_npu  # noqa: F401  (must be imported before triton on this stack)
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _up_ne3d_bwd_kernel(
    go_ptr,
    gi_ptr,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    NC,
    C,
    go_sn,
    go_sc,
    go_sd,
    go_sh,
    go_sw,
    gi_sn,
    gi_sc,
    gi_sd,
    gi_sh,
    gi_sw,
    scale_d,
    scale_h,
    scale_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_INT: tl.constexpr,
    FUSED_NC: tl.constexpr,
    FULL: tl.constexpr,
    ROW_BULK: tl.constexpr,
    ROW_W: tl.constexpr,
    TILE_STRIDE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    LOG_BW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_nc = tl.program_id(2)

    lane = tl.arange(0, BLOCK)
    col = lane & (BLOCK_W - 1)
    row = lane >> LOG_BW

    ih = pid_h * TILE_STRIDE + row
    lane_ok = (ih < IH) & (col < IW)

    # ---- exact nearest-neighbour source intervals (fp32, as ATen) ----
    odf = OD.to(tl.float32)
    ohf = OH.to(tl.float32)
    owf = OW.to(tl.float32)

    xd = pid_d.to(tl.float32)
    sd = tl.minimum(tl.ceil(xd * scale_d - 0.5), odf).to(tl.int32)
    ed = tl.minimum(tl.ceil((xd + 1.0) * scale_d - 0.5), odf).to(tl.int32)

    xh = ih.to(tl.float32)
    sh_f = tl.ceil(xh * scale_h - 0.5)
    eh_f = tl.ceil((xh + 1.0) * scale_h - 0.5)
    sh = tl.where(sh_f < ohf, sh_f, ohf).to(tl.int32)
    eh = tl.where(eh_f < ohf, eh_f, ohf).to(tl.int32)

    xw = col.to(tl.float32)
    sw_f = tl.ceil(xw * scale_w - 0.5)
    ew_f = tl.ceil((xw + 1.0) * scale_w - 0.5)
    sw = tl.where(sw_f < owf, sw_f, owf).to(tl.int32)
    ew = tl.where(ew_f < owf, ew_f, owf).to(tl.int32)

    # ``sd`` / ``sh`` / ``sw`` are already absolute indices into the output
    # volume (``od = sd + kd``), so the depth plane must not be added a second
    # time to the gather base -- only the store side needs ``pid_d``.
    d_store = pid_d * gi_sd

    nc = pid_nc
    if FUSED_NC:
        go_base = nc * go_sc
        gi_base = nc * gi_sc
    else:
        c = nc % C
        n = nc // C
        go_base = n * go_sn + c * go_sc
        gi_base = n * gi_sn + c * gi_sc

    if IS_FP64:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
    elif IS_INT:
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for kd in range(MAX_D):
        od = sd + kd
        md = od < ed
        for kh in range(MAX_H):
            oh = sh + kh
            mh = md & (oh < eh) & lane_ok
            base = go_base + od * go_sd + oh * go_sh
            for kw in range(MAX_W):
                ow = sw + kw
                m = mh & (ow < ew)
                if IS_INT:
                    v = tl.load(go_ptr + base + ow * go_sw, mask=m, other=0)
                else:
                    v = tl.load(go_ptr + base + ow * go_sw, mask=m, other=0.0)
                acc += v.to(acc.dtype)

    if FULL:
        # ``block_w == iw`` with ``gi_sw == 1`` and contiguous ``grad_input``
        # implies ``gi_sh == iw``, so ``ih * gi_sh + col * gi_sw`` is exactly
        # ``pid_h * TILE_STRIDE * gi_sh + lane``: the tile covers one
        # contiguous run of ``BLOCK`` elements starting at the scalar ``base``.
        # Profiling showed the previous per-lane destination expression
        # (``ih*gi_sh + col*gi_sw``) lowered to ``BLOCK`` single-element
        # UB->GM stores even though the addresses were contiguous, so the
        # destination is now a scalar base plus an explicit unit-stride
        # ``arange`` to make the contiguity syntactically visible.  A *scalar*
        # destination pointer cannot be used at all -- ``tile.copy`` to a bare
        # ``!tt.ptr`` fails to lower.
        #
        # The value is narrowed to the stored dtype *before* it enters the UB
        # buffer, so the buffer element type matches ``gi_ptr`` exactly and the
        # move stays a pure transfer.  For fp32 this is the identity.
        base = gi_base + d_store + pid_h * TILE_STRIDE * gi_sh
        dst = base + tl.arange(0, BLOCK)
        y = acc.to(gi_ptr.dtype.element_ty)
        y_ub = tle.dsa.to_buffer(y, space=tle.dsa.ascend.UB)
        tle.dsa.copy(y_ub, gi_ptr + dst, [BLOCK])
    elif ROW_BULK:
        # Contiguous but ``block_w != iw``: the tile is ``TILE_STRIDE`` rows of
        # ``BLOCK_W`` lanes of which only ``ROW_W`` are valid, so a single flat
        # copy would write the masked-off tail.  Each valid row span is
        # contiguous in both the UB buffer and ``grad_input``, so the store
        # becomes one bulk move per row instead of one DMA per element.
        # ``tle.dsa.copy`` carries no mask, which is why the extent is ``ROW_W``
        # and not ``BLOCK_W``.
        #
        # ``ROW_W`` is the *actual* extent ``iw``, not its power-of-two
        # rounding: a probe showed this backend accepts a non-power-of-two
        # ``tl.arange``, and using ``next_pow2(iw)`` would run each row copy
        # past its own row into the next one (and out of bounds on the final
        # row of the final tile).
        #
        # The unroll must be a Python-level one: a ``range`` loop becomes an
        # ``scf.for`` whose induction variable feeds the subview offset, and a
        # loop-carried ``tle.dsa.subview`` offset fails ``CommonIRToHIVM``.
        # ``tl.static_range`` forces the constant-offset unroll instead.
        y = acc.to(gi_ptr.dtype.element_ty)
        y_ub = tle.dsa.to_buffer(y, space=tle.dsa.ascend.UB)
        row_base = gi_base + d_store + pid_h * TILE_STRIDE * gi_sh
        for r in tl.static_range(TILE_STRIDE):
            src = tle.dsa.subview(y_ub, [r * BLOCK_W], [ROW_W], [1])
            dst = gi_ptr + (row_base + r * gi_sh) + tl.arange(0, ROW_W)
            tle.dsa.copy(src, dst, [ROW_W])
    else:
        dst = gi_base + d_store + ih * gi_sh + col * gi_sw
        y = acc.to(gi_ptr.dtype.element_ty)
        tl.store(gi_ptr + dst, y, mask=lane_ok)


# Deterministic host dispatch.  Tile sizes are derived from the workload
# shape only; no physical core count is baked in.
_TARGET_TILE = 2048
_MIN_BLOCK_W = 16

# Uniform-integer fast paths.  ``_UP_CHUNK_CAP`` bounds the single contiguous
# slab load, ``_DN_BLOCK_CAP`` the zero/widen tile of the downsample form.
_UP_CHUNK_CAP = 4096
_UP_BLOCK_CAP = 4096
_DN_ROW_CAP = 1024
_DN_BLOCK_CAP = 4096


@triton.jit
def _ne3d_bwd_up_kernel(
    go_ptr,
    gi_ptr,
    GO_PLANE,
    GI_PLANE,
    GO_SD,
    GO_SH,
    GI_SD,
    GI_SH,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    TILE_H: tl.constexpr,
    RW: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Integer upsample ``S >= 1``: ``gi[i] = sum(go[i*S : (i+1)*S])``.

    ``i*S`` is exact in fp32, so ``src(i) = i*S`` and the source interval is the
    fixed-length window ``[i*S, (i+1)*S)``.  A whole ``TILE_H``-row band of one
    input plane is therefore one contiguous run of ``CHUNK`` output elements,
    loaded in a single unmasked stride-1 ``tl.load`` and reduced twice over the
    reshaped ``[TILE_H, S_H, RW, S_W]`` view.  No per-lane index arithmetic, and
    no ``arith.remsi``: the two inner extents are the tile shape.
    """
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_nc = tl.program_id(2)

    ih0 = pid_h * TILE_H
    go_base = pid_nc * GO_PLANE
    gi_base = pid_nc * GI_PLANE

    arr = tl.arange(0, CHUNK)
    acc = tl.zeros([TILE_H, RW], dtype=tl.float32)
    for d in tl.static_range(S_D):
        off = go_base + (pid_d * S_D + d) * GO_SD + (ih0 * S_H) * GO_SH
        v = tl.load(go_ptr + off + arr).to(tl.float32)
        t = tl.sum(tl.reshape(v, [TILE_H, S_H, RW, S_W]), axis=3)
        acc += tl.sum(t, axis=1)

    y = tl.reshape(acc, [BLOCK]).to(gi_ptr.dtype.element_ty)
    y_ub = tle.dsa.to_buffer(y, space=tle.dsa.ascend.UB)
    gi_off = gi_base + pid_d * GI_SD + ih0 * GI_SH
    tle.dsa.copy(y_ub, gi_ptr + gi_off + tl.arange(0, BLOCK), [BLOCK])


@triton.jit
def _ne3d_bwd_dn_kernel(
    go_ptr,
    gi_ptr,
    GO_PLANE,
    GI_PLANE,
    GO_SD,
    GO_SH,
    GI_SD,
    GI_SH,
    Q_D: tl.constexpr,
    Q_H: tl.constexpr,
    Q_W: tl.constexpr,
    RHO_D: tl.constexpr,
    RHO_H: tl.constexpr,
    RHO_W: tl.constexpr,
    TILE_H: tl.constexpr,
    ROWS: tl.constexpr,
    OW_C: tl.constexpr,
    RW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Uniform downsample ``s == 1/q``, ``q in {2, 4}``.

    ``src(j) = floor((2j + q - 1) / 2q)``, so ``[src(i), src(i+1))`` is
    non-empty exactly when ``i mod q == q//2``, and then it holds the single
    element ``i//q``.  ``gi[i]`` is therefore ``go[i//q]`` on one residue per
    axis and exactly 0 everywhere else.  The tile is produced by DMA in two
    passes -- an unmasked bulk zero store over the whole tile, then one bulk
    copy per surviving row placed at its residue row -- rather than by building
    the zero-interleaved tile in registers, because the register form silently
    dropped the duplicate-row mask at ``Q_H = 2``.
    """
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_nc = tl.program_id(2)

    ih0 = pid_h * TILE_H
    go_base = pid_nc * GO_PLANE
    gi_base = pid_nc * GI_PLANE
    gi_off = gi_base + pid_d * GI_SD + ih0 * GI_SH

    # pass 1: the whole tile is zero unless this output d-plane is a survivor
    z_ub = tle.dsa.to_buffer(
        tl.zeros([BLOCK], dtype=tl.float32).to(gi_ptr.dtype.element_ty),
        space=tle.dsa.ascend.UB,
    )
    tle.dsa.copy(z_ub, gi_ptr + gi_off + tl.arange(0, BLOCK), [BLOCK])

    if (pid_d % Q_D) == RHO_D:
        # ``ROWS`` compact rows are contiguous in grad_output (``OW_C == OW``),
        # so this is one stride-1 load of the surviving source rows.
        off = go_base + (pid_d // Q_D) * GO_SD + (ih0 // Q_H) * GO_SH
        g = tl.load(go_ptr + off + tl.arange(0, ROWS * OW_C))
        g2 = tl.reshape(g, [ROWS, OW_C])
        if Q_W > 1:
            # widen each compact row to ``RW`` in registers: the column residue
            # is ``RHO_W``, everything else is an exact 0
            wsel = tl.arange(0, Q_W)[None, None, :] == RHO_W
            v = tl.reshape(
                tl.where(wsel, tl.reshape(g2, [ROWS, OW_C, 1]), 0.0), [ROWS, RW]
            )
        else:
            v = g2
        v_ub = tle.dsa.to_buffer(
            tl.reshape(v, [ROWS * RW]).to(gi_ptr.dtype.element_ty),
            space=tle.dsa.ascend.UB,
        )
        # pass 2: one contiguous bulk row copy per surviving h row; the offset
        # must be a compile-time constant, hence ``tl.static_range``
        for r in tl.static_range(ROWS):
            src = tle.dsa.subview(v_ub, [r * RW], [RW], [1])
            dst = gi_ptr + gi_off + (r * Q_H + RHO_H) * GI_SH + tl.arange(0, RW)
            tle.dsa.copy(src, dst, [RW])


def _next_pow2(value):
    p = 1
    while p < value:
        p <<= 1
    return p


def _log2(value):
    n = 0
    while (1 << n) < value:
        n += 1
    return n


def _choose_tile(iw, ih):
    """(BLOCK_W, TILE_H) for the input spatial extents."""
    block_w = max(_MIN_BLOCK_W, _next_pow2(max(1, iw)))
    cap = max(1, _TARGET_TILE // block_w)
    limit = min(cap, ih) if ih > 0 else 1
    tile_h = 1
    t = 1
    while t <= limit:
        if ih % t == 0:
            tile_h = t
        t <<= 1
    return block_w, tile_h


def _as_float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _as_tuple(value):
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _rounded_scale(scale, out_extent, in_extent):
    """The fp32-rounded backward scale exactly as ATen derives it."""
    if scale is not None and float(scale) > 0.0:
        return _as_float32(float(scale))
    if in_extent <= 0:
        return 0.0
    return _as_float32(out_extent / in_extent)


def _uniform_class(scale, out_extent, in_extent):
    """``("up", S)`` / ``("dn", q)`` / ``None`` for one axis.

    Only the exactly-representable uniform cases qualify.  Integer ``S`` is
    exact for any ``S`` because ``i*S`` is; ``1/q`` is exact in fp32 only for a
    power-of-two ``q``, so the downsample form is restricted to ``q in {2, 4}``.
    """
    s = _rounded_scale(scale, out_extent, in_extent)
    if in_extent <= 0 or out_extent <= 0:
        return None
    sv = int(s)
    if s >= 1.0 and float(sv) == s and in_extent * sv == out_extent:
        return ("up", sv)
    if 0.0 < s < 1.0 and in_extent % out_extent == 0:
        q = in_extent // out_extent
        if q in (2, 4) and s == _as_float32(1.0 / q):
            return ("dn", q)
    return None


def _divisor_at_most(n, cap):
    """Largest divisor of ``n`` that is ``<= cap`` (at least 1)."""
    if n <= 0:
        return 1
    best = 1
    t = 1
    while t <= cap and t <= n:
        if n % t == 0:
            best = t
        t += 1
    return best


def _launch_up(grad_output, grad_input, sizes, kinds, strides):
    n, c, id_, ih, iw = sizes
    od, oh, ow = kinds["out"]
    sd, sh, sw = kinds["S"]
    gi_sn, gi_sc, gi_sd, gi_sh, gi_sw = strides["gi"]
    go_sn, go_sc, go_sd, go_sh, go_sw = strides["go"]
    del n, c, od, oh, gi_sn, go_sn, gi_sw, go_sw

    if iw > _UP_BLOCK_CAP or sh * ow > _UP_CHUNK_CAP:
        return False
    tile_h = _divisor_at_most(ih, min(_UP_CHUNK_CAP // (sh * ow), _UP_BLOCK_CAP // iw))
    chunk = tile_h * sh * ow
    block = tile_h * iw
    grid = (ih // tile_h, id_, grad_input.shape[0] * grad_input.shape[1])
    _ne3d_bwd_up_kernel[grid](
        grad_output,
        grad_input,
        go_sc,
        gi_sc,
        go_sd,
        go_sh,
        gi_sd,
        gi_sh,
        S_D=sd,
        S_H=sh,
        S_W=sw,
        TILE_H=tile_h,
        RW=iw,
        CHUNK=chunk,
        BLOCK=block,
    )
    return True


def _launch_dn(grad_output, grad_input, sizes, kinds, strides):
    n, c, id_, ih, iw = sizes
    qd, qh, qw = kinds["q"]
    gi_sn, gi_sc, gi_sd, gi_sh, gi_sw = strides["gi"]
    go_sn, go_sc, go_sd, go_sh, go_sw = strides["go"]
    del n, c, gi_sn, go_sn, gi_sw, go_sw

    ow_c = iw // qw
    cap = min(_DN_ROW_CAP // max(1, ow_c), _DN_BLOCK_CAP // (qh * iw))
    if cap < 1:
        return False
    rows = _divisor_at_most(ih // qh, cap)
    tile_h = rows * qh
    block = tile_h * iw
    grid = (ih // tile_h, id_, grad_input.shape[0] * grad_input.shape[1])
    _ne3d_bwd_dn_kernel[grid](
        grad_output,
        grad_input,
        go_sc,
        gi_sc,
        go_sd,
        go_sh,
        gi_sd,
        gi_sh,
        Q_D=qd,
        Q_H=qh,
        Q_W=qw,
        RHO_D=qd // 2,
        RHO_H=qh // 2,
        RHO_W=qw // 2,
        TILE_H=tile_h,
        ROWS=rows,
        OW_C=ow_c,
        RW=iw,
        BLOCK=block,
    )
    return True


def _scale_and_bound(scale, out_extent, in_extent):
    if scale is not None and float(scale) > 0.0:
        exact = float(scale)
    else:
        exact = out_extent / in_extent
    rounded = _as_float32(exact)
    bound = max(1, math.ceil(exact), math.ceil(rounded))
    return rounded, bound


def _validate(grad_output, output_size, input_size):
    if grad_output.ndim != 5:
        raise RuntimeError("grad_output must be a 5D tensor")
    output_size = _as_tuple(output_size)
    input_size = _as_tuple(input_size)
    if len(output_size) != 3:
        raise RuntimeError("output_size must contain three values")
    if len(input_size) != 5:
        raise RuntimeError("input_size must contain five values")

    od, oh, ow = (int(v) for v in output_size)
    n, c, id_, ih, iw = (int(v) for v in input_size)
    expected = (n, c, od, oh, ow)
    if tuple(grad_output.shape) != expected:
        raise RuntimeError(
            f"grad_output shape {tuple(grad_output.shape)} does not match "
            f"expected shape {expected}"
        )
    return n, c, id_, ih, iw, od, oh, ow


def _impl(
    grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input
):
    n, c, id_, ih, iw, od, oh, ow = _validate(grad_output, output_size, input_size)

    if grad_input is None:
        grad_input = torch.empty(
            (n, c, id_, ih, iw),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if not isinstance(grad_input, torch.Tensor):
            raise TypeError("grad_input must be a Tensor")
        if grad_input.device != grad_output.device:
            raise RuntimeError("grad_input and grad_output must be on the same device")
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype} instead"
            )
        if tuple(grad_input.shape) != (n, c, id_, ih, iw):
            grad_input.resize_((n, c, id_, ih, iw))

    if grad_input.numel() == 0:
        return grad_input

    # ---- uniform-integer fast path ----------------------------------------
    # Both fast kernels accumulate in fp32 and need unit-stride source slabs,
    # so they are gated on the three fixed-width float dtypes and on both
    # tensors being contiguous.  The classification is exact (it reproduces
    # ATen's fp32-rounded scale before deciding), so a miss just falls through
    # to the general kernel.
    go_sn, go_sc, go_sd, go_sh, go_sw = grad_output.stride()
    gi_sn, gi_sc, gi_sd, gi_sh, gi_sw = grad_input.stride()
    if (
        grad_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and grad_input.is_contiguous()
        and grad_output.is_contiguous()
        and gi_sw == 1
        and go_sw == 1
    ):
        axis = [
            _uniform_class(sc, o, i)
            for sc, o, i in (
                (scales_d, od, id_),
                (scales_h, oh, ih),
                (scales_w, ow, iw),
            )
        ]
        kinds = [a[0] if a else None for a in axis]
        if kinds[0] is not None and kinds == [kinds[0]] * 3:
            strides = {
                "go": (go_sn, go_sc, go_sd, go_sh, go_sw),
                "gi": (gi_sn, gi_sc, gi_sd, gi_sh, gi_sw),
            }
            sizes = (n, c, id_, ih, iw)
            if kinds[0] == "up":
                kinds = {"S": tuple(a[1] for a in axis), "out": (od, oh, ow)}
                if _launch_up(grad_output, grad_input, sizes, kinds, strides):
                    return grad_input
            else:
                kinds = {"q": tuple(a[1] for a in axis)}
                if _launch_dn(grad_output, grad_input, sizes, kinds, strides):
                    return grad_input

    scale_d, max_d = _scale_and_bound(scales_d, od, id_)
    scale_h, max_h = _scale_and_bound(scales_h, oh, ih)
    scale_w, max_w = _scale_and_bound(scales_w, ow, iw)

    block_w, tile_h = _choose_tile(iw, ih)
    block = block_w * tile_h
    h_tiles = (ih + tile_h - 1) // tile_h
    nc = n * c

    fused_nc = (go_sn == c * go_sc) and (gi_sn == c * gi_sc)

    is_fp64 = grad_output.dtype == torch.float64
    is_int = not grad_output.is_floating_point()
    # Full unmasked contiguous tile store: the value is narrowed to the stored
    # dtype before it enters the UB buffer, so any fixed-width float dtype can
    # take the TLE bulk move (fp64/uint8 keep the masked ``tl.store``).
    full = (
        block_w == iw
        and tile_h <= ih
        and ih % tile_h == 0
        and gi_sw == 1
        and grad_input.is_contiguous()
        and not is_fp64
        and not is_int
    )
    # Same store, but ``block_w != iw``: each tile row is still a contiguous
    # run of ``iw`` elements, so the store becomes one bulk move per row
    # instead of one DMA per element.  ``row_w`` is ``iw`` itself: the backend
    # accepts a non-power-of-two ``tl.arange`` (probe Mode 1), whereas rounding
    # it up to ``next_pow2(iw)`` would run every row copy past its own row.
    row_w = iw
    row_bulk = (
        not full
        and 1 <= row_w <= block_w
        and tile_h <= ih
        and ih % tile_h == 0
        and gi_sw == 1
        and grad_input.is_contiguous()
        and not is_fp64
        and not is_int
    )

    grid = (h_tiles, id_, nc)
    _up_ne3d_bwd_kernel[grid](
        grad_output,
        grad_input,
        id_,
        ih,
        iw,
        od,
        oh,
        ow,
        nc,
        c,
        go_sn,
        go_sc,
        go_sd,
        go_sh,
        go_sw,
        gi_sn,
        gi_sc,
        gi_sd,
        gi_sh,
        gi_sw,
        scale_d,
        scale_h,
        scale_w,
        MAX_D=max_d,
        MAX_H=max_h,
        MAX_W=max_w,
        IS_FP64=is_fp64,
        IS_INT=is_int,
        FUSED_NC=fused_nc,
        FULL=full,
        ROW_BULK=row_bulk,
        ROW_W=row_w,
        TILE_STRIDE=tile_h,
        BLOCK_W=block_w,
        LOG_BW=_log2(block_w),
        BLOCK=block,
    )
    return grad_input


def _upsample_nearest_exact3d_backward(
    grad_output,
    output_size,
    input_size,
    scales_d=None,
    scales_h=None,
    scales_w=None,
):
    """Allocating form of ``_upsample_nearest_exact3d_backward``."""
    logger.debug("GEMS _UPSAMPLE_NEAREST_EXACT3D_BACKWARD")
    return _impl(
        grad_output,
        output_size,
        input_size,
        scales_d,
        scales_h,
        scales_w,
        None,
    )
