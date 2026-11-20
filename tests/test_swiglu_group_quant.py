# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Accuracy tests for ``swiglu_group_quant`` (SwiGLU fused with group quant).

The reference is a torch transcription of the vendor kernel
(``aclnnSwigluGroupQuant``, ops-nn arch35), computed step by step in fp32
because the order matters: the operator divides by ``1 + exp(-x0)`` rather than
multiplying by a sigmoid, multiplies the amax by a rounded ``1/fp8max`` constant
instead of dividing, and normalises ``-0.0`` to ``+0.0`` after the optional
weight -- each of which is visible in the output bytes.

Two tolerances exist because the reference is torch and the operator is Triton:
``aclnnExp`` and ``tl.exp`` agree to the bit, but torch's fp32 SwiGLU on NPU
differs from both by up to 3 ULP, which can flip an fp8 payload at a
round-to-nearest tie or an fp32 scale in its last bits.  Differences are
therefore allowed only at single-step magnitude and only at a low rate.
``test_swiglu_group_quant_matches_vendor`` compares byte-for-byte instead, and
runs wherever ``cann_ops_nn`` is importable.
"""

import os
import sys

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

requires_swiglu_group_quant = pytest.mark.skipif(
    "swiglu_group_quant" not in vars(flag_gems),
    reason="backend does not export swiglu_group_quant",
)


# The operator multiplies by a *rounded* 1/fp8max instead of dividing by
# fp8max (INV_FP8_E4M3_MAX_VALUE / INV_FP8_E5M2_MAX_VALUE in the vendor
# source); reproducing the exact fp32 constant decides the scale exponent at
# the boundary.
INV_FP8_MAX = {
    torch.float8_e4m3fn: torch.tensor([0x3B124925], dtype=torch.int32)
    .view(torch.float32)
    .item(),
    torch.float8_e5m2: torch.tensor([0x37924925], dtype=torch.int32)
    .view(torch.float32)
    .item(),
}
FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
BLOCK = {0: 128, 1: 32}
EPS = 1e-4
# Payload bytes may differ from the torch reference only by one fp8 step, and
# only on this fraction of the elements.  In practice they do not differ at all
# (measured: 0 of 524288 bytes at T=256 D=4096); the band is here because the
# reference's exp is a few ULP off the kernel's and could flip a tie.
TIE_RATE = 1e-3
# fp32 scales (mode 0, round_scale=False) carry the SwiGLU's fp32 rounding
# straight through, so they are compared in ULP: 308 of 4096 scales differ from
# the torch reference there, by up to 3 ULP.
SCALE_ULP = 8

DST_TYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# aclnn dst_type codes, for the vendor cross-check.
DST_CODE = {torch.float8_e4m3fn: 292, torch.float8_e5m2: 291}


def _vendor_op():
    """``cann_ops_nn.swiglu_group_quant``, or None when CANN does not ship it.

    It is the only entry point that reaches the fp8 modes from python:
    ``torch_npu.npu_swiglu_group_quant`` only allocates outputs for the
    HiFloat8 modes, and vllm-ascend's ``_C_ascend`` op needs its own build.
    """
    cann = os.environ.get("ASCEND_TOOLKIT_LATEST_HOME") or os.environ.get(
        "ASCEND_HOME_PATH", ""
    )
    site = os.path.join(cann, "python", "site-packages")
    if cann and os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)
    try:
        import cann_ops_nn  # noqa: F401  registers torch.ops.cann_ops_nn

        return torch.ops.cann_ops_nn.swiglu_group_quant
    except Exception:
        return None


def _bits_to_f32(i):
    """``((i & 0xFF) << 23)`` viewed as fp32, i.e. 2^(i-127) for i in [1, 254]."""
    return ((i & 0xFF) << 23).to(torch.int32).view(torch.float32)


def swiglu_group_quant_ref(
    x,
    weight=None,
    group_index=None,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    round_scale=None,
    clamp_limit=-1.0,
    output_origin=False,
    y_out=None,
    scale_out=None,
    y_origin_out=None,
):
    """Reference ``(y, scale, y_origin)`` for quant_mode 0 and 1.

    ``y_out`` / ``scale_out`` / ``y_origin_out`` are the pre-existing output
    buffers: with ``group_index`` the operator writes only the first
    ``sum(group_index)`` rows and leaves the rest of the buffer untouched, so
    checking the tail needs the caller's buffer rather than zeros.
    """
    if round_scale is None:
        round_scale = quant_mode == 1
    shape = tuple(x.shape)
    D = shape[-1]
    H = D // 2
    B = BLOCK[quant_mode]
    nblk = -(-H // B)
    T = x.numel() // D

    xf = x.reshape(T, D).to(torch.float32)
    x0, x1 = xf[:, :H], xf[:, H:]
    if clamp_limit is not None and float(clamp_limit) != -1.0:
        c = float(clamp_limit)
        # The gate half is clamped above only; the linear half symmetrically.
        x0 = torch.minimum(x0, torch.tensor(c, device=x.device))
        x1 = x1.clamp(-c, c)

    y_origin = (x0 / (1.0 + torch.exp(-x0))) * x1
    y = y_origin
    if weight is not None:
        y = y * weight.reshape(T, 1).to(torch.float32)
    # -0.0 + 0.0 == +0.0: the operator's `Add(y, y, zero)` normalises the sign
    # of zero, which is visible in the fp8 payload byte.  y_origin keeps it.
    y = y + 0.0

    pad = nblk * B - H
    yb = torch.nn.functional.pad(y, (0, pad)).reshape(T, nblk, B)

    if quant_mode == 0:
        # inf/nan are masked out of the amax (`y * 0 != y * 0` in the kernel).
        amax = torch.where(torch.isfinite(yb), yb.abs(), torch.zeros_like(yb))
        amax = amax.amax(-1)
    else:
        amax = yb.abs().amax(-1)
    amax = amax.clamp_min(EPS)

    sf = amax * INV_FP8_MAX[dst_type]
    if round_scale:
        bits = sf.view(torch.int32)
        # ceil(log2(sf)) off the fp32 fields.
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        inv = _bits_to_f32(127 - e)
        scale = (e + 127).to(torch.uint8) if quant_mode == 1 else _bits_to_f32(e + 127)
    else:
        # A true division here, not 1/sf.
        inv = FP8_MAX[dst_type] / amax
        scale = sf

    ys = yb * inv[:, :, None]
    if quant_mode == 0:
        # A scaled value that went non-finite falls back to the unscaled one.
        ys = torch.where(torch.isfinite(ys), ys, yb)
    yq = ys.reshape(T, nblk * B)[:, :H].to(dst_type)

    if quant_mode == 1:
        if nblk % 2:
            scale = torch.nn.functional.pad(scale, (0, 1))
        scale = scale.reshape(*shape[:-1], (nblk + 1) // 2, 2)
    else:
        scale = scale.reshape(*shape[:-1], nblk)
    yq = yq.reshape(*shape[:-1], H)
    yo = y_origin.reshape(*shape[:-1], H).to(x.dtype) if output_origin else None

    real = T
    if group_index is not None:
        real = min(max(int(group_index.sum().item()), 0), T)
    if real < T:

        def tail(v, buf):
            if v is None:
                return None
            if buf is None:
                v = v.clone()
                # fp8 / e8m0 have no fill_ kernel; zero the bytes instead.
                if v.dtype.itemsize == 1:
                    v.view(torch.uint8).reshape(T, -1)[real:] = 0
                else:
                    v.reshape(T, -1)[real:] = 0
                return v
            out = buf.clone()
            out.reshape(T, -1)[:real] = v.reshape(T, -1)[:real]
            return out

        yq = tail(yq, y_out)
        scale = tail(scale, scale_out)
        yo = tail(yo, y_origin_out)
    return yq, scale, yo


def _rows_of(t, T):
    """Host view of ``t`` as ``[T, -1]`` integer bit patterns.

    Single-byte dtypes (both fp8 types, and the vendor's ``float8_e8m0fnu``
    scale) go through ``uint8``: torch refuses to promote them, and the bytes
    are what is being compared anyway.
    """
    h = t.cpu()
    if h.dtype.itemsize == 1:
        h = h.view(torch.uint8)
    elif h.dtype in (torch.bfloat16, torch.float16):
        h = h.view(torch.int16)
    elif h.dtype == torch.float32:
        h = h.view(torch.int32)
    return h.reshape(T, -1)


def _assert_bits_adjacent(actual, expected, T, name, rows=None, rate=TIE_RATE):
    """Allow only single-step differences, and only on ``rate`` of the elements.

    One step in the bit pattern is one representable value: fp8 and fp16/bf16
    encodings are monotonic in magnitude, so a sign flip or a dropped zero sign
    shows up as a large difference and is rejected.
    """
    a, b = _rows_of(actual, T), _rows_of(expected, T)
    if rows is not None:
        a, b = a[:rows], b[:rows]
    if a.numel() == 0:
        return
    diff = (a.to(torch.int64) - b.to(torch.int64)).abs()
    worst = int(diff.max())
    assert worst <= 1, (
        f"{name}: {int((diff > 1).sum())}/{diff.numel()} elements differ by "
        f"more than one representable step (max {worst})"
    )
    differing = int((diff != 0).sum())
    assert differing <= max(1, int(rate * diff.numel())), (
        f"{name}: {differing}/{diff.numel()} elements differ by one step, "
        f"above the {rate} tie rate"
    )


def _assert_bits_equal(actual, expected, T, name, rows=None):
    a, b = _rows_of(actual, T), _rows_of(expected, T)
    if rows is not None:
        a, b = a[:rows], b[:rows]
    bad = int((a != b).sum())
    assert bad == 0, f"{name}: {bad}/{a.numel()} values differ"


def _assert_ulp(actual, expected, T, name, rows=None, ulp=SCALE_ULP):
    a, b = _rows_of(actual, T), _rows_of(expected, T)
    if rows is not None:
        a, b = a[:rows], b[:rows]
    diff = (a.to(torch.int64) - b.to(torch.int64)).abs()
    assert int(diff.max()) <= ulp, (
        f"{name}: {int((diff > ulp).sum())}/{diff.numel()} values over "
        f"{ulp} ULP (max {int(diff.max())})"
    )


def _make_x(T, D, dtype, device, seed=0):
    torch.manual_seed(seed)
    return torch.randn(T, D, device=device, dtype=dtype) * 3


def _make_zero_sign_x(T, D, dtype, device):
    """x that lands on every sign-of-zero case the operator distinguishes.

    An exact ``-0.0`` result must come out as payload ``0x00`` (the operator
    adds a zero to normalise it), while a tiny negative that merely underflows
    keeps its sign (``0x80``).  Random data produces neither.  One of each case
    per 32-element block, so every block of either quant mode contains all of
    them plus a normal amax to quantise against.  The underflow case stays
    within fp16's range so that the pattern means the same in both input
    dtypes.
    """
    H = D // 2
    x = torch.zeros(T, D, dtype=dtype)
    x[:, 0::32] = 0.0
    x[:, H + 0 :: 32] = -1.0  # y = -0.0
    x[:, 1::32] = -0.0
    x[:, H + 1 :: 32] = 1.0  # y = -0.0
    x[:, 2::32] = -0.0
    x[:, H + 2 :: 32] = -1.0  # y = +0.0
    x[:, 3::32] = 1e-4
    x[:, H + 3 :: 32] = -1e-4  # y = -5e-9: underflows fp8, keeps its sign
    x[:, 4::32] = 10.0
    x[:, H + 4 :: 32] = 0.4  # y = 4.0, sets the amax
    return x.to(device)


def _check(
    x,
    weight=None,
    group_index=None,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    round_scale=None,
    clamp_limit=-1.0,
    output_origin=False,
    out=None,
    scale=None,
    y_origin=None,
):
    """Run the operator and the reference on the same inputs and compare.

    Output buffers are snapshotted first: the operator writes them in place and
    leaves the rows past ``sum(group_index)`` alone, so the reference needs the
    contents from before the call to predict that tail.
    """
    T = x.numel() // x.shape[-1]
    if round_scale is None:
        round_scale = quant_mode == 1
    before = tuple(b.clone() if b is not None else None for b in (out, scale, y_origin))

    y, s, yo = flag_gems.swiglu_group_quant(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        0,
        round_scale,
        clamp_limit,
        output_origin,
        out=out,
        scale=scale,
        y_origin=y_origin,
    )
    ry, rs, ryo = swiglu_group_quant_ref(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        round_scale,
        clamp_limit,
        output_origin,
        y_out=before[0],
        scale_out=before[1],
        y_origin_out=before[2],
    )
    assert tuple(y.shape) == tuple(ry.shape)
    assert tuple(s.shape) == tuple(rs.shape)
    assert y.dtype == dst_type and s.dtype == rs.dtype
    _assert_bits_adjacent(y, ry, T, "y")
    if quant_mode == 0 and not round_scale:
        _assert_ulp(s, rs, T, "scale")
    else:
        _assert_bits_equal(s, rs, T, "scale")
    if output_origin:
        assert tuple(yo.shape) == tuple(ryo.shape)
        _assert_bits_adjacent(yo, ryo, T, "y_origin")
    else:
        assert yo is None
    return y, s, yo


SHAPES = [
    (1, 4096),  # one row, H = 2048 = DeepSeek-V3 moe_intermediate_size
    (7, 4096),  # T not a multiple of the tile's row count
    (64, 256),  # H = 128, the smallest legal hidden size
    (64, 2560),  # H = 1280, not a power of two -> overlapping last chunk
    (100, 3584),  # H = 1792, overlap plus a row mask
    (256, 4096),
    (1023, 1024),
]
if QUICK_MODE:
    SHAPES = [(64, 2560)]


MODES = [(1, True), (0, True), (0, False)]
DTYPES = [torch.bfloat16, torch.float16]
if QUICK_MODE:
    MODES = [(1, True)]
    DTYPES = [torch.bfloat16]


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize("shape", SHAPES)
def test_swiglu_group_quant(shape):
    T, D = shape
    x = _make_x(T, D, torch.bfloat16, flag_gems.device)
    _check(x)


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dst_type", DST_TYPES)
@pytest.mark.parametrize("quant_mode,round_scale", MODES)
def test_swiglu_group_quant_modes(dtype, dst_type, quant_mode, round_scale):
    """Both input dtypes, both fp8 types, and all three scale forms.

    Several specializations therefore get built in one process, which is the
    case a plan that bound the wrong compiled kernel would fail: Triton's cache
    only grows on a miss, so picking the newest entry would hand a mode-0
    kernel to a mode-1 plan.
    """
    x = _make_x(64, 4096, dtype, flag_gems.device)
    _check(x, dst_type=dst_type, quant_mode=quant_mode, round_scale=round_scale)


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize("quant_mode", [1] if QUICK_MODE else [1, 0])
@pytest.mark.parametrize(
    "use_weight,clamp_limit,output_origin",
    [
        (True, -1.0, False),
        (False, 7.0, False),
        (False, 0.5, False),  # clamps nearly everything
        (False, -1.0, True),
        (True, 2.5, True),
    ],
)
def test_swiglu_group_quant_optional_inputs(
    quant_mode, use_weight, clamp_limit, output_origin
):
    T, D = 64, 4096
    x = _make_x(T, D, torch.bfloat16, flag_gems.device)
    weight = (
        torch.rand(T, device=flag_gems.device, dtype=torch.float32)
        if use_weight
        else None
    )
    _check(
        x,
        weight=weight,
        quant_mode=quant_mode,
        clamp_limit=clamp_limit,
        output_origin=output_origin,
    )


def _junk_buffers(T, D, dst_type, quant_mode, dtype, device, output_origin):
    """Output buffers pre-filled with junk, so an untouched tail is visible."""
    H = D // 2
    nblk = -(-H // BLOCK[quant_mode])
    out = torch.empty((T, H), dtype=dst_type, device=device)
    out.view(torch.uint8).random_(0, 256)
    if quant_mode == 1:
        scale = torch.zeros((T, (nblk + 1) // 2, 2), dtype=torch.uint8, device=device)
        scale.random_(0, 256)
    else:
        scale = torch.empty((T, nblk), dtype=torch.float32, device=device).uniform_(
            1.0, 2.0
        )
    y_origin = (
        torch.empty((T, H), dtype=dtype, device=device).normal_()
        if output_origin
        else None
    )
    return out, scale, y_origin


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize(
    "T,groups,quant_mode,use_weight,output_origin",
    (
        [(256, 8, 1, False, False)]
        if QUICK_MODE
        else [
            (256, 1, 1, False, False),
            (256, 8, 1, False, False),
            (256, 4, 1, True, True),
            (256, 4, 0, False, False),
            (100, 3, 1, False, False),  # H = 1280 overlap plus a dynamic row count
        ]
    ),
)
def test_swiglu_group_quant_group_index(
    T, groups, quant_mode, use_weight, output_origin
):
    """Only the first ``sum(group_index)`` rows are written; the tail is kept."""
    D = 2560 if T == 100 else 4096
    dtype = torch.bfloat16
    x = _make_x(T, D, dtype, flag_gems.device)
    weight = (
        torch.rand(T, device=flag_gems.device, dtype=torch.float32)
        if use_weight
        else None
    )
    real = max(1, int(0.7 * T))
    per = real // groups
    counts = [per] * groups
    counts[-1] += real - per * groups
    group_index = torch.tensor(counts, device=flag_gems.device, dtype=torch.int64)
    real = min(sum(counts), T)

    out, scale, y_origin = _junk_buffers(
        T, D, torch.float8_e4m3fn, quant_mode, dtype, flag_gems.device, output_origin
    )
    keep = tuple(b.clone() if b is not None else None for b in (out, scale, y_origin))

    _check(
        x,
        weight=weight,
        group_index=group_index,
        quant_mode=quant_mode,
        output_origin=output_origin,
        out=out,
        scale=scale,
        y_origin=y_origin,
    )
    for name, got, kept in zip(
        ("y", "scale", "y_origin"), (out, scale, y_origin), keep
    ):
        if got is None:
            continue
        a, b = _rows_of(got, T)[real:], _rows_of(kept, T)[real:]
        assert torch.equal(a, b), f"{name}: rows {real}..{T} were overwritten"


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_mode,round_scale", MODES)
def test_swiglu_group_quant_sign_of_zero(dtype, quant_mode, round_scale):
    """``-0.0`` quantises to ``0x00``, an underflowing negative keeps ``0x80``.

    ``y_origin`` is written before the fixup, so it keeps the ``-0.0``.
    """
    T, D = 64, 4096
    x = _make_zero_sign_x(T, D, dtype, flag_gems.device)
    y, _, yo = _check(
        x, quant_mode=quant_mode, round_scale=round_scale, output_origin=True
    )
    H = D // 2
    payload = y.view(torch.uint8).cpu()
    # -0.0 * -1.0 and -0.0 * 1.0 both give -0.0 -> normalised to +0.0.
    assert int(payload[0, 0]) == 0x00
    assert int(payload[0, 1]) == 0x00
    assert int(payload[0, 2]) == 0x00
    # 1e-20 * -1.0 underflows the fp8 range but keeps its sign.
    assert int(payload[0, 3]) == 0x80
    # y_origin keeps the signed zero it was given.
    origin_bits = yo.view(torch.int16).cpu()
    assert int(origin_bits[0, 0]) == -0x8000  # 0x8000, i.e. -0.0
    assert tuple(yo.shape) == (T, H)


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize(
    "shape", [(3, 5, 1024)] if QUICK_MODE else [(3, 5, 1024), (2, 3, 4, 2048)]
)
def test_swiglu_group_quant_higher_rank(shape):
    """Leading dims are folded into the token axis; the scale keeps them."""
    x = _make_x(1, 1, torch.bfloat16, flag_gems.device)  # seeds the generator
    x = torch.randn(shape, device=flag_gems.device, dtype=torch.bfloat16) * 3
    y, s, _ = _check(x)
    H = shape[-1] // 2
    nblk = -(-H // BLOCK[1])
    assert tuple(y.shape) == tuple(shape[:-1]) + (H,)
    assert tuple(s.shape) == tuple(shape[:-1]) + ((nblk + 1) // 2, 2)


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
def test_swiglu_group_quant_rejects_unsupported():
    """The constraints the vendor also enforces, raised on the host instead."""
    device = flag_gems.device
    x = _make_x(8, 1024, torch.bfloat16, device)
    with pytest.raises(ValueError, match="multiple of 256"):
        flag_gems.swiglu_group_quant(_make_x(8, 384, torch.bfloat16, device))
    with pytest.raises(ValueError, match="HiFloat8"):
        flag_gems.swiglu_group_quant(x, quant_mode=2)
    with pytest.raises(ValueError, match="requires round_scale"):
        flag_gems.swiglu_group_quant(x, quant_mode=1, round_scale=False)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        flag_gems.swiglu_group_quant(x.float())
    with pytest.raises(ValueError, match="dst_type"):
        flag_gems.swiglu_group_quant(x, dst_type=torch.int8, quant_mode=0)
    with pytest.raises(ValueError, match="weight must be float32"):
        flag_gems.swiglu_group_quant(x, weight=torch.ones(8, device=device).half())
    with pytest.raises(ValueError, match="group_index must be int64"):
        flag_gems.swiglu_group_quant(
            x, group_index=torch.ones(2, device=device, dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="clamp_limit"):
        flag_gems.swiglu_group_quant(x, clamp_limit=0.0)
    with pytest.raises(ValueError, match="block_size"):
        flag_gems.swiglu_group_quant(x, block_size=128)


VENDOR_CASES = [
    (256, 4096, torch.bfloat16, torch.float8_e4m3fn, 1, True),
    (64, 2560, torch.float16, torch.float8_e5m2, 1, True),
    (64, 4096, torch.bfloat16, torch.float8_e4m3fn, 0, False),
]
if QUICK_MODE:
    VENDOR_CASES = VENDOR_CASES[:1]


@pytest.mark.swiglu_group_quant
@requires_swiglu_group_quant
@pytest.mark.parametrize("T,D,dtype,dst_type,quant_mode,round_scale", VENDOR_CASES)
def test_swiglu_group_quant_matches_vendor(
    T, D, dtype, dst_type, quant_mode, round_scale
):
    """Byte-exact against ``aclnnSwigluGroupQuant`` where CANN provides it.

    This is the comparison without a tolerance: the torch reference carries a
    few ULP of its own from ``exp``, the vendor kernel does not.
    """
    vendor = _vendor_op()
    if vendor is None:
        pytest.skip("cann_ops_nn is not importable; no vendor to compare against")
    x = _make_x(T, D, dtype, flag_gems.device)
    weight = torch.rand(T, device=flag_gems.device, dtype=torch.float32)

    y, s, yo = flag_gems.swiglu_group_quant(
        x,
        weight,
        None,
        dst_type,
        quant_mode,
        0,
        round_scale,
        -1.0,
        True,
    )
    vy, vs, vyo = vendor(
        x,
        weight,
        None,
        None,
        dst_type=DST_CODE[dst_type],
        quant_mode=quant_mode,
        block_size=0,
        round_scale=round_scale,
        clamp_limit=-1.0,
        dst_type_max=15.0,
        output_origin=True,
    )
    _assert_bits_equal(y, vy, T, "y vs vendor")
    _assert_bits_equal(s, vs, T, "scale vs vendor")
    _assert_bits_equal(yo, vyo, T, "y_origin vs vendor")
