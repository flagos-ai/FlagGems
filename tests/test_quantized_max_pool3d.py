# Copyright 2026 FlagOS Contributors.
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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Per-tensor quantized max-pool operates on the integer representation of a
# quantized tensor. The result is exact (no floating-point arithmetic), so the
# tests compare integer representations with exact equality.
#
# PyTorch ships no native QuantizedCUDA kernel for ``quantized_max_pool3d``, so
# the reference is always evaluated on the CPU (QuantizedCPU backend), while the
# FlagGems kernel runs on the GPU.
QDTYPE = torch.quint8
SCALE = 0.05
ZERO_POINT = 100

# The three quantized dtypes ``aten::quantized_max_pool3d`` dispatches over
# (its kernel body is wrapped in ``AT_DISPATCH_QINT_TYPES``), with a zero_point
# that is representable in each.
QINT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]
QINT_ZERO_POINTS = {torch.quint8: 100, torch.qint8: -8, torch.qint32: 0}


def _make_qinput(shape, device, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE):
    x = torch.randn(shape, device=device)
    return torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=dtype)


def _make_qout(shape, device, scale, zero_point, dtype):
    return torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=device),
        scale,
        zero_point,
        dtype,
    )


def _aten_out(qx_cpu, out_cpu, kernel_size, stride, padding, dilation, ceil_mode):
    """Call the native ``quantized_max_pool3d.out`` overload.

    The overload only accepts ``int[3]`` lists positionally, so the parameters
    are expanded here.
    """
    return torch.ops.aten.quantized_max_pool3d.out(
        qx_cpu,
        _as3(kernel_size),
        [] if stride is None else _as3(stride),
        _as3(padding),
        _as3(dilation),
        ceil_mode,
        out=out_cpu,
    )


def _as3(v):
    if isinstance(v, int):
        return [v, v, v]
    return list(v)


# (shape, kernel_size, stride, padding, dilation, ceil_mode)
QUANTIZED_MAXPOOL3D_CONFIGS = [
    # Classic cubic kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    # Large batch, stride 1
    ((4, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


def _assert_qpool_equal(res_out, ref_out, dtype=QDTYPE):
    """Compare two quantized max-pool results for exact equality."""
    assert (
        res_out.shape == ref_out.shape
    ), f"shape mismatch: res={tuple(res_out.shape)} ref={tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype
    # scale / zero_point must be preserved from the input
    assert (
        abs(res_out.q_scale() - ref_out.q_scale()) < 1e-9
    ), f"scale mismatch: res={res_out.q_scale()} ref={ref_out.q_scale()}"
    assert (
        res_out.q_zero_point() == ref_out.q_zero_point()
    ), f"zero_point mismatch: res={res_out.q_zero_point()} ref={ref_out.q_zero_point()}"
    # Max-pool over the quantized integers is exact, so compare integer reprs.
    # Take ``int_repr()`` before moving to the host: ``Tensor.to("cpu")`` is a
    # silent no-op on a non-contiguous quantized CUDA tensor.
    res_int = res_out.int_repr().to("cpu")
    ref_int = ref_out.int_repr()
    utils.gems_assert_equal(res_int, ref_int)


def _pool_args(kernel_size, stride, padding, dilation, ceil_mode):
    return dict(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANTIZED_MAXPOOL3D_CONFIGS,
)
def test_quantized_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode):
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize("scale_zp", [(0.1, 128), (0.02, 50), (0.5, 30)])
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((1, 3, 16, 16, 16), 3, 2, 1, 1, False),
        ((2, 8, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
        ((1, 4, 15, 15, 15), 3, 2, 1, 1, True),
    ],
)
def test_quantized_max_pool3d_quant_params(
    shape, kernel_size, stride, padding, dilation, ceil_mode, scale_zp
):
    """Verify the output preserves arbitrary quantization parameters."""
    scale, zero_point = scale_zp
    qx = _make_qinput(
        shape, device=flag_gems.device, scale=scale, zero_point=zero_point
    )
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize("dtype", QINT_DTYPES)
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((2, 4, 8, 10, 10), 3, 2, 1, 1, False),
        ((1, 3, 9, 9, 9), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
        ((1, 2, 7, 7, 7), 3, 2, 1, 1, True),
        ((1, 2, 9, 9, 9), 2, 1, 0, 2, False),
    ],
)
def test_quantized_max_pool3d_qint_dtypes(
    dtype, shape, kernel_size, stride, padding, dilation, ceil_mode
):
    """Cover every dtype ``aten::quantized_max_pool3d`` dispatches over.

    ``qint32`` in particular has a 4-byte integer representation, so a kernel
    that assumed a 1-byte ``uint8`` storage would silently corrupt it.
    """
    zero_point = QINT_ZERO_POINTS[dtype]
    qx = _make_qinput(
        shape, device=flag_gems.device, zero_point=zero_point, dtype=dtype
    )
    ref_inp = qx.to("cpu")
    args = _pool_args(kernel_size, stride, padding, dilation, ceil_mode)

    ref_out = torch.quantized_max_pool3d(ref_inp, **args)
    res_out = flag_gems.quantized_max_pool3d(qx, **args)

    _assert_qpool_equal(res_out, ref_out, dtype=dtype)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize("dtype", QINT_DTYPES)
def test_quantized_max_pool3d_out_qint_dtypes(dtype):
    """The ``out=`` path must round-trip every supported dtype, not just quint8."""
    shape = (2, 3, 8, 8, 8)
    zero_point = QINT_ZERO_POINTS[dtype]
    qx = _make_qinput(
        shape, device=flag_gems.device, zero_point=zero_point, dtype=dtype
    )
    ref_inp = qx.to("cpu")
    args = _pool_args(2, 2, 0, 1, False)

    ref_out = torch.quantized_max_pool3d(ref_inp, **args)
    out_q = _make_qout(ref_out.shape, flag_gems.device, SCALE, zero_point, dtype)

    res_out = flag_gems.quantized_max_pool3d_out(qx, out=out_q, **args)

    assert res_out is out_q
    _assert_qpool_equal(res_out, ref_out, dtype=dtype)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize("stride", [None, [], ()])
@pytest.mark.parametrize("kernel_size", [2, 3, (2, 3, 4)])
def test_quantized_max_pool3d_default_stride(kernel_size, stride):
    """An omitted or empty ``stride`` must fall back to ``kernel_size``.

    ``aten::quantized_max_pool3d`` declares ``int[3] stride=[]`` and replaces an
    empty list with ``kernel_size``, so all three spellings agree.
    """
    shape = (2, 3, 9, 10, 11)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    if stride is None:
        ref_out = torch.ops.aten.quantized_max_pool3d(ref_inp, _as3(kernel_size))
        res_out = flag_gems.quantized_max_pool3d(qx, kernel_size)
    else:
        ref_out = torch.ops.aten.quantized_max_pool3d(
            ref_inp, _as3(kernel_size), list(stride)
        )
        res_out = flag_gems.quantized_max_pool3d(qx, kernel_size, stride)

    _assert_qpool_equal(res_out, ref_out)

    # ... and the fallback really is kernel_size, not 1.
    explicit = flag_gems.quantized_max_pool3d(qx, kernel_size, _as3(kernel_size))
    _assert_qpool_equal(explicit, ref_out)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize("stride", [None, []])
def test_quantized_max_pool3d_out_default_stride(stride):
    """The ``out=`` variant shares the empty-stride default."""
    shape = (1, 2, 8, 8, 8)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    if stride is None:
        ref_out = torch.ops.aten.quantized_max_pool3d(ref_inp, [2, 2, 2])
        out_q = _make_qout(ref_out.shape, flag_gems.device, SCALE, ZERO_POINT, QDTYPE)
        res_out = flag_gems.quantized_max_pool3d_out(qx, 2, out=out_q)
    else:
        ref_out = torch.ops.aten.quantized_max_pool3d(ref_inp, [2, 2, 2], [])
        out_q = _make_qout(ref_out.shape, flag_gems.device, SCALE, ZERO_POINT, QDTYPE)
        res_out = flag_gems.quantized_max_pool3d_out(qx, 2, stride, out=out_q)

    assert res_out is out_q
    _assert_qpool_equal(res_out, ref_out)


# (name, kwargs) pairs that ``aten::quantized_max_pool3d`` rejects. Each must be
# rejected by the FlagGems kernel with the same error type and message, so that a
# caller cannot get a silently wrong result out of an invalid configuration.
INVALID_PARAM_CASES = [
    ("kernel_size len 1", dict(kernel_size=[2])),
    ("kernel_size len 2", dict(kernel_size=[2, 2])),
    ("kernel_size len 4", dict(kernel_size=[2, 2, 2, 2])),
    ("stride len 1", dict(kernel_size=[2, 2, 2], stride=[1])),
    ("stride len 2", dict(kernel_size=[2, 2, 2], stride=[1, 1])),
    ("padding len 2", dict(kernel_size=[2, 2, 2], padding=[0, 0])),
    ("dilation len 2", dict(kernel_size=[2, 2, 2], dilation=[1, 1])),
    ("kernel_size 0", dict(kernel_size=0)),
    ("kernel_size 0 partial", dict(kernel_size=[2, 0, 2])),
    ("kernel_size negative", dict(kernel_size=-1)),
    ("stride 0", dict(kernel_size=2, stride=0)),
    ("stride negative", dict(kernel_size=2, stride=-1)),
    ("dilation 0", dict(kernel_size=2, dilation=0)),
    ("dilation 0 partial", dict(kernel_size=2, dilation=[1, 0, 1])),
    ("dilation negative", dict(kernel_size=2, dilation=-1)),
    ("padding negative", dict(kernel_size=2, padding=-1)),
    ("padding negative partial", dict(kernel_size=2, padding=[0, -1, 0])),
    ("padding over half kernel", dict(kernel_size=2, padding=2)),
    ("padding over half kernel partial", dict(kernel_size=2, padding=[0, 5, 0])),
    ("padding 2 kernel 3", dict(kernel_size=3, padding=2)),
    ("output size zero", dict(kernel_size=8)),
    ("output size negative", dict(kernel_size=20)),
    ("output size negative dilated", dict(kernel_size=4, stride=1, dilation=3)),
]


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize(
    "name, kwargs", INVALID_PARAM_CASES, ids=[c[0] for c in INVALID_PARAM_CASES]
)
def test_quantized_max_pool3d_invalid_params(name, kwargs):
    """Invalid pooling parameters raise, with ATen's error message."""
    shape = (1, 2, 6, 6, 6)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    with pytest.raises(RuntimeError) as ref_exc:
        torch.quantized_max_pool3d(ref_inp, **kwargs)
    with pytest.raises(RuntimeError) as res_exc:
        flag_gems.quantized_max_pool3d(qx, **kwargs)

    assert str(res_exc.value) == str(ref_exc.value), (
        f"error text mismatch for {name}:\n"
        f"  aten: {ref_exc.value}\n  gems: {res_exc.value}"
    )


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize(
    "name, kwargs",
    INVALID_PARAM_CASES[:8],
    ids=[c[0] for c in INVALID_PARAM_CASES[:8]],
)
def test_quantized_max_pool3d_out_invalid_params(name, kwargs):
    """The ``out=`` variant validates before it touches ``out``."""
    qx = _make_qinput((1, 2, 6, 6, 6), device=flag_gems.device)
    out_q = _make_qout((1, 1, 1, 1, 1), flag_gems.device, SCALE, ZERO_POINT, QDTYPE)

    with pytest.raises(RuntimeError):
        flag_gems.quantized_max_pool3d_out(qx, out=out_q, **kwargs)
    # out must be left untouched when the arguments are rejected
    assert tuple(out_q.shape) == (1, 1, 1, 1, 1)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize(
    "shape",
    [(2, 6, 6, 6), (6, 6, 6), (1, 1, 2, 6, 6, 6)],
    ids=["rank4", "rank3", "rank6"],
)
def test_quantized_max_pool3d_invalid_rank(shape):
    """Only rank-5 (N, C, D, H, W) inputs are accepted."""
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    with pytest.raises(RuntimeError) as ref_exc:
        torch.quantized_max_pool3d(ref_inp, kernel_size=2)
    with pytest.raises(RuntimeError) as res_exc:
        flag_gems.quantized_max_pool3d(qx, kernel_size=2)

    assert str(res_exc.value) == str(ref_exc.value)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize(
    "shape",
    [(1, 0, 6, 6, 6), (1, 2, 0, 6, 6), (1, 2, 6, 0, 6), (1, 2, 6, 6, 0)],
    ids=["C0", "D0", "H0", "W0"],
)
def test_quantized_max_pool3d_zero_sized_dims(shape):
    """A zero-sized C/D/H/W dimension is rejected, matching ATen."""
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    with pytest.raises(RuntimeError) as ref_exc:
        torch.quantized_max_pool3d(ref_inp, kernel_size=2)
    with pytest.raises(RuntimeError) as res_exc:
        flag_gems.quantized_max_pool3d(qx, kernel_size=2)

    assert str(res_exc.value) == str(ref_exc.value)


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d_empty_batch():
    """N == 0 is legal and yields an empty result (ATen allows it)."""
    shape = (0, 2, 6, 6, 6)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_out = torch.quantized_max_pool3d(qx.to("cpu"), kernel_size=2)

    res_out = flag_gems.quantized_max_pool3d(qx, kernel_size=2)

    assert res_out.numel() == 0
    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d_rejects_non_quantized():
    """A plain float tensor is not a valid input."""
    with pytest.raises(RuntimeError):
        flag_gems.quantized_max_pool3d(
            torch.randn(1, 2, 6, 6, 6, device=flag_gems.device), kernel_size=2
        )


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d_rejects_per_channel():
    """Only per-tensor affine quantization is supported, as in ATen."""
    qx = torch.quantize_per_channel(
        torch.randn(1, 2, 6, 6, 6, device=flag_gems.device),
        torch.tensor([SCALE, 0.1], device=flag_gems.device),
        torch.tensor([ZERO_POINT, 90], device=flag_gems.device),
        1,
        QDTYPE,
    )
    with pytest.raises(RuntimeError):
        flag_gems.quantized_max_pool3d(qx, kernel_size=2)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANTIZED_MAXPOOL3D_CONFIGS,
)
def test_quantized_max_pool3d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode
):
    """``.out`` variant writes into a pre-allocated quantized tensor."""
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    # Pre-allocate an out tensor with the expected (ref) shape and matching
    # quantization parameters.
    out_shape = ref_out.shape
    out_q = _make_qout(
        out_shape, flag_gems.device, ref_out.q_scale(), ref_out.q_zero_point(), QDTYPE
    )

    res_out = flag_gems.quantized_max_pool3d_out(
        qx,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        out=out_q,
    )

    # The out variant must alias and return the provided tensor.
    assert res_out is out_q
    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize(
    "out_shape", [(1, 1, 1, 1, 1), (4, 4, 4, 4, 4), (1, 2, 3, 3, 4)]
)
def test_quantized_max_pool3d_out_resizes(out_shape):
    """A mis-sized ``out`` is resized to the pooled shape, as ATen does."""
    shape = (1, 2, 6, 6, 6)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")
    args = _pool_args(2, 2, 0, 1, False)

    ref_out = torch.quantized_max_pool3d(ref_inp, **args)
    ref_o = _make_qout(out_shape, "cpu", 0.9, 7, QDTYPE)
    _aten_out(ref_inp, ref_o, 2, 2, 0, 1, False)

    out_q = _make_qout(out_shape, flag_gems.device, 0.9, 7, QDTYPE)
    res_out = flag_gems.quantized_max_pool3d_out(qx, out=out_q, **args)

    assert res_out is out_q
    assert tuple(out_q.shape) == tuple(ref_o.shape) == tuple(ref_out.shape)
    _assert_qpool_equal(out_q, ref_out)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize("out_qparams", [(0.9, 7), (1.0, 0), (0.02, 200)])
def test_quantized_max_pool3d_out_adopts_input_qparams(out_qparams):
    """``out`` ends up carrying the *input*'s scale/zero_point.

    ATen allocates the result with the input's quantization parameters and the
    generated ``out`` wrapper copies them onto ``out``, so a pre-existing
    scale/zero_point on ``out`` is overwritten rather than respected.
    """
    shape = (1, 2, 6, 6, 6)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")
    args = _pool_args(2, 2, 0, 1, False)

    ref_out = torch.quantized_max_pool3d(ref_inp, **args)
    ref_o = _make_qout(ref_out.shape, "cpu", *out_qparams, QDTYPE)
    _aten_out(ref_inp, ref_o, 2, 2, 0, 1, False)

    out_q = _make_qout(ref_out.shape, flag_gems.device, *out_qparams, QDTYPE)
    res_out = flag_gems.quantized_max_pool3d_out(qx, out=out_q, **args)

    assert abs(res_out.q_scale() - ref_o.q_scale()) < 1e-9
    assert res_out.q_zero_point() == ref_o.q_zero_point()
    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d_out
def test_quantized_max_pool3d_out_dtype_mismatch():
    """An ``out`` whose dtype differs from the input is rejected."""
    qx = _make_qinput((1, 2, 6, 6, 6), device=flag_gems.device)
    ref_inp = qx.to("cpu")
    ref_out = torch.quantized_max_pool3d(ref_inp, kernel_size=2, stride=2)

    for dtype, zero_point in [(torch.qint8, 0), (torch.qint32, 0)]:
        ref_o = _make_qout(ref_out.shape, "cpu", SCALE, zero_point, dtype)
        with pytest.raises(RuntimeError) as ref_exc:
            _aten_out(ref_inp, ref_o, 2, 2, 0, 1, False)

        out_q = _make_qout(ref_out.shape, flag_gems.device, SCALE, zero_point, dtype)
        with pytest.raises(RuntimeError) as res_exc:
            flag_gems.quantized_max_pool3d_out(qx, 2, 2, out=out_q)

        assert str(res_exc.value) == str(ref_exc.value)

    # A non-quantized out is rejected too.
    out_float = torch.zeros(ref_out.shape, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.quantized_max_pool3d_out(qx, 2, 2, out=out_float)


@pytest.mark.quantized_max_pool3d_out
def test_quantized_max_pool3d_out_preserves_storage_offset():
    """Writing into a view must not clobber the rest of its storage.

    ATen's ``resize_output`` keeps ``out``'s storage offset, so pooling into
    ``big[1]`` leaves ``big[0]`` untouched.
    """
    shape = (1, 2, 6, 6, 6)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")
    args = _pool_args(2, 2, 0, 1, False)
    ref_out = torch.quantized_max_pool3d(ref_inp, **args)

    # (2, *pooled_shape[1:]) so that big[1] already has the right shape.
    big_shape = (2,) + tuple(ref_out.shape)[1:]
    big = _make_qout(big_shape, flag_gems.device, SCALE, ZERO_POINT, QDTYPE)
    sub = big[1]
    assert sub.storage_offset() != 0

    res_out = flag_gems.quantized_max_pool3d_out(qx, out=sub, **args)

    assert res_out is sub
    utils.gems_assert_equal(sub.int_repr().to("cpu"), ref_out.int_repr())
    # The sibling slice keeps its original (zero-point) contents.
    assert bool((big[0].int_repr() == ZERO_POINT).all())


@pytest.mark.quantized_max_pool3d_out
def test_quantized_max_pool3d_out_non_contiguous():
    """A non-contiguous ``out`` of the right shape keeps its strides.

    ATen only restrides ``out`` when a resize actually happens, so a strided
    ``out`` that already has the pooled shape is written through its own strides.
    """
    shape = (1, 2, 6, 8, 10)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")
    args = _pool_args(2, 2, 0, 1, False)
    ref_out = torch.quantized_max_pool3d(ref_inp, **args)

    # Every other element along W gives a non-contiguous view of the right shape.
    holder_shape = tuple(ref_out.shape)[:-1] + (ref_out.shape[-1] * 2,)
    holder = _make_qout(holder_shape, flag_gems.device, 0.9, 7, QDTYPE)
    view = holder[..., ::2]
    assert not view.is_contiguous()
    strides_before = view.stride()

    res_out = flag_gems.quantized_max_pool3d_out(qx, out=view, **args)

    assert res_out is view
    assert view.stride() == strides_before, "out was silently restrided"
    _assert_qpool_equal(view, ref_out)
    # The interleaved elements are untouched.
    assert bool((holder[..., 1::2].int_repr() == 7).all())


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize("layout", ["channels_last_3d", "permuted", "strided_slice"])
def test_quantized_max_pool3d_non_contiguous_input(layout):
    """Non-contiguous inputs pool correctly and keep ATen's output layout.

    ATen has a dedicated channels-last path that preserves the memory format of
    a ``channels_last_3d`` input; every other layout produces a contiguous
    NCDHW result.
    """
    shape = (2, 4, 6, 8, 10)
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    def as_layout(q):
        # Restride the quantized tensor itself: rebuilding one over a strided
        # ``int_repr`` would silently normalise the strides on CUDA.
        if layout == "channels_last_3d":
            return q.contiguous(memory_format=torch.channels_last_3d)
        if layout == "permuted":
            return q.permute(0, 1, 2, 4, 3)
        return q[:, :, :, :, ::2]

    view_res, view_ref = as_layout(qx), as_layout(ref_inp)
    args = _pool_args(2, 2, 0, 1, False)

    ref_out = torch.quantized_max_pool3d(view_ref, **args)
    res_out = flag_gems.quantized_max_pool3d(view_res, **args)

    _assert_qpool_equal(res_out, ref_out)
    assert res_out.is_contiguous(
        memory_format=torch.channels_last_3d
    ) == ref_out.is_contiguous(memory_format=torch.channels_last_3d)
