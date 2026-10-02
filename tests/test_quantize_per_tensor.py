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

import importlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import flag_gems
from flag_gems.ops.quantize_per_tensor import _quantize_per_tensor_impl

from . import accuracy_utils as utils

# ``quantize_per_tensor`` only accepts float32 tensors on the quantized CUDA
# backend (Half/BFloat16/Double raise "Quantize only works on Float Tensor"),
# so there is no ``FLOAT_DTYPES`` parametrization here. The output is a
# quantized tensor whose ``int_repr`` matches the reference exactly (round to
# nearest, ties to even), hence ``gems_assert_equal`` on the int representation.
#
# The computation is fp32 end to end -- matching ATen, which narrows the double
# scale to float and multiplies by the fp32 reciprocal -- so the result does not
# depend on whether the device supports fp64.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]
QUANT_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SCALES = [0.1, 0.01, 1.0]
# ``zero_point`` must lie within the representable integer range of *every*
# tested quantized dtype. quint8 covers [0, 255], qint8 covers [-128, 127] and
# qint32 covers the full int32 range, so the common range is [0, 127]. PyTorch
# validates this bound on ``quantize_per_tensor`` and rejects out-of-range values.
ZERO_POINTS = [0, 10, 64]


def _make_input(shape, device="cuda"):
    # Spread values across a wide range so that clamping to the integer range
    # is exercised alongside ordinary in-range values.
    return torch.randn(shape, dtype=torch.float32, device=device) * 3.0


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor(shape, in_dtype, scale, zero_point):
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor(ref_inp, scale, zero_point, in_dtype)
    res_out = flag_gems.quantize_per_tensor(res_inp, scale, zero_point, in_dtype)

    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()


@pytest.mark.quantize_per_tensor_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor_out(shape, in_dtype, scale, zero_point):
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    # Pre-allocate a quantized `out` buffer with *different* scale/zero_point so
    # that we verify the kernel writes the passed parameters back onto it.
    res_out = torch.quantize_per_tensor(res_inp, 0.5, 100, in_dtype)
    ref_out = torch.quantize_per_tensor(ref_inp, 0.5, 100, in_dtype)

    ref_r = torch.ops.aten.quantize_per_tensor.out(
        ref_inp, scale, zero_point, in_dtype, out=ref_out
    )
    res_r = flag_gems.quantize_per_tensor_out(
        res_inp, scale, zero_point, in_dtype, out=res_out
    )

    assert res_r is res_out
    utils.gems_assert_equal(res_r.int_repr(), ref_r.int_repr())
    assert res_r.q_scale() == ref_r.q_scale()
    assert res_r.q_zero_point() == ref_r.q_zero_point()


@pytest.mark.quantize_per_tensor
@pytest.mark.skipif(
    utils.TO_CPU,
    reason="half-way quotients are backend-specific: CPU aten rounds the fp32 "
    "product, CUDA aten rounds the fp64 quotient, and the two disagree by 1. "
    "This kernel targets CUDA, so the comparison is only meaningful there.",
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantize_per_tensor_half_way_values(in_dtype):
    """Values whose quotient lands exactly on ``k + 0.5``.

    Half-way quotients are the only input class that pins down the precision of
    the division and the position of the ``zero_point`` add; uniformly random
    inputs almost never produce one. A non-zero ``zero_point`` is essential here,
    since adding it before rounding rather than after only changes the result at
    a tie.

    Skipped under ``--ref=cpu``: at exactly these inputs the two aten backends
    genuinely disagree (for x=0.85, scale=0.1 CPU gives 8 and CUDA gives 9), so no
    single kernel can be bit-exact against both. Every other test in this file
    passes under either reference, since random inputs essentially never tie.
    """
    scale = 0.14897697696685788
    ks = torch.arange(-400, 400, dtype=torch.float64) + 0.5
    res_inp = (ks * scale).to(torch.float32).cuda()

    ref_out = torch.quantize_per_tensor(res_inp, scale, 5, in_dtype)
    res_out = flag_gems.quantize_per_tensor(res_inp, scale, 5, in_dtype)
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "slice", "narrow"])
def test_quantize_per_tensor_non_contiguous(in_dtype, layout):
    base = _make_input((64, 64))
    view = {
        "transpose": lambda t: t.t(),
        "slice": lambda t: t[:, ::2],
        "narrow": lambda t: t[8:24, 4:20],
    }[layout](base)
    assert not view.is_contiguous()
    ref_view = utils.to_reference(view)

    ref_out = torch.quantize_per_tensor(ref_view, 0.05, 7, in_dtype)
    res_out = flag_gems.quantize_per_tensor(view, 0.05, 7, in_dtype)
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    assert tuple(res_out.shape) == tuple(view.shape)


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_quantize_per_tensor_rejects_non_float32(bad_dtype):
    # ATen raises "Quantize only works on Float Tensor" rather than upcasting.
    inp = torch.randn(32, dtype=bad_dtype, device="cuda")
    with pytest.raises(RuntimeError, match="torch.float32"):
        flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.quint8)


@pytest.mark.quantize_per_tensor
def test_quantize_per_tensor_empty():
    inp = torch.empty(0, dtype=torch.float32, device="cuda")
    ref_out = torch.quantize_per_tensor(utils.to_reference(inp), 0.1, 0, torch.quint8)
    res_out = flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.quint8)
    assert res_out.numel() == 0
    assert res_out.dtype == ref_out.dtype
    assert res_out.q_scale() == ref_out.q_scale()


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_dtype_mismatch():
    inp = _make_input((8, 8))
    # Requesting quint8 while handing in a qint8 buffer: ATen validates the out
    # dtype against the *requested* quantized dtype and rejects the mismatch.
    out = torch._empty_affine_quantized(
        (8, 8), scale=0.1, zero_point=0, dtype=torch.qint8, device="cuda"
    )
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 0, torch.quint8, out=out)


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_shape_mismatch():
    inp = _make_input((8, 8))
    out = torch._empty_affine_quantized(
        (2, 2), scale=0.1, zero_point=0, dtype=torch.quint8, device="cuda"
    )
    # A shape mismatch is a resize in ATen, but ``aten::resize_`` has no
    # QuantizedCUDA kernel, so the native op fails here too.
    with pytest.raises(RuntimeError):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 0, torch.quint8, out=out)


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize(
    "memory_format,shape",
    [
        (torch.channels_last, (2, 3, 4, 5)),
        (torch.channels_last_3d, (2, 3, 4, 5, 6)),
    ],
    ids=["channels_last", "channels_last_3d"],
)
def test_quantize_per_tensor_memory_format(memory_format, shape):
    # ATen materializes the input with `rtensor.suggest_memory_format()` and
    # allocates the quantized output in that same format, so a channels-last
    # input yields a channels-last int_repr. Check both the values and the
    # resulting strides against the native op.
    utils.init_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).contiguous(
        memory_format=memory_format
    )

    ref_int_repr = torch.quantize_per_tensor(
        utils.to_reference(inp), 0.1, 0, torch.qint8
    ).int_repr()
    res_int_repr = flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.qint8).int_repr()

    # The kernel allocates and fills the output in the suggested memory format,
    # which is what this asserts (see `_quantize_per_tensor_impl`).
    impl_int_repr = _quantize_per_tensor_impl(inp, 0.1, 0, torch.qint8)
    assert impl_int_repr.stride() == ref_int_repr.stride(), (
        f"int_repr strides differ: got {tuple(impl_int_repr.stride())}, "
        f"expected {tuple(ref_int_repr.stride())}"
    )
    utils.gems_assert_equal(impl_int_repr, utils.to_reference(ref_int_repr))

    # Values must match through the public entry point regardless of layout.
    utils.gems_assert_equal(res_int_repr, utils.to_reference(ref_int_repr))


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize(
    "memory_format,shape",
    [
        (torch.contiguous_format, (2, 3, 4, 5)),
        (torch.channels_last, (2, 3, 4, 5)),
        (torch.channels_last_3d, (2, 3, 4, 5, 6)),
    ],
    ids=["contig", "cl", "cl3d"],
)
def test_quantize_per_tensor_public_entry_keeps_layout(memory_format, shape):
    """The public result must keep the suggested memory format end to end.

    The tensor-building step must not re-lay the storage out to contiguous:
    a channels-last input yields a channels-last ``int_repr`` on native, and
    the public entry point is what the memory-format contract has to hold on.
    """
    utils.init_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).contiguous(
        memory_format=memory_format
    )

    ref = torch.quantize_per_tensor(utils.to_reference(inp), 0.1, 0, torch.qint8)
    res = flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.qint8)

    assert res.int_repr().stride() == ref.int_repr().stride(), (
        f"public int_repr strides differ: got {tuple(res.int_repr().stride())}, "
        f"expected {tuple(ref.int_repr().stride())}"
    )
    utils.gems_assert_equal(res.int_repr(), utils.to_reference(ref.int_repr()))
    assert res.q_scale() == ref.q_scale()
    assert res.q_zero_point() == ref.q_zero_point()
    # The wrapped tensor is a functioning quantized tensor downstream (checked
    # only against the CUDA reference: the CPU and CUDA quantizers round the
    # dequantized float differently, the same disagreement that makes
    # test_quantize_per_tensor_half_way_values skip under --ref=cpu).
    if not utils.TO_CPU:
        utils.gems_assert_equal(res.dequantize(), utils.to_reference(ref.dequantize()))


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize(
    "dtype,zp,word",
    [
        (torch.quint8, 300, "above"),
        (torch.quint8, -1, "below"),
        (torch.qint8, 128, "above"),
        (torch.qint8, -129, "below"),
    ],
)
def test_quantize_per_tensor_zero_point_range(dtype, zp, word):
    """An out-of-range zero_point raises ATen's message instead of clamping.

    Native rejects the value synchronously with
    'quantize_tensor_per_tensor_affine zero_point <zp> is above/below upper/
    lower bound.'; it does not silently clamp it into the valid bin.
    """
    inp = _cuda_input((2, 3, 4, 4))

    ref_err = None
    try:
        torch.quantize_per_tensor(utils.to_reference(inp), 0.1, zp, dtype)
    except RuntimeError as e:
        ref_err = str(e)

    with pytest.raises(RuntimeError) as exc_info:
        flag_gems.quantize_per_tensor(inp, 0.1, zp, dtype)
    assert f"zero_point {zp} is {word}" in str(exc_info.value)
    if ref_err is not None:
        assert str(exc_info.value) == ref_err

    # The out= overload performs the same check.
    out = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.1, zero_point=0, dtype=dtype, device=flag_gems.device
    )
    with pytest.raises(RuntimeError, match=f"zero_point {zp} is {word}"):
        flag_gems.quantize_per_tensor_out(inp, 0.1, zp, dtype, out=out)


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (2, 3, 4, 5, 6)], ids=["4d", "5d"])
def test_quantize_per_tensor_contiguous_unchanged(shape):
    # A default-layout input must keep producing default-layout output.
    utils.init_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_int_repr = torch.quantize_per_tensor(
        utils.to_reference(inp), 0.1, 0, torch.qint8
    ).int_repr()
    res_int_repr = flag_gems.quantize_per_tensor(inp, 0.1, 0, torch.qint8).int_repr()
    assert res_int_repr.is_contiguous()
    utils.gems_assert_equal(res_int_repr, utils.to_reference(ref_int_repr))


def _cuda_input(shape, dtype=torch.float32):
    """Deterministic CUDA input, independent of the global RNG.

    A local generator with a fixed seed keeps the tests reproducible regardless
    of execution order (``torch.manual_seed`` would still leave an implicit
    dependence on how many draws happened earlier), and no value derived from
    ``hash()`` is used, since Python randomises that per process.
    """
    gen = torch.Generator(device="cuda").manual_seed(20250918)
    return torch.randn(shape, dtype=dtype, device="cuda", generator=gen) * 3.0


def _qbuf(shape, scale=0.5, zero_point=100, dtype=torch.quint8, **kwargs):
    return torch._empty_affine_quantized(
        shape, scale=scale, zero_point=zero_point, dtype=dtype, device="cuda", **kwargs
    )


def _sentinel_buffer(shape, dtype=torch.quint8, value=100, ref=(0.5, 100)):
    """A quantized buffer filled with a known integer, for neighbour checks."""
    buf = _qbuf(shape, scale=ref[0], zero_point=ref[1], dtype=dtype)
    storage_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]
    buf.copy_(
        torch._make_per_tensor_quantized_tensor(
            torch.full(shape, value, dtype=storage_dtype, device="cuda"),
            scale=ref[0],
            zero_point=ref[1],
        )
    )
    return buf


def _native_out(inp, scale, zero_point, dtype, out):
    """Reference call, returning None when this build's native op rejects it."""
    try:
        return torch.ops.aten.quantize_per_tensor.out(
            utils.to_reference(inp), scale, zero_point, dtype, out=out
        )
    except (RuntimeError, NotImplementedError):
        return None


@pytest.mark.quantize_per_tensor_out
@pytest.mark.parametrize(
    "layout",
    ["contiguous", "strided", "transposed", "offset", "narrowed", "channels_last"],
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantize_per_tensor_out_matches_native_layout(in_dtype, layout):
    """``out`` is written through its own strides, matching the native op.

    The kernel stores into ``out``'s storage directly, so a non-contiguous
    ``out`` must receive exactly the values ``out.copy_(result)`` would have
    placed: same integers, same resulting strides, and the surrounding elements
    of the buffer left alone. The native op is the reference for all three.
    """
    inp = _cuda_input((64, 64))
    ref_scale, ref_zp = 0.5, 100

    def make_out():
        if layout == "contiguous":
            return _qbuf((64, 64), ref_scale, ref_zp, in_dtype)
        if layout == "strided":
            return _sentinel_buffer((64, 128), in_dtype, ref=(ref_scale, ref_zp))[
                :, ::2
            ]
        if layout == "transposed":
            return _qbuf((64, 64), ref_scale, ref_zp, in_dtype).t()
        if layout == "offset":
            return _sentinel_buffer((2, 64, 64), in_dtype, ref=(ref_scale, ref_zp))[1]
        if layout == "narrowed":
            return _sentinel_buffer((64, 128), in_dtype, ref=(ref_scale, ref_zp))[
                :, 32:96
            ]
        return _qbuf(
            (2, 3, 4, 5), ref_scale, ref_zp, in_dtype, memory_format=torch.channels_last
        )

    ref_out = make_out()
    res_out = make_out()
    if layout == "channels_last":
        a4 = _cuda_input((2, 3, 4, 5))
        ref_r = _native_out(a4, 0.1, 7, in_dtype, ref_out)
        if ref_r is None:
            pytest.skip("native op does not support this case in this build")
        res_r = flag_gems.quantize_per_tensor_out(a4, 0.1, 7, in_dtype, out=res_out)
    else:
        ref_r = _native_out(inp, 0.1, 7, in_dtype, ref_out)
        if ref_r is None:
            pytest.skip("native op does not support this case in this build")
        res_r = flag_gems.quantize_per_tensor_out(inp, 0.1, 7, in_dtype, out=res_out)

    assert res_r is res_out
    utils.gems_assert_equal(res_r.int_repr(), utils.to_reference(ref_r.int_repr()))
    assert tuple(res_r.stride()) == tuple(ref_r.stride())
    assert res_r.q_scale() == ref_r.q_scale()
    assert res_r.q_zero_point() == ref_r.q_zero_point()


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_preserves_offset_and_strides():
    """A strided or offset ``out`` is filled without touching its neighbours.

    ``out`` is a view into a larger sentinel-filled buffer. The operator must
    write exactly that view: the interleaved/allocation-adjacent elements that
    are not part of ``out`` keep their sentinel, and the *base* tensor's
    quantizer stays as it was (only the view's quantizer may move).
    """
    inp = _cuda_input((64, 64))

    # Strided view: every other column of a wider buffer.
    buf = _sentinel_buffer((64, 128))
    view = buf[:, ::2]
    res_r = flag_gems.quantize_per_tensor_out(inp, 0.1, 7, torch.quint8, out=view)
    assert res_r is view
    assert bool((buf[:, 1::2].int_repr() == 100).all().item())
    assert buf.q_scale() == 0.5 and buf.q_zero_point() == 100
    ref = torch.quantize_per_tensor(utils.to_reference(inp), 0.1, 7, torch.quint8)
    utils.gems_assert_equal(res_r.int_repr(), utils.to_reference(ref.int_repr()))
    assert res_r.q_scale() == 0.1 and res_r.q_zero_point() == 7

    # Offset view: the second slice of a batched buffer.
    big = _sentinel_buffer((2, 64, 64))
    offset_view = big[1]
    assert offset_view.storage_offset() != 0
    res_r2 = flag_gems.quantize_per_tensor_out(
        inp, 0.1, 7, torch.quint8, out=offset_view
    )
    assert res_r2 is offset_view
    assert bool((big[0].int_repr() == 100).all().item())
    utils.gems_assert_equal(res_r2.int_repr(), utils.to_reference(ref.int_repr()))


@pytest.mark.quantize_per_tensor_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantize_per_tensor_out_mismatched_qparams(in_dtype):
    """A per-tensor ``out`` carrying different qparams adopts the requested ones.

    ``out.copy_`` is what moves the quantizer in ATen; the operator must
    reproduce that observable result (same object, requested scale /
    zero_point) without the extra native O(n) pass over the values.
    """
    zero_point = 0 if in_dtype is not torch.quint8 else 10
    inp = _cuda_input((16, 32))
    ref = torch.quantize_per_tensor(utils.to_reference(inp), 0.25, zero_point, in_dtype)

    out = _qbuf((16, 32), 9.0, 1, in_dtype)
    ref_out = _qbuf((16, 32), 9.0, 1, in_dtype)
    ref_r = _native_out(inp, 0.25, zero_point, in_dtype, ref_out)
    if ref_r is None:
        pytest.skip("native op does not support this case in this build")
    res_r = flag_gems.quantize_per_tensor_out(inp, 0.25, zero_point, in_dtype, out=out)

    assert res_r is out
    assert res_r.q_scale() == ref_r.q_scale() == 0.25
    assert res_r.q_zero_point() == ref_r.q_zero_point() == zero_point
    utils.gems_assert_equal(res_r.int_repr(), utils.to_reference(ref.int_repr()))


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_rejects_per_channel_out():
    """A per-channel ``out`` raises ATen's "same qscheme" message.

    The kernel stores integers and has no way to install a per-tensor quantizer
    over a per-channel tensor, so the rejection that native ``copy_`` supplied
    is raised directly.
    """
    inp = _cuda_input((8, 8))
    per_channel = torch.quantize_per_channel(
        torch.randn(8, 8, device="cuda"),
        torch.rand(8, device="cuda") + 0.1,
        torch.zeros(8, dtype=torch.long, device="cuda"),
        0,
        torch.quint8,
    )
    with pytest.raises(RuntimeError, match="same qscheme"):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 7, torch.quint8, out=per_channel)


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_aliasing_input():
    """An ``out`` that shares storage with the float input still matches native.

    ``out`` is a quantized alias of the input's storage, so the kernel cannot
    read and write it at once; the integers are moved device-side afterwards.
    """
    inp = _cuda_input((64,))
    ref_out = _qbuf((64,))
    ref_r = _native_out(inp, 0.1, 7, torch.quint8, ref_out)
    if ref_r is None:
        pytest.skip("native op does not support this case in this build")

    res_out = torch._empty_affine_quantized(
        0, scale=0.5, zero_point=100, dtype=torch.quint8, device="cuda"
    )
    res_out.set_(inp.untyped_storage(), 0, (64,), (1,))
    res_r = flag_gems.quantize_per_tensor_out(inp, 0.1, 7, torch.quint8, out=res_out)

    assert res_r is res_out
    utils.gems_assert_equal(res_r.int_repr(), utils.to_reference(ref_r.int_repr()))
    assert res_r.q_scale() == 0.1 and res_r.q_zero_point() == 7


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_uses_triton_not_native_copy(monkeypatch):
    """Guard: the ``out=`` path launches the Triton kernel and no native copy_.

    The reviewed concern was that ``out.copy_(result)`` redispatches a
    quantized tensor to ATen, leaving a native O(n) pass after the Triton
    kernel. This counts the Triton launches and asserts no ``aten::copy_`` is
    dispatched, so the guard fails if a ``copy_`` is reintroduced.
    """
    gems_impl = importlib.import_module("flag_gems.ops.quantize_per_tensor")

    calls = []

    def _counting(name, original):
        # A separate class per kernel: ``original`` must be bound now, not read
        # from the loop variable later (closures capture by reference).
        class _Counting:
            def __getitem__(self, grid):
                calls.append(name)
                return original[grid]

        return _Counting()

    for name in ("quantize_per_tensor_kernel", "quantize_per_tensor_strided_kernel"):
        monkeypatch.setattr(gems_impl, name, _counting(name, getattr(gems_impl, name)))

    dispatched = []

    class _Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    inp = _cuda_input((32, 32))
    out = _qbuf((32, 32), 9.0, 1)
    with _Recorder():
        flag_gems.quantize_per_tensor_out(inp, 0.1, 7, torch.quint8, out=out)

    assert calls, "quantize_per_tensor_out did not launch a Triton kernel"
    native_copies = [op for op in dispatched if "copy_" in op]
    assert not native_copies, f"native copy_ still dispatched: {native_copies}"


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out_rejects_internally_overlapping_out():
    """A stride-0 (broadcast) ``out`` raises native's overlap message.

    Such an ``out`` has no single value to hold, so writing it would leave
    whichever element was written last; the native op rejects it in the copy it
    would otherwise perform.
    """
    inp = _cuda_input((4, 8))
    out = _qbuf((4, 8))[0:1].expand(4, 8)
    with pytest.raises(RuntimeError, match="more than one element"):
        flag_gems.quantize_per_tensor_out(inp, 0.1, 7, torch.quint8, out=out)
