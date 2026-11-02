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

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device

# The tests call the FlagGems entry points -- ``flag_gems.quantized_max_pool2d``
# and ``flag_gems.quantized_max_pool2d_out`` -- directly, rather than going
# through ``torch.quantized_max_pool2d`` under a registered override. Direct
# calls keep the test file free of ``use_gems()`` (forbidden by the
# ``check-kernelgen-tests`` CI rule) and are what the sibling
# ``quantized_max_pool1d``/``3d`` test files do.
#
# The consequence is that these tests do not exercise operator *dispatch*: they
# validate the kernel and its argument handling, not that
# ``torch.quantized_max_pool2d`` reaches FlagGems. The
# ``QUANTIZED_CUDA_DISPATCH_KEY`` registration in ``flag_gems/__init__.py`` is
# what makes dispatch work for real users, and it is not covered here.
# ``test_quantized_max_pool2d_runs_triton_kernel`` at least pins down that the
# entry points run the Triton kernel rather than delegating to PyTorch.

# ``aten::quantized_max_pool2d`` dispatches over the three per-tensor quantized
# integer dtypes (its body is wrapped in ``AT_DISPATCH_QINT_TYPES_AND(Byte,...)``).
# PyTorch's own accelerator kernel is cuDNN-backed and only handles qint8, so the
# reference always runs on CPU while the tested tensors live on the accelerator.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# The integer range of each quantized dtype, used to build inputs that cover it.
_QINT_RANGE = {
    torch.quint8: (0, 255),
    torch.qint8: (-128, 127),
    torch.qint32: (-(2**31), 2**31 - 1),
}


# (shape, kernel_size, stride, padding, dilation, ceil_mode)
QUANT_MAX_POOL2D_CONFIGS = [
    # Classic 2x2 pooling with default (empty) stride -> falls back to kernel_size
    ((2, 4, 16, 16), (2, 2), [], 0, 1, False),
    # 3x3 kernel, stride 2, padding 1 (ResNet style)
    ((4, 8, 32, 32), 3, 2, 1, 1, False),
    # Non-square kernel/stride/padding
    ((2, 3, 28, 28), (3, 5), (1, 2), (1, 0), 1, False),
    # Dilation
    ((1, 4, 20, 20), 2, 1, 0, 2, False),
    # ceil_mode
    ((2, 4, 15, 15), 3, 2, 1, 1, True),
    # Larger spatial dims
    ((1, 16, 56, 56), 3, 2, 1, 1, False),
    # Asymmetric padding
    ((2, 2, 16, 20), 2, 2, (1, 0), 1, False),
    # 3-D (C, H, W) input: the native operator treats it as a single batch
    ((3, 16, 16), (2, 2), [], 0, 1, False),
    ((5, 15, 17), 3, 2, 1, 1, True),
    # Window entirely inside the padding region (k // 2 == p), so the output
    # picks up the padding sentinel rather than a real value
    ((1, 2, 6, 6), 4, 4, 2, 1, False),
]


def _make_quant_tensor(shape, dtype, scale, zero_point, dev=device):
    """Build a per-tensor quantized tensor whose integers span the dtype range.

    The integer representation is constructed directly rather than by quantizing
    floats, so the test controls the exact ``int_repr`` (including the extremes
    of each dtype) instead of relying on the rounding of a float round-trip.
    """
    low, high = _QINT_RANGE[dtype]
    int_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]
    ints = torch.randint(low, high, shape, dtype=torch.int64).to(int_dtype)
    flat = ints.reshape(-1)
    if flat.numel() >= 2:
        # Pin the dtype extremes into the data so saturation is always covered.
        flat[0] = low
        flat[1] = high
    return torch._make_per_tensor_quantized_tensor(
        ints.to(dev), scale=scale, zero_point=zero_point
    )


def _valid_zero_point(dtype, zero_point):
    """Clamp a zero_point into the range the dtype accepts."""
    low, high = _QINT_RANGE[dtype]
    return max(low, min(high, zero_point))


def _to_cpu_keep_layout(inp):
    """Copy a quantized tensor to CPU without losing its memory format.

    ``Tensor.to("cpu")`` on a quantized tensor normalises the result to
    contiguous strides, which would silently turn a channels-last reference into
    a contiguous one.
    """
    host = inp.cpu()
    if inp.dim() == 4 and inp.is_contiguous(memory_format=torch.channels_last):
        host = host.contiguous(memory_format=torch.channels_last)
    return host


def _reference(inp, *args, **kwargs):
    """Run the native operator on a CPU copy of ``inp``."""
    return torch.quantized_max_pool2d(_to_cpu_keep_layout(inp), *args, **kwargs)


def _assert_same_quantized(res, ref, dtype):
    assert res.dtype == dtype
    assert res.shape == ref.shape
    assert res.q_scale() == ref.q_scale()
    assert res.q_zero_point() == ref.q_zero_point()
    # The integer representation is the authoritative result: max pooling picks
    # an existing quantized value, so it must be bit-exact.
    utils.gems_assert_equal(res.int_repr().cpu(), ref.int_repr())
    # Dequantize both sides on CPU. For qint32, ``dequantize`` evaluates
    # ``q - zero_point`` in int32 and wraps for integers near the dtype's
    # extremes, and CPU and CUDA wrap differently; going through CPU for both
    # compares the pooling result rather than that backend difference.
    utils.gems_assert_close(res.cpu().dequantize(), ref.dequantize(), torch.float32)


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANT_MAX_POOL2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", [0.1, 1.5])
@pytest.mark.parametrize("zero_point", [0, -128])
def test_quantized_max_pool2d(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype, scale, zero_point
):
    zero_point = _valid_zero_point(dtype, zero_point)
    res_inp = _make_quant_tensor(shape, dtype, scale, zero_point)

    ref_out = _reference(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    res_out = flag_gems.quantized_max_pool2d(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    assert res_out.device.type == res_inp.device.type
    _assert_same_quantized(res_out, ref_out, dtype)


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_channels_last(dtype):
    """A channels_last input keeps that layout on the output, as in ATen.

    The accelerator kernel this operator replaces is cuDNN-backed and always
    allocates a channels-last 4-D result; the CPU reference picks channels-last
    only for an already channels-last input. Both agree for this input, so the
    test pins the shared part of the contract.
    """
    res_inp = _make_quant_tensor(
        (2, 5, 16, 20), dtype, 0.05, _valid_zero_point(dtype, 7)
    ).contiguous(memory_format=torch.channels_last)
    assert res_inp.is_contiguous(memory_format=torch.channels_last)

    ref_out = _reference(res_inp, 3, 2, 1)
    res_out = flag_gems.quantized_max_pool2d(res_inp, 3, 2, 1)

    assert ref_out.is_contiguous(memory_format=torch.channels_last)
    assert res_out.is_contiguous(memory_format=torch.channels_last)
    assert res_out.stride() == ref_out.stride()
    _assert_same_quantized(res_out, ref_out, dtype)


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_4d_output_is_channels_last(dtype):
    """A 4-D input yields a channels-last output, whatever its layout.

    This is the layout contract of the accelerator kernel being replaced
    (``quantized_max_pool2d_cudnn``): it permutes every 4-D input to
    channels-last and allocates the result channels-last -- NCHW, strided and
    already channels-last inputs alike. The CPU kernel instead gives an
    NCHW-contiguous result for a non-channels-last input, so the reference is
    used for the pooled integers, not for the output strides.

    The reference input is built on the host from the same integers rather
    than moved with ``.cpu()``: ``Tensor.cpu`` on a quantized tensor with
    strides no memory format describes (the sliced case) keeps the tensor on
    the accelerator with compacted strides, which would dispatch the
    "reference" to CUDA.
    """
    scale = 0.05
    zero_point = _valid_zero_point(dtype, 7)
    low, high = _QINT_RANGE[dtype]
    int_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]

    def make_pair(shape):
        ints = torch.randint(low, high, shape, dtype=torch.int64).to(int_dtype)
        host = torch._make_per_tensor_quantized_tensor(
            ints, scale=scale, zero_point=zero_point
        )
        dev = torch._make_per_tensor_quantized_tensor(
            ints.to(device), scale=scale, zero_point=zero_point
        )
        return host, dev

    cases = {
        # name -> (view of the host tensor, view of the device tensor)
        "nchw": lambda t: t,
        "transposed": lambda t: t.transpose(2, 3),
        "sliced": lambda t: t[..., ::2],
    }
    for name, view in cases.items():
        host, dev = make_pair((2, 4, 16, 32))
        host_v, dev_v = view(host), view(dev)
        if name != "nchw":
            assert not dev_v.is_contiguous()

        res_out = flag_gems.quantized_max_pool2d(dev_v, (2, 2))
        assert res_out.is_contiguous(memory_format=torch.channels_last), name
        # The result's strides are those of a channels-last allocation, which
        # is what the accelerator kernel allocates for every 4-D input.
        exp = torch._empty_affine_quantized(
            tuple(res_out.shape),
            scale=scale,
            zero_point=zero_point,
            dtype=dtype,
            device=device,
            memory_format=torch.channels_last,
        )
        assert res_out.stride() == exp.stride(), name

        ref_out = torch.quantized_max_pool2d(host_v, (2, 2))
        _assert_same_quantized(res_out, ref_out, dtype)


@pytest.mark.quantized_max_pool2d
@pytest.mark.skipif(
    torch.device(flag_gems.device).type != "cuda",
    reason="compares the operator against native quantized_max_pool2d_cudnn",
)
def test_quantized_max_pool2d_matches_native_cuda():
    """qint8 is compared directly against the native CUDA kernel.

    ``quantized_max_pool2d_cudnn`` is the kernel this operator replaces on the
    accelerator, and qint8 is the only dtype it accepts (quint8/qint32 raise
    "TensorDescriptor does not support ..."). It is also the only case where a
    native same-device reference exists, so the layout *and* the values are
    checked against it here rather than against the CPU kernel, whose output
    layout differs for a non-channels-last input.
    """
    # (shape, kernel_size, stride, padding, dilation, ceil_mode). Dilation must
    # stay 1: cuDNN rejects anything else, so a dilated config has no native
    # CUDA reference to compare against.
    configs = [
        ((2, 4, 16, 16), (2, 2), [], 0, False),
        ((4, 8, 32, 32), 3, 2, 1, False),
        ((2, 3, 28, 28), (3, 5), (1, 2), (1, 0), False),
        ((2, 4, 15, 15), 3, 2, 1, True),
        ((1, 16, 56, 56), 3, 2, 1, False),
        ((2, 2, 16, 20), 2, 2, (1, 0), False),
        ((1, 2, 6, 6), 4, 4, 2, False),
    ]
    scale = 0.05

    for shape, kernel_size, stride, padding, ceil_mode in configs:
        for layout in ("nchw", "channels_last", "transposed", "sliced"):
            if layout == "sliced":
                # Needs a stride-2 view, so build a wider tensor first.
                wide = torch._make_per_tensor_quantized_tensor(
                    torch.randint(-128, 127, (*shape[:-1], shape[-1] * 2))
                    .to(torch.int8)
                    .to(device),
                    scale=scale,
                    zero_point=0,
                )
                res_inp = wide[..., ::2]
            else:
                res_inp = torch._make_per_tensor_quantized_tensor(
                    torch.randint(-128, 127, shape, dtype=torch.int64)
                    .to(torch.int8)
                    .to(device),
                    scale=scale,
                    zero_point=0,
                )
                if layout == "channels_last":
                    res_inp = res_inp.contiguous(memory_format=torch.channels_last)
                elif layout == "transposed":
                    res_inp = res_inp.transpose(2, 3)

            kwargs = dict(
                stride=stride,
                padding=padding,
                dilation=1,
                ceil_mode=ceil_mode,
            )
            native = torch.quantized_max_pool2d(res_inp, kernel_size, **kwargs)
            gems = flag_gems.quantized_max_pool2d(res_inp, kernel_size, **kwargs)

            tag = f"{shape} ks={kernel_size} {layout}"
            assert gems.shape == native.shape, tag
            assert gems.stride() == native.stride(), tag
            assert gems.is_contiguous(
                memory_format=torch.channels_last
            ) == native.is_contiguous(memory_format=torch.channels_last), tag
            utils.gems_assert_equal(gems.int_repr().cpu(), native.int_repr().cpu())
            assert gems.q_scale() == native.q_scale(), tag
            assert gems.q_zero_point() == native.q_zero_point(), tag


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_3d_output_stays_contiguous(dtype):
    """A 3-D (C, H, W) input keeps an NCHW-contiguous output.

    The accelerator kernel views a 3-D input as a single batch and does not
    permute it, so both it and the CPU reference agree on an NCHW-contiguous
    result here -- for a contiguous and a transposed input alike.
    """
    zero_point = _valid_zero_point(dtype, 7)
    for shape in [(3, 16, 20), (3, 20, 16)]:
        res_inp = _make_quant_tensor(shape, dtype, 0.05, zero_point)
        if shape == (3, 20, 16):
            res_inp = res_inp.transpose(1, 2)
            assert not res_inp.is_contiguous()

        res_out = flag_gems.quantized_max_pool2d(res_inp, (2, 2))
        assert res_out.is_contiguous()
        ref_out = _reference(res_inp, (2, 2))
        assert res_out.stride() == ref_out.stride()
        _assert_same_quantized(res_out, ref_out, dtype)


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d_zero_batch():
    # A zero-element batch produces an empty output; only C/H/W must be
    # non-zero, so this is a legal input for the native operator.
    res_inp = _make_quant_tensor((0, 3, 8, 8), torch.quint8, 0.5, 3)

    ref_out = _reference(res_inp, [2, 2])
    res_out = flag_gems.quantized_max_pool2d(res_inp, [2, 2])
    assert res_out.shape == ref_out.shape
    assert res_out.numel() == 0
    assert res_out.dtype == torch.quint8


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_extreme_values(dtype):
    """Uniform and saturated inputs exercise the edges of the max reduction."""
    low, high = _QINT_RANGE[dtype]
    int_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]
    zero_point = _valid_zero_point(dtype, 0)

    for fill in (low, high, 0):
        ints = torch.full((2, 3, 8, 8), fill, dtype=int_dtype)
        res_inp = torch._make_per_tensor_quantized_tensor(
            ints.to(device), scale=0.1, zero_point=zero_point
        )
        # padding=1 with a 3x3 kernel makes the corner windows read padding,
        # so an all-``low`` input also checks the padding sentinel cannot win.
        ref_out = _reference(res_inp, 3, 1, 1)
        res_out = flag_gems.quantized_max_pool2d(res_inp, 3, 1, 1)
        utils.gems_assert_equal(res_out.int_repr().cpu(), ref_out.int_repr())


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_padding_sentinel(dtype):
    """A window lying entirely in the padding must not leak the fill value.

    ATen fills out-of-image positions with the *underlying integer* dtype's
    lowest value, so the result there is that value -- notably ``-2**31`` for
    qint32, not ``-128``.
    """
    low, _ = _QINT_RANGE[dtype]
    int_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]
    # Every real value is the dtype minimum, so the max over any window that
    # mixes real values with padding is still the minimum.
    ints = torch.full((1, 2, 6, 6), low, dtype=int_dtype)
    res_inp = torch._make_per_tensor_quantized_tensor(
        ints.to(device), scale=0.5, zero_point=_valid_zero_point(dtype, 0)
    )
    ref_out = _reference(res_inp, 4, 4, 2)
    res_out = flag_gems.quantized_max_pool2d(res_inp, 4, 4, 2)
    utils.gems_assert_equal(res_out.int_repr().cpu(), ref_out.int_repr())
    assert int(res_out.int_repr().min().item()) == low


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("stride", [[], (), None])
def test_quantized_max_pool2d_default_stride(stride):
    """An empty/omitted stride falls back to kernel_size (schema default)."""
    res_inp = _make_quant_tensor((2, 3, 12, 12), torch.quint8, 0.2, 5)
    ref_out = _reference(res_inp, (3, 3), [] if stride is None else stride)
    res_out = flag_gems.quantized_max_pool2d(res_inp, (3, 3), stride)
    _assert_same_quantized(res_out, ref_out, torch.quint8)


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d_runs_triton_kernel(monkeypatch):
    """Guard: both entry points really launch the Triton kernel.

    The tests in this file call ``flag_gems.quantized_max_pool2d`` /
    ``flag_gems.quantized_max_pool2d_out`` and compare against a native CPU
    reference. That comparison is only meaningful if the FlagGems side is the
    Triton kernel and not a delegation to PyTorch, so this counts the launches.

    Note the scope: this establishes that the *entry points* are the Triton
    implementation. It says nothing about whether
    ``torch.quantized_max_pool2d`` dispatches to FlagGems -- that depends on the
    ``QUANTIZED_CUDA_DISPATCH_KEY`` registration and is not covered by this file.
    """
    # ``flag_gems.ops.quantized_max_pool2d`` is re-exported as the function, so
    # reach the module itself to observe the Triton launch.
    gems_impl = importlib.import_module("flag_gems.ops.quantized_max_pool2d")

    calls = []
    original = gems_impl.quantized_max_pool2d_kernel

    class _Counting:
        def __getitem__(self, grid):
            calls.append(grid)
            return original[grid]

    monkeypatch.setattr(gems_impl, "quantized_max_pool2d_kernel", _Counting())

    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.1, 5)
    flag_gems.quantized_max_pool2d(res_inp, (2, 2))
    assert calls, "flag_gems.quantized_max_pool2d did not launch the Triton kernel"

    calls.clear()
    out_tensor = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.1, zero_point=5, dtype=torch.quint8, device=device
    )
    flag_gems.quantized_max_pool2d_out(res_inp, (2, 2), out=out_tensor)
    assert calls, "flag_gems.quantized_max_pool2d_out did not launch the Triton kernel"


@pytest.mark.quantized_max_pool2d_out
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANT_MAX_POOL2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", [0.1, 1.5])
@pytest.mark.parametrize("zero_point", [0, -128])
def test_quantized_max_pool2d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype, scale, zero_point
):
    zero_point = _valid_zero_point(dtype, zero_point)
    res_inp = _make_quant_tensor(shape, dtype, scale, zero_point)

    ref_out = _reference(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # A correctly shaped out carrying *different* quantization parameters: the
    # operator must overwrite them with the input's, as ATen's copy_ does.
    out_tensor = torch._empty_affine_quantized(
        ref_out.shape,
        scale=scale * 3.0,
        zero_point=_valid_zero_point(dtype, zero_point + 1),
        dtype=dtype,
        device=device,
    )
    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp,
        _as_int_pair(kernel_size),
        _as_int_pair(stride, allow_empty=True),
        _as_int_pair(padding),
        _as_int_pair(dilation),
        ceil_mode,
        out=out_tensor,
    )

    assert res_r is out_tensor
    _assert_same_quantized(res_r, ref_out, dtype)


def _as_int_pair(value, allow_empty=False):
    """Expand a pooling parameter to the 2-element list the schema wants."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value, value]
    value = list(value)
    if not value and allow_empty:
        return []
    return value


@pytest.mark.quantized_max_pool2d_out
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_out_resizes(dtype):
    """A wrongly shaped out is resized to the result's shape, as ATen does."""
    res_inp = _make_quant_tensor(
        (2, 3, 16, 16), dtype, 0.25, _valid_zero_point(dtype, 4)
    )
    ref_out = _reference(res_inp, (2, 2))

    for out_shape in [(1,), (0,), (2, 3, 2, 2), (2, 3, 9, 9), (7,)]:
        out_tensor = torch._empty_affine_quantized(
            out_shape, scale=9.0, zero_point=0, dtype=dtype, device=device
        )
        res_r = flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
        )
        assert res_r is out_tensor
        assert tuple(res_r.shape) == tuple(ref_out.shape)
        _assert_same_quantized(res_r, ref_out, dtype)


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_preserves_offset_and_strides():
    """A strided/offset out is written through its own strides.

    ``out`` is a non-contiguous view into a larger buffer; the operator must fill
    exactly that view and leave the surrounding elements untouched.
    """
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    # Strided view: every other column of a wider buffer.
    buffer = torch._empty_affine_quantized(
        (2, 3, 4, 8), scale=0.25, zero_point=7, dtype=torch.quint8, device=device
    )
    buffer.copy_(
        torch._make_per_tensor_quantized_tensor(
            torch.full((2, 3, 4, 8), 11, dtype=torch.uint8, device=device),
            scale=0.25,
            zero_point=7,
        )
    )
    view = buffer[..., ::2]
    flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=view
    )
    utils.gems_assert_equal(view.int_repr().cpu(), ref_out.int_repr())
    # The interleaved columns that are not part of ``out`` keep their sentinel.
    assert bool((buffer[..., 1::2].int_repr() == 11).all().item())

    # Offset view: the second slice of a batched buffer.
    big = torch._empty_affine_quantized(
        (2, 2, 3, 4, 4), scale=0.25, zero_point=7, dtype=torch.quint8, device=device
    )
    big.copy_(
        torch._make_per_tensor_quantized_tensor(
            torch.full((2, 2, 3, 4, 4), 11, dtype=torch.uint8, device=device),
            scale=0.25,
            zero_point=7,
        )
    )
    offset_view = big[1]
    assert offset_view.storage_offset() != 0
    flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=offset_view
    )
    utils.gems_assert_equal(offset_view.int_repr().cpu(), ref_out.int_repr())
    assert bool((big[0].int_repr() == 11).all().item())


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_channels_last_out():
    """A channels_last out of the right shape keeps its layout."""
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    out_tensor = torch._empty_affine_quantized(
        (2, 3, 4, 4),
        scale=0.25,
        zero_point=7,
        dtype=torch.quint8,
        device=device,
        memory_format=torch.channels_last,
    )
    flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
    )
    assert out_tensor.is_contiguous(memory_format=torch.channels_last)
    utils.gems_assert_equal(out_tensor.int_repr().cpu(), ref_out.int_repr())


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_aliasing_input():
    """An out that aliases the input still produces the functional result."""
    res_inp = _make_quant_tensor((1, 1, 4, 4), torch.quint8, 0.5, 0)
    ref_out = _reference(res_inp, [2, 2])

    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=res_inp
    )
    utils.gems_assert_equal(res_r.int_repr().cpu(), ref_out.int_repr())


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_mismatched_qparams():
    """A per-tensor out with different qparams adopts the input's.

    ``out.copy_`` is what moves the quantizer in ATen; the operator must
    reproduce that without a second native pass over the values, so this pins
    the observable contract (same object, resized shape, input's qparams).
    """
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    out_tensor = torch._empty_affine_quantized(
        (1,), scale=9.0, zero_point=1, dtype=torch.quint8, device=device
    )
    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
    )
    assert res_r is out_tensor
    assert tuple(res_r.shape) == tuple(ref_out.shape)
    assert res_r.q_scale() == res_inp.q_scale()
    assert res_r.q_zero_point() == res_inp.q_zero_point()
    _assert_same_quantized(res_r, ref_out, torch.quint8)


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_rejects_per_channel_out():
    """A per-channel out raises ATen's "same qscheme" message.

    Nothing can be written in that case: the kernel stores integers and has no
    way to install a per-tensor quantizer over a per-channel tensor, which is
    why ATen's quantized ``copy_`` rejects it.
    """
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    per_channel = torch.quantize_per_channel(
        torch.randn(2, 3, 4, 4, device=device),
        torch.rand(3, device=device) + 0.1,
        torch.zeros(3, dtype=torch.long, device=device),
        1,
        torch.quint8,
    )

    if device == "cpu":
        with pytest.raises(RuntimeError, match="same qscheme"):
            torch.ops.aten.quantized_max_pool2d.out(
                _to_cpu_keep_layout(res_inp),
                [2, 2],
                [],
                [0, 0],
                [1, 1],
                False,
                out=per_channel,
            )
    with pytest.raises(RuntimeError, match="same qscheme"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=per_channel
        )


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_rejects_mismatched_out():
    """dtype/device mismatches raise the same messages as ATen."""
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)

    bad_dtype = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.25, zero_point=0, dtype=torch.qint8, device=device
    )
    with pytest.raises(RuntimeError, match="dtype c10::quint8"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=bad_dtype
        )

    bad_device = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.25, zero_point=7, dtype=torch.quint8, device="cpu"
    )
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=bad_device
        )


# Each entry is (kwargs, the exact message ATen raises).
INVALID_PARAM_CASES = [
    (dict(kernel_size=[2, 2, 2]), "Expected 1d or 2d kernel size, got 3"),
    (dict(kernel_size=[]), "Expected 1d or 2d kernel size, got 0"),
    (
        dict(kernel_size=[2, 2], stride=[2]),
        "Expected no strides or 2d strides, got1",
    ),
    (
        dict(kernel_size=[2, 2], stride=[2, 2, 2]),
        "Expected no strides or 2d strides, got3",
    ),
    (
        dict(kernel_size=[3, 3], padding=[1, 1, 1]),
        "Expected 1d or 2d padding, got 3",
    ),
    (
        dict(kernel_size=[3, 3], dilation=[1, 1, 1]),
        "Expected 1d or 2d dilation, got 3",
    ),
    (dict(kernel_size=[2, 2], dilation=[0, 0]), "Expected dilation >= 1"),
    (dict(kernel_size=[2, 2], dilation=[-1, -1]), "Expected dilation >= 1"),
    (dict(kernel_size=[0, 0]), "kernel_size should be greater than zero."),
    (dict(kernel_size=[-1, -1]), "kernel_size should be greater than zero."),
    (
        dict(kernel_size=[2, 2], stride=[0, 0]),
        "strides should be greater than zero.",
    ),
    (
        dict(kernel_size=[2, 2], stride=[-1, -1]),
        "strides should be greater than zero.",
    ),
    (
        dict(kernel_size=[2, 2], padding=[-1, -1]),
        "pad must be non-negative, but got pad: -1",
    ),
    (
        dict(kernel_size=[2, 2], padding=[2, 2]),
        "padding should be smaller than half of kernel_size.",
    ),
    (
        dict(kernel_size=[3, 3], padding=[2, 2]),
        "padding should be smaller than half of kernel_size.",
    ),
    (
        dict(kernel_size=[16, 16]),
        "Given input size: (3x8x8). Calculated output size: (3x0x0). "
        "Output size is too small.",
    ),
]


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize("kwargs, message", INVALID_PARAM_CASES)
def test_quantized_max_pool2d_invalid_params(kwargs, message):
    """Bad parameters raise the same RuntimeError message ATen raises."""
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.1, 5)
    ref_inp = res_inp.cpu()

    kernel_size = kwargs.pop("kernel_size")
    stride = kwargs.pop("stride", [])
    padding = kwargs.pop("padding", [0, 0])
    dilation = kwargs.pop("dilation", [1, 1])

    # Confirm the expected message really is the native one, then require the
    # FlagGems path to reproduce it.
    with pytest.raises(RuntimeError) as native:
        torch.ops.aten.quantized_max_pool2d(
            ref_inp, kernel_size, stride, padding, dilation
        )
    assert message in str(native.value)

    with pytest.raises(RuntimeError) as gems:
        flag_gems.quantized_max_pool2d(res_inp, kernel_size, stride, padding, dilation)
    assert message in str(gems.value)


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize(
    "shape, message",
    [
        ((8, 8), "Expecting the input tensor of rank 3 or 4."),
        ((1, 2, 3, 8, 8), "Expecting the input tensor of rank 3 or 4."),
        ((2, 0, 8, 8), "input dimensions must be non-zero."),
        ((2, 3, 0, 8), "input dimensions must be non-zero."),
        ((2, 3, 8, 0), "input dimensions must be non-zero."),
        ((0, 8, 8), "input dimensions must be non-zero."),
    ],
)
def test_quantized_max_pool2d_invalid_shapes(shape, message):
    """Unsupported ranks and zero C/H/W raise ATen's messages."""
    res_inp = _make_quant_tensor(shape, torch.quint8, 0.1, 5)
    ref_inp = res_inp.cpu()

    with pytest.raises(RuntimeError) as native:
        torch.ops.aten.quantized_max_pool2d(ref_inp, [2, 2])
    assert message in str(native.value)

    with pytest.raises(RuntimeError) as gems:
        flag_gems.quantized_max_pool2d(res_inp, [2, 2])
    assert message in str(gems.value)


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d_rejects_per_channel_input():
    """A per-channel input is rejected, as the native operator does."""
    per_channel = torch.quantize_per_channel(
        torch.randn(2, 3, 8, 8),
        torch.rand(3) + 0.1,
        torch.zeros(3, dtype=torch.long),
        1,
        torch.quint8,
    ).to(device)

    # Establish the native message first, then require FlagGems to reproduce it.
    with pytest.raises(RuntimeError, match="kPerTensorAffine"):
        torch.quantized_max_pool2d(per_channel.cpu(), [2, 2])

    with pytest.raises(RuntimeError, match="kPerTensorAffine"):
        flag_gems.quantized_max_pool2d(per_channel, [2, 2])
