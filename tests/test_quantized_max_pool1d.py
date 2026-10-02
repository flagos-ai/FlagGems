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

# quantized_max_pool1d operates on quantized tensors (torch.quint8 by default).
# The reference implementation only exists on the CPU backend, so we always
# compare the CUDA FlagGems result against the CPU PyTorch result.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Pooling is applied along the last dimension. Shapes cover both 2D (N, L)
# and 3D (N, C, L) inputs as well as a few larger reduce dimensions.
POOL_SHAPES = [
    (4, 16),  # 2D input (N, L)
    (2, 3, 16),  # 3D input (N, C, L)
    (1, 8),  # minimal 2D
    (1, 1, 8),  # minimal 3D
    (32, 50257),  # large reduce dim, sequence-like
    (8, 3, 8192),  # large reduce dim, channel-like
]

# (kernel_size, stride, padding, dilation, ceil_mode)
POOL_CONFIGS = [
    (2, 2, 0, 1, False),
    (3, 2, 1, 1, False),
    (3, 2, 1, 1, True),  # ceil_mode
    (2, 1, 0, 1, False),  # stride=1, no padding
    (2, 1, 0, 2, False),  # dilation
    (5, 3, 2, 1, False),  # larger kernel
    (3, 2, 1, 1, True),  # ceil_mode + padding
]


def _make_quantized(shape, scale, zero_point, dtype, device):
    fp = torch.randn(shape, device="cpu").clamp_(-2, 2)
    return torch.quantize_per_tensor(
        fp, scale=scale, zero_point=zero_point, dtype=dtype
    ).to(device)


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d(
    shape, kernel_size, stride, padding, dilation, ceil_mode, in_dtype
):
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_inp = res_inp.to("cpu")

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    res_out = flag_gems.quantized_max_pool1d(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # Compare dequantized values; pool over the last dim so reduce_dim=1.
    # The native op only exists on CPU, so the reference runs on CPU and the
    # FlagGems result is moved to CPU before comparison.
    utils.gems_assert_close(
        res_out.to("cpu").dequantize(),
        ref_out.dequantize(),
        dtype=torch.float32,
        reduce_dim=1,
    )


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, in_dtype
):
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_inp = res_inp.to("cpu")

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    scale = float(res_inp.q_scale())
    zero_point = int(res_inp.q_zero_point())
    out_shape = ref_out.shape
    res_out = torch.quantize_per_tensor(
        torch.zeros(out_shape), scale=scale, zero_point=zero_point, dtype=in_dtype
    ).to(flag_gems.device)

    got = flag_gems.quantized_max_pool1d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=res_out,
    )

    # The .out variant must return the same tensor that was passed in.
    assert got.data_ptr() == res_out.data_ptr()
    assert got.dtype == in_dtype
    utils.gems_assert_close(
        res_out.to("cpu").dequantize(),
        ref_out.dequantize(),
        dtype=torch.float32,
        reduce_dim=1,
    )


# Padding contributes a neutral element chosen from the *underlying* integer
# dtype, so qint32 must pad with INT32_MIN. Padding with -128 would beat any
# real value below -128, which only qint32 can represent.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_padding_neutral(in_dtype):
    # Values chosen so every window that touches padding has all its real
    # values below the qint8 floor of -128.
    if in_dtype == torch.qint32:
        scale, values = 1.0, [-200.0, -300.0, -400.0, -129.0, -500.0, -600.0]
    else:
        scale, values = 0.1, [-12.8, -12.8, -12.7, -12.8, -12.8, -12.8]
    ref_inp = torch.quantize_per_tensor(
        torch.tensor([values]), scale=scale, zero_point=0, dtype=in_dtype
    )
    res_inp = ref_inp.to(flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 3, stride=2, padding=1)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)

    # Exact integer equality: any wrong neutral element shows up here.
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


# A window can consist entirely of padding once dilation spreads it past both
# edges, and the result is then purely the neutral element.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_all_padding_window(in_dtype):
    scale = 1.0 if in_dtype == torch.qint32 else 0.1
    ref_inp = torch.quantize_per_tensor(
        torch.tensor([[5.0]]), scale=scale, zero_point=0, dtype=in_dtype
    )
    res_inp = ref_inp.to(flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 2, stride=1, padding=1, dilation=2)
    res_out = flag_gems.quantized_max_pool1d(
        res_inp, 2, stride=1, padding=1, dilation=2
    )

    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_non_contiguous(in_dtype):
    """A strided out must be filled in place without touching its neighbours."""
    shape, out_len = (2, 3, 16), 8
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(res_inp.to("cpu"), 3, stride=2, padding=1)
    assert ref_out.shape[-1] == out_len

    # A view with stride 2 along the last dim: the odd lanes must survive.
    buffer = _make_quantized(
        shape[:-1] + (out_len * 2,), 0.5, 0, in_dtype, flag_gems.device
    )
    out = buffer[..., ::2]
    untouched_before = buffer.int_repr()[..., 1::2].to("cpu").clone()

    got = flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out.int_repr().to("cpu"), ref_out.int_repr())
    assert torch.equal(buffer.int_repr()[..., 1::2].to("cpu"), untouched_before)


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_storage_offset(in_dtype):
    """An out with a nonzero storage offset must be written at that offset."""
    shape, out_len, head = (2, 3, 16), 8, 4
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(res_inp.to("cpu"), 3, stride=2, padding=1)

    buffer = _make_quantized(
        shape[:-1] + (out_len + 2 * head,), 0.5, 0, in_dtype, flag_gems.device
    )
    out = buffer[..., head : head + out_len]
    before = buffer.int_repr().to("cpu").clone()

    flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)

    assert torch.equal(out.int_repr().to("cpu"), ref_out.int_repr())
    assert torch.equal(buffer.int_repr()[..., :head].to("cpu"), before[..., :head])
    assert torch.equal(
        buffer.int_repr()[..., head + out_len :].to("cpu"),
        before[..., head + out_len :],
    )


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_resizes_and_adopts_quant_params(in_dtype):
    """A wrongly-shaped out is resized and takes the input's quant params."""
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(res_inp.to("cpu"), 3, stride=2, padding=1)

    # Wrong shape and deliberately different scale/zero_point.
    out = torch.quantize_per_tensor(
        torch.zeros(1, 1, 1), scale=0.5, zero_point=1, dtype=in_dtype
    ).to(flag_gems.device)

    got = flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)

    assert tuple(got.shape) == tuple(ref_out.shape)
    assert got.q_scale() == ref_out.q_scale()
    assert got.q_zero_point() == ref_out.q_zero_point()
    assert torch.equal(got.int_repr().to("cpu"), ref_out.int_repr())


# ``out is input`` is a legal call (aten tolerates an aliasing out). The pool
# changes the input's shape, so this exercises the ordering inside the .out
# path: if the input's integer representation is captured after ``out`` has
# been resized, the kernel pools the resized tensor instead of the original.
#
# The failure is silent rather than loud, and shows up differently depending on
# which way the resize goes. Shrinking (8 -> 4) leaves the first half intact and
# pads the tail with the previous buffer contents, so the trailing outputs come
# back wrong; growing (8 -> 9) makes ``_resize_out`` reallocate the storage, so
# the input bytes are gone entirely and every output reads back zero.
#
# The reviewer asked specifically for L=8, kernel_size=2, stride=2 (shrink), so
# that case leads; the 3D and growing variants guard the other two directions.
@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation",
    [
        ((1, 8), 2, 2, 0, 1),  # the reviewer's case: 8 -> 4, shrinks
        ((2, 3, 8), 2, 2, 0, 1),  # 3D shrink
        ((1, 8), 2, 1, 1, 1),  # 8 -> 9, grows and reallocates
    ],
)
def test_quantized_max_pool1d_out_is_input(
    in_dtype, shape, kernel_size, stride, padding, dilation
):
    """``out=input`` must pool the *original* input, not the resized one."""
    # Deterministic data derived from the case, not from a global RNG, so the
    # values a failure prints are reproducible.
    seed = sum(shape) * 1000 + kernel_size * 100 + stride * 10 + padding
    gen = torch.Generator().manual_seed(seed)
    fp = torch.randint(0, 100, shape, generator=gen).to(torch.float32)
    ref_inp = torch.quantize_per_tensor(fp, 1.0, 0, in_dtype)
    res_inp = ref_inp.to(flag_gems.device)

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    # Guard the test itself: the resize must actually change the shape, or the
    # ordering this covers would never be exercised.
    assert tuple(ref_out.shape) != tuple(ref_inp.shape)

    got = flag_gems.quantized_max_pool1d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        out=res_inp,
    )

    # The result is written through the tensor that was passed in.
    assert got.data_ptr() == res_inp.data_ptr()
    assert tuple(got.shape) == tuple(ref_out.shape)
    assert got.q_scale() == ref_out.q_scale()
    assert got.q_zero_point() == ref_out.q_zero_point()
    # Integer equality: a truncated or zeroed input cannot pass this.
    assert torch.equal(got.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out_rejects_mismatched_out():
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, torch.quint8, flag_gems.device)
    bad_dtype = torch.quantize_per_tensor(
        torch.zeros(2, 3, 8), scale=0.1, zero_point=0, dtype=torch.qint8
    ).to(flag_gems.device)
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.quantized_max_pool1d_out(
            res_inp, 3, stride=2, padding=1, out=bad_dtype
        )


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out_rejects_wrong_device():
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, torch.quint8, flag_gems.device)
    cpu_out = torch.quantize_per_tensor(
        torch.zeros(2, 3, 8), scale=0.1, zero_point=0, dtype=torch.quint8
    )
    with pytest.raises(RuntimeError, match="device"):
        flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=cpu_out)


# Parameter validation, matched against the messages aten's max_pool1d emits.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kernel_size": 0}, "kernel_size must be greater than zero"),
        ({"kernel_size": -1}, "kernel_size must be greater than zero"),
        ({"kernel_size": [2, 2]}, "kernel_size must be an int"),
        ({"kernel_size": []}, "kernel_size must be an int"),
        ({"kernel_size": 2, "stride": 0}, "stride must be greater than zero"),
        ({"kernel_size": 2, "stride": [2, 2]}, "stride must be None"),
        ({"kernel_size": 2, "padding": -1}, "padding must be non-negative"),
        ({"kernel_size": 3, "padding": 2}, "padding should be at most half"),
        ({"kernel_size": 2, "padding": [0, 0]}, "padding must be an int"),
        ({"kernel_size": 2, "dilation": 0}, "dilation must be greater than zero"),
        ({"kernel_size": 2, "dilation": [1, 1]}, "dilation must be an int"),
        ({"kernel_size": 20}, "Invalid computed output size"),
    ],
)
def test_quantized_max_pool1d_invalid_params(kwargs, message):
    res_inp = _make_quantized((2, 3, 8), 0.1, 0, torch.quint8, flag_gems.device)
    ref_inp = res_inp.to("cpu")

    # The same call must fail the same way on the aten reference.
    with pytest.raises(RuntimeError, match=message):
        torch.quantized_max_pool1d(ref_inp, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        flag_gems.quantized_max_pool1d(res_inp, **kwargs)


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "shape, message",
    [
        ((8,), "Expected 2D or 3D input tensor"),
        ((2, 2, 2, 8), "Expected 2D or 3D input tensor"),
        ((2, 0, 8), "input dimensions must be non-zero"),
        ((2, 3, 0), "Invalid computed output size"),
    ],
)
def test_quantized_max_pool1d_invalid_shapes(shape, message):
    res_inp = _make_quantized(shape, 0.1, 0, torch.quint8, flag_gems.device)
    with pytest.raises(RuntimeError, match=message):
        torch.quantized_max_pool1d(res_inp.to("cpu"), 2, stride=2)
    with pytest.raises(RuntimeError, match=message):
        flag_gems.quantized_max_pool1d(res_inp, 2, stride=2)


# A zero-size batch is legal for 3D input and yields an empty result.
@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d_empty_batch():
    res_inp = _make_quantized((0, 3, 8), 0.1, 0, torch.quint8, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(res_inp.to("cpu"), 2, stride=2)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 2, stride=2)
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.numel() == 0


# Non-contiguous inputs: pooling reads along the last dim, so a strided or
# transposed input must be gathered with the right element stride.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("layout", ["slice", "transpose", "offset"])
def test_quantized_max_pool1d_non_contiguous_input(in_dtype, layout):
    def _view(base):
        if layout == "slice":
            return base[:, :, ::2]
        if layout == "transpose":
            return base.transpose(1, 2)
        return base[:, :, 3:19]

    # Tensor.to("cpu") is a silent no-op for a non-contiguous quantized tensor,
    # so the reference view is built on a CPU tensor from the start.
    cpu_base = _make_quantized((2, 4, 24), 0.1, 0, in_dtype, "cpu")
    ref_inp = _view(cpu_base)
    res_inp = _view(cpu_base.to(flag_gems.device))
    assert not res_inp.is_contiguous()
    assert res_inp.device.type != "cpu"

    ref_out = torch.quantized_max_pool1d(ref_inp, 3, stride=2, padding=1)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


# Nonzero zero_point must be carried through to the output unchanged.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "in_dtype, scale, zero_point",
    [
        (torch.quint8, 0.1, 128),
        (torch.qint8, 0.1, -32),
        (torch.qint32, 1.0, 7),
    ],
)
def test_quantized_max_pool1d_quant_params(in_dtype, scale, zero_point):
    res_inp = _make_quantized((2, 3, 16), scale, zero_point, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(res_inp.to("cpu"), 3, stride=2, padding=1)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)

    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


def _aten_unit_length_layout_is_broken(q_input):
    """The documented domain of the ATen layout defect described below.

    Used only to attribute an observed mismatch: if ATen and ``max_pool1d``
    disagree outside this domain, something else changed and the test must not
    silently repair it.
    """
    return q_input.dim() == 3 and q_input.shape[1] > 1 and q_input.shape[-1] == 1


def _aten_unit_length_int_repr(out):
    """The integer result ATen's mislabelled buffer actually holds.

    See ``test_quantized_max_pool1d_unit_length`` for the full description. The
    bytes are correct; only the strides are wrong, so reading the raw buffer as
    (N, L, C) and permuting back to (N, C, L) recovers the intended result.
    """
    n, c = out.shape[0], out.shape[1]
    out_l = out.shape[-1]
    return out.int_repr().reshape(n, out_l, c).permute(0, 2, 1).contiguous()


# L=1 with C>1: `quantized_max_pool1d` unsqueezes a (N, C, L) input to
# (N, C, 1, L) and dispatches to the 2D kernel. When L == 1 that 4D tensor is
# *trivially* channels-last-contiguous, so `q_maxpool_2d` takes its NHWC fast
# path, writes the result in NHWC (here NLC) order, and returns a tensor whose
# strides nonetheless claim a contiguous NCL layout. The bytes are right and the
# strides lie, so the *logical* values are wrong. This is an ATen CPU defect, not
# a FlagGems one: an independent window-max reference and the float
# `max_pool1d` both agree with FlagGems, and ATen's own buffer read as NLC
# reproduces the correct answer exactly.
#
# Reproduce on torch 2.11.0:
#     q = torch.quantize_per_tensor(
#         torch.arange(1.0, 7.0).reshape(2, 3, 1), 1.0, 0, torch.quint8
#     )
#     torch.quantized_max_pool1d(q, 2, stride=1, padding=1).int_repr()
#     # -> [[[1, 2], [3, 1], [2, 3]], [[4, 5], [6, 4], [5, 6]]]   (logical, wrong)
#     torch.nn.functional.max_pool1d(q.int_repr().float(), 2, stride=1, padding=1)
#     # -> [[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]   (correct)
#     # ATen's buffer read as NLC equals the correct result:
#     torch.quantized_max_pool1d(q, 2, stride=1, padding=1).int_repr()
#         .reshape(2, 2, 3).permute(0, 2, 1)
#     # -> [[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]
# Only 3D inputs with L == 1 and C > 1 are affected; 2D (N, L) inputs and 3D
# inputs with C == 1 or L > 1 match the reference exactly.
#
# This case is therefore still compared against `torch.quantized_max_pool1d`;
# the expectation is the native operator's output with its mislabelled buffer
# re-read, which the test verifies against an independent reference. Remove the
# repair (and this note) once the upstream kernel is fixed.
@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("shape", [(2, 3, 1), (1, 3, 1), (3, 5, 1), (2, 1, 1)])
def test_quantized_max_pool1d_unit_length(in_dtype, shape):
    scale = 1.0
    numel = 1
    for size in shape:
        numel *= size
    fp = torch.arange(1.0, numel + 1.0).reshape(shape)
    ref_inp = torch.quantize_per_tensor(fp, scale, 0, in_dtype)
    res_inp = ref_inp.to(flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 2, stride=1, padding=1)

    # Pool the integer representation directly: comparing dequantized floats
    # would trip over scale round-trip error rather than the layout.
    expected = torch.nn.functional.max_pool1d(
        ref_inp.int_repr().to(torch.float32), 2, stride=1, padding=1
    )

    if not torch.equal(ref_out.int_repr().to(torch.float32), expected):
        # ATen hit the layout defect described above. Pin it so the repair below
        # cannot silently become stale: re-reading the buffer as NLC *must*
        # reproduce the independent reference exactly, otherwise this is a
        # different failure and the test should fail rather than paper over it.
        assert _aten_unit_length_layout_is_broken(ref_inp), (
            "torch.quantized_max_pool1d disagrees with max_pool1d on "
            f"{tuple(ref_inp.shape)}, which the documented ATen layout defect "
            "does not explain"
        )
        ref_int_repr = _aten_unit_length_int_repr(ref_out)
        assert torch.equal(ref_int_repr.to(torch.float32), expected)
    else:
        # Outside the affected domain the native op already agrees, so the
        # comparison above needs no repair.
        ref_int_repr = ref_out.int_repr()

    res_out = flag_gems.quantized_max_pool1d(res_inp, 2, stride=1, padding=1)

    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert torch.equal(res_out.int_repr().to("cpu"), ref_int_repr)
