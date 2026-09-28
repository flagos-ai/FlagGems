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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# torch.rand / torch.randn have no FP8 implementation, so FP8 inputs are built
# in FP32 and cast; FP8 results are also compared bitwise because torch rejects
# non-zero tolerances for 1-byte floats.
CONJ_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _supported_fp8_dtypes(device):
    """Keep only the FP8 dtypes the current device can actually materialize."""
    supported = []
    for dtype in CONJ_FP8_DTYPES:
        try:
            torch.randn(1, device=device).to(dtype)
        except Exception:
            continue
        supported.append(dtype)
    return supported


# Dtype sweep: the low-precision dtypes first, then the float and int sweep.
CONJ_DTYPES = [
    torch.int8,
    torch.uint8,
    *_supported_fp8_dtypes(flag_gems.device),
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
]

# Integer results must come back bit-identical, so they are compared exactly,
# together with FP8, which torch only compares bitwise.
CONJ_EXACT_DTYPES = (
    torch.int8,
    torch.uint8,
    *CONJ_FP8_DTYPES,
    torch.int32,
)

if cfg.QUICK_MODE:
    CONJ_SHAPES = [(2, 19, 7)]
    CONJ_VALUE_RANGES = ["[-1,1]"]
else:
    CONJ_SHAPES = [
        (),
        (1,),
        (256,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
    CONJ_VALUE_RANGES = ["[-1,1]", "[0,1]", "[-1,0]", "[0,max]", "[min,0]"]


def _bounds(dtype, value_range):
    """Inclusive (low, high) bounds of the values to generate."""
    fixed_ranges = {
        "[-1,1]": (-1.0, 1.0),
        "[0,1]": (0.0, 1.0),
        "[-1,0]": (-1.0, 0.0),
    }
    if value_range in fixed_ranges:
        return fixed_ranges[value_range]

    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    if value_range == "[0,max]":
        return 0.0, info.max
    return info.min, 0.0


def _make_input(shape, dtype, device, value_range):
    """Sample values inside ``value_range`` and pin both of its ends."""
    low, high = _bounds(dtype, value_range)
    if dtype.is_floating_point:
        # torch.rand has no FP8 implementation, so sample in FP32 and cast.
        values = torch.rand(shape, dtype=torch.float32, device=device)
        out = (values * (high - low) + low).to(dtype)
        edges = torch.tensor([low, high], dtype=torch.float32, device=device).to(dtype)
    else:
        # Unsigned dtypes cannot hold the negative end of [-1,1] / [-1,0], so
        # the bounds are clamped to what the dtype can represent.
        info = torch.iinfo(dtype)
        low, high = max(int(low), info.min), min(int(high), info.max)
        upper = high + 1
        out = torch.randint(low, upper, shape, dtype=dtype, device=device)
        edges = torch.tensor([low, high], dtype=dtype, device=device)

    if out.numel() > 1:
        out.view(-1)[:2] = edges
    return out


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", CONJ_SHAPES)
@pytest.mark.parametrize("value_range", CONJ_VALUE_RANGES)
@pytest.mark.parametrize("dtype", CONJ_DTYPES)
def test_conj_physical(shape, dtype, value_range):
    device = flag_gems.device
    input = _make_input(shape, dtype, device, value_range)
    out_dtype = dtype

    if dtype in CONJ_EXACT_DTYPES:
        # Compare against an unmodified copy of the input, so the exact
        # comparison does not depend on the upcast reference.
        ref_input = utils.to_reference(input.clone())
    else:
        ref_input = utils.to_reference(input, True)

    ref_out = torch.conj_physical(ref_input)
    res_out = flag_gems.conj_physical(input)

    if dtype in CONJ_EXACT_DTYPES:
        utils.gems_assert_equal(res_out, ref_out.to(out_dtype))
    else:
        utils.gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=1)


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", CONJ_SHAPES)
@pytest.mark.parametrize("value_range", CONJ_VALUE_RANGES)
def test_conj_physical_complex(shape, value_range):
    device = flag_gems.device
    real = _make_input(shape, torch.float32, device, value_range)
    imag = _make_input(shape, torch.float32, device, value_range)
    input = torch.complex(real, imag)
    out_dtype = input.dtype

    ref_input = utils.to_reference(input, True)
    ref_out = torch.conj_physical(ref_input)
    res_out = flag_gems.conj_physical(input)

    assert res_out.dtype == out_dtype
    # Not every device implements a complex comparison kernel: the Ascend NPU
    # rejects both isclose and equal for complex64 ("aclnnIsClose failed, error
    # code is 161002"), so compare the real and imaginary lanes as plain
    # float32, the same way tests/test_chalf.py does.
    utils.gems_assert_close(
        torch.view_as_real(res_out),
        torch.view_as_real(ref_out),
        torch.float32,
        reduce_dim=1,
    )


# ---------------------------------------------------------------------------
# aten::_conj_physical / aten::_conj_physical.out
#
# The private op shares the kernel with the public one above but differs in two
# contractual ways, both exercised here: it never aliases its input (even for a
# real dtype, where conjugation is a no-op), and it reports the input's layout
# rather than a contiguous one.
# ---------------------------------------------------------------------------


def _layout_case(name, shape, dtype, device):
    """Build an input with a specific layout; mirrors the ATen layout rules."""
    if dtype.is_complex or dtype.is_floating_point:
        base = torch.randn(shape, dtype=dtype, device=device)
    else:
        base = torch.randint(-9, 9, shape, dtype=dtype, device=device)

    if name == "contiguous":
        return base
    if name == "transposed":
        return base.t()
    if name == "channels_last":
        return base.to(memory_format=torch.channels_last)
    if name == "channels_last_3d":
        return base.to(memory_format=torch.channels_last_3d)
    if name == "gappy":
        # Non-overlapping but *not* dense: ATen returns a contiguous output.
        return base[:, ::2]
    if name == "expanded":
        # Overlapping (stride 0 over a size > 1 dim); also contiguous out.
        return base[:1].expand(shape[0], *shape[1:])
    raise AssertionError(name)


_LAYOUT_SHAPES = {
    "contiguous": (3, 4),
    "transposed": (4, 3),
    "channels_last": (2, 3, 4, 4),
    "channels_last_3d": (1, 4, 2, 3, 4),
    "gappy": (3, 8),
    "expanded": (3, 4),
}

_UNDERSCORE_DTYPES = [torch.complex64, torch.float32, torch.int32]


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("layout", list(_LAYOUT_SHAPES))
@pytest.mark.parametrize("dtype", _UNDERSCORE_DTYPES)
def test__conj_physical_layout_parity(layout, dtype):
    """Values *and* strides must match ATen for every layout.

    ATen allocates the output with ``preserve_format``: a dense input keeps its
    strides (so channels-last stays channels-last), while a non-dense one
    (``gappy``, ``expanded``) comes back contiguous. Building the output from a
    contiguous temporary would silently flatten the first group.
    """
    inp = _layout_case(layout, _LAYOUT_SHAPES[layout], dtype, flag_gems.device)

    # Layout is a property of the op, not of the device, so the stride contract
    # is pinned against a same-device ATen call; the value comparison below goes
    # through to_reference so it still works under --ref=cpu.
    aten_out = torch.ops.aten._conj_physical(inp)
    res_out = flag_gems.ops._conj_physical(inp)

    assert (
        res_out.stride() == aten_out.stride()
    ), f"{layout}: stride {res_out.stride()} != ATen's {aten_out.stride()}"
    assert res_out.shape == aten_out.shape

    ref_out = torch.ops.aten._conj_physical(utils.to_reference(inp))
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
)
def test__conj_physical_real_does_not_alias(dtype):
    """A real input is returned as a fresh tensor, not aliased.

    This is the one place ``_conj_physical`` deviates from the public
    ``conj_physical``, which may hand the input straight back.
    """
    if dtype == torch.bool:
        inp = torch.randint(0, 2, (3, 4), device=flag_gems.device).bool()
    elif dtype.is_floating_point:
        inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-9, 9, (3, 4), dtype=dtype, device=flag_gems.device)

    # Aliasing is a same-device property, so compare against ATen on-device.
    aten_out = torch.ops.aten._conj_physical(inp)
    res_out = flag_gems.ops._conj_physical(inp)

    # ATen itself must not alias here; assert that first so the test documents
    # the contract it is pinning rather than just gems' behaviour.
    assert aten_out.data_ptr() != inp.data_ptr()
    assert res_out.data_ptr() != inp.data_ptr(), "result aliases its input"
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()

    ref_out = torch.ops.aten._conj_physical(utils.to_reference(inp))
    utils.gems_assert_equal(res_out, ref_out)

    # Writing through the result must leave the input untouched.
    original = utils.to_reference(inp.clone())
    res_out.zero_()
    utils.gems_assert_equal(inp, original)


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test__conj_physical_conj_bit_input(dtype):
    """A conj-bit input conjugates its *logical* value: conj(conj(x)) == x."""
    base = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    inp = base.conj()
    assert inp.is_conj()

    res_out = flag_gems.ops._conj_physical(inp)
    assert not res_out.is_conj(), "conjugate bit leaked into the result"

    ref_out = torch.ops.aten._conj_physical(utils.to_reference(inp))
    utils.gems_assert_equal(torch.view_as_real(res_out), torch.view_as_real(ref_out))
    # conj(conj(base)) == base, so the result is the original tensor's values.
    utils.gems_assert_equal(
        torch.view_as_real(res_out), torch.view_as_real(utils.to_reference(base))
    )


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32, torch.int32])
@pytest.mark.parametrize("shape", [(0,), (0, 4), (3, 0)])
def test__conj_physical_empty(dtype, shape):
    """Zero-element inputs skip the launch but still return a correct tensor."""
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    aten_out = torch.ops.aten._conj_physical(inp)
    res_out = flag_gems.ops._conj_physical(inp)

    assert res_out.shape == aten_out.shape
    assert res_out.dtype == aten_out.dtype
    assert res_out.stride() == aten_out.stride()
    assert res_out.numel() == 0


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test__conj_physical_matches_backward_formula(dtype):
    """The forward result is what makes the ATen gradient formula come out right.

    ``_conj_physical`` is its own derivative (``d/dx conj(x)`` contributes
    ``conj(grad)``), so ATen's registered backward is another
    ``conj_physical`` call on the incoming gradient. Feeding the gradient
    through the implementation and comparing against ATen therefore checks the
    backward path's arithmetic without needing an autograd graph: the
    implementation writes into a freshly allocated buffer and so is not
    differentiable on its own, and building the graph would require wrapping the
    call at the dispatcher level.
    """
    grad = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)

    # What ATen's backward computes for an incoming gradient `grad`.
    ref_grad_in = torch.ops.aten._conj_physical(utils.to_reference(grad))
    res_grad_in = flag_gems.ops._conj_physical(grad)

    utils.gems_assert_equal(
        torch.view_as_real(res_grad_in), torch.view_as_real(ref_grad_in)
    )
    # Applying it twice is the identity, which pins the sign convention: a
    # missing negation would pass a single comparison against a wrong reference
    # but cannot survive the round trip.
    round_trip = flag_gems.ops._conj_physical(res_grad_in)
    utils.gems_assert_equal(
        torch.view_as_real(round_trip), torch.view_as_real(utils.to_reference(grad))
    )


@pytest.mark.underscore_conj_physical_out
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "channels_last"])
@pytest.mark.parametrize("dtype", _UNDERSCORE_DTYPES)
def test__conj_physical_out_layout_parity(layout, dtype):
    """``out=`` accepts a real or non-complex input and any strided target."""
    inp = _layout_case(layout, _LAYOUT_SHAPES[layout], dtype, flag_gems.device)

    aten_out = torch.empty_like(inp)
    torch.ops.aten._conj_physical.out(inp, out=aten_out)

    res_out = torch.empty_like(inp)
    returned = flag_gems.ops._conj_physical_out(inp, out=res_out)

    assert returned is res_out, "out= must return the out tensor itself"
    assert res_out.stride() == aten_out.stride()

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten._conj_physical.out(ref_inp, out=ref_out)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_conj_physical_out
@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32])
def test__conj_physical_out_non_contiguous(dtype):
    """A strided (non-contiguous) ``out`` is written through, not re-laid-out."""
    inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)

    def strided_out():
        # Every other element of a 2x-sized buffer: stride (8, 2), non-dense.
        return torch.empty(24, dtype=dtype, device=flag_gems.device)[::2].reshape(3, 4)

    res_out = strided_out()
    assert not res_out.is_contiguous()
    stride_before = res_out.stride()
    returned = flag_gems.ops._conj_physical_out(inp, out=res_out)

    assert returned is res_out
    assert res_out.stride() == stride_before, "out= re-laid-out its target"

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten._conj_physical.out(ref_inp, out=ref_out)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_conj_physical_out
@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32, torch.int32])
def test__conj_physical_out_resizes(dtype):
    """A mismatched ``out`` is resized, matching ATen."""
    if dtype.is_complex or dtype.is_floating_point:
        inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-9, 9, (3, 4), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    for out_shape in [(0,), (1, 1), (12,)]:
        aten_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        torch.ops.aten._conj_physical.out(inp, out=aten_out)

        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        flag_gems.ops._conj_physical_out(inp, out=res_out)

        assert res_out.shape == aten_out.shape == inp.shape

        ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
        torch.ops.aten._conj_physical.out(ref_inp, out=ref_out)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_conj_physical_out
def test__conj_physical_out_rejects_wrong_dtype():
    """Mismatched ``out`` dtype raises with ATen's exact message."""
    inp = torch.randn((3, 4), dtype=torch.complex64, device=flag_gems.device)
    bad_out = torch.empty((3, 4), dtype=torch.complex128, device=flag_gems.device)

    try:
        torch.ops.aten._conj_physical.out(inp, out=bad_out.clone())
        raise AssertionError("ATen unexpectedly accepted the mismatched out dtype")
    except RuntimeError as e:
        ref_message = str(e)

    with pytest.raises(RuntimeError) as exc_info:
        flag_gems.ops._conj_physical_out(inp, out=bad_out)

    assert str(exc_info.value) == ref_message
