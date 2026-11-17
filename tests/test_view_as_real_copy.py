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
from . import test_utils as tu

_ASCEND_COMPLEX_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend complex input/reference support is incomplete",
)


# aten::view_as_real_copy returns a complex tensor's real components as a fresh
# contiguous real tensor whose trailing dimension is 2. The native kernels only
# accept complex inputs, so the real / bool / integer dtypes below are negative
# cases rather than a missing dtype dimension.
_COMPLEX_DTYPES = utils.COMPLEX_DTYPES + (
    [torch.complex128] if utils.fp64_is_supported else []
)

_REAL_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
_COMPLEX_TO_REAL = {value: key for key, value in _REAL_TO_COMPLEX.items()}

# Rejected natively with "view_as_real is only supported for complex tensors".
# Dtypes the device cannot hold at all are left out, so each case asserts an
# operator rejection instead of a device allocation failure.
_UNSUPPORTED_DTYPES = (
    [torch.float32, torch.float16, torch.int8, torch.uint8, torch.int32, torch.bool]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.int64] if utils.int64_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
    + ([torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else [])
)

# The out buffer must have exactly the real component dtype of the input.
_WRONG_OUT_DTYPES = [torch.float16, torch.int32, torch.complex64] + (
    [torch.float64] if utils.fp64_is_supported else []
)

# Real dtypes whose special values can be lifted into a complex dtype; ATen has
# no complex bfloat16, so bfloat16 has no special-value case here.
_SPECIAL_REAL_DTYPES = tu.selected_cases(
    [torch.float16, torch.float32]
    + ([torch.float64] if utils.fp64_is_supported else []),
    quick=[],
)
_SPECIAL_CASES = tu.special_value_cases(_SPECIAL_REAL_DTYPES)

_NON_CONTIGUOUS_LAYOUTS = ["transpose", "offset", "slice"]

# A lazily conjugated input carrying a storage offset, and one carrying a
# zero-stride broadcast dimension.
_CONJUGATE_LAYOUTS = tu.selected_cases(["offset", "broadcast"], quick=[])

_EMPTY_SHAPES = tu.selected_cases([(0,), (4, 0), (2, 0, 3)], quick=[])

# The real-valued gradient is folded back into the complex input.
_BACKWARD_SHAPES = tu.selected_cases([(16, 64), (7, 13, 29), (4, 0)], quick=[])

# The supplementary groups below are default-only: quick keeps the main grid,
# its out form and every negative case, and none of these layouts.
_DEFAULT_ONLY_DTYPES = tu.selected_cases(_COMPLEX_DTYPES, quick=[])
_CONJUGATE_SHAPES = tu.selected_cases(tu.selected_shapes(), quick=[])
_OFFSET_BUFFER_SHAPES = tu.selected_cases([(8, 16, 32)], quick=[])
_MUTATION_SHAPES = tu.selected_cases([(4, 7), (2, 3, 5)], quick=[])


def _distinct_complex(shape, dtype):
    """Complex values whose real and imaginary parts differ everywhere.

    A swapped real/imaginary pair or a dropped conjugation cannot survive this
    input, unlike a symmetric random tensor.
    """
    numel = 1
    for extent in shape:
        numel *= extent
    real = torch.linspace(-1.0, 1.0, numel, device=flag_gems.device)
    real = real.reshape(shape).to(_COMPLEX_TO_REAL[dtype])
    return torch.complex(real, real + 0.5)


def _make_special_complex(real_dtype, scenario):
    # tu.make_special_input keeps the requested dtype and torch.complex lifts
    # matching real components to the complex dtype of that width.
    real = tu.make_special_input(real_dtype, scenario)
    return torch.complex(real, tu.make_special_input(real_dtype, scenario).flip(0))


def _assert_equal(actual, expected):
    if flag_gems.vendor_name != "mthreads":
        tu.assert_result_equal(actual, expected)
        return

    target = torch.empty(0, device=flag_gems.device).device
    assert actual.device == target
    assert actual.dtype == expected.dtype
    actual, expected = actual.cpu(), expected.cpu()
    if actual.is_complex():
        actual = torch.view_as_real(actual.resolve_conj())
        expected = torch.view_as_real(expected.resolve_conj())
    tu.assert_result_equal(actual, expected)


def _assert_copy_contract(res_out, ref_out, inp, ref_inp, allocating=True):
    """Native values, allocating geometry, and an input left untouched.

    Base-storage pointers are compared, so a result aliasing an offset view of
    the input is rejected as well; an empty result has no storage independence
    to observe, so that check is limited to non-empty inputs.
    """
    _assert_equal(res_out, ref_out)
    _assert_equal(inp, ref_inp)
    if allocating:
        # Compare the result geometry with the native allocating result instead
        # of assuming it; an out buffer keeps the caller's own layout.
        assert res_out.stride() == ref_out.stride()
        assert res_out.is_contiguous() == ref_out.is_contiguous()
    if inp.numel() > 0:
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _COMPLEX_DTYPES
    ],
)
def test_view_as_real_copy(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _COMPLEX_DTYPES
    ],
)
def test_view_as_real_copy_out(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    out_shape = shape + (2,)
    real_dtype = _COMPLEX_TO_REAL[dtype]

    ref_out = torch.full(out_shape, 7, dtype=real_dtype, device=ref_inp.device)
    out = torch.full(out_shape, 7, dtype=real_dtype, device=flag_gems.device)

    torch.ops.aten.view_as_real_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_real_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_contract(res_ret, ref_out, inp, ref_inp, allocating=False)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _OFFSET_BUFFER_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_out_offset_buffer(shape, dtype):
    # The out buffer is a non-contiguous, offset slice of a wider real tensor:
    # the components must land at that buffer's own offsets, the sentinel
    # padding around it must stay untouched, and the input must stay untouched.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    filled = shape + (4,)

    ref_base = torch.full(
        filled, 7, dtype=_COMPLEX_TO_REAL[dtype], device=ref_inp.device
    )
    base = torch.full(filled, 7, dtype=_COMPLEX_TO_REAL[dtype], device=flag_gems.device)
    ref_out = ref_base[..., 1:3]
    out = base[..., 1:3]

    torch.ops.aten.view_as_real_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_real_copy(inp, out=out)

    assert res_ret is out
    _assert_equal(out, ref_out)
    _assert_equal(base, ref_base)
    _assert_equal(inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_out_from_non_contiguous_input(dtype):
    base = tu.make_input(dtype, (8, 16, 32), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    out_shape = inp.shape + (2,)
    real_dtype = _COMPLEX_TO_REAL[dtype]

    ref_out = torch.full(out_shape, 7, dtype=real_dtype, device=ref_inp.device)
    out = torch.full(out_shape, 7, dtype=real_dtype, device=flag_gems.device)

    torch.ops.aten.view_as_real_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.view_as_real_copy(inp, out=out)

    assert res_ret is out
    _assert_copy_contract(res_ret, ref_out, inp, ref_inp, allocating=False)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _CONJUGATE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_conjugate_input(shape, dtype):
    # Unlike aten::view_as_real, this kernel resolves a lazily conjugated input
    # rather than raising, so the values are checked against a resolved
    # reference while the input keeps its conjugate bit.
    inp = _distinct_complex(shape, dtype).conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)
    assert inp.is_conj()


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("layout", _CONJUGATE_LAYOUTS)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_conjugate_layouts(layout, dtype):
    values = _distinct_complex((4, 6), dtype)
    if layout == "offset":
        physical = values[1:4]
    else:
        physical = values[0:1].expand(3, 6)
    inp = physical.conj()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)
    assert inp.is_conj()


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("real_dtype, scenario", _SPECIAL_CASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend complex input construction is unsupported",
)
def test_view_as_real_copy_special_values(real_dtype, scenario):
    # The shared generator supplies the payload shape (one value per scenario
    # entry), so no extra shape parameter is advertised here.
    inp = _make_special_complex(real_dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("layout", _NON_CONTIGUOUS_LAYOUTS)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_non_contiguous(layout, dtype):
    # Reading the input's storage as if it were packed would be wrong for these
    # layouts, so the candidate has to copy element-wise.
    values = _distinct_complex((10, 16, 32), dtype)
    if layout == "transpose":
        inp = values.transpose(-1, -2)
    elif layout == "offset":
        inp = values[2:8, 1:9]
    else:
        inp = values[..., ::2]
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _DEFAULT_ONLY_DTYPES
    ],
)
def test_view_as_real_copy_result_is_independent_storage(shape, dtype):
    # A write on either side must not reach the other one.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    if flag_gems.vendor_name == "mthreads":
        components = torch.view_as_real(inp)
        components[..., 0].fill_(7)
        components[..., 1].zero_()
        expected = torch.empty_like(ref_inp, device="cpu")
        expected_components = torch.view_as_real(expected)
        expected_components[..., 0].fill_(7)
        expected_components[..., 1].zero_()
    else:
        inp.fill_(7)
        expected = torch.full_like(ref_inp, 7)
    _assert_equal(res_out, ref_out)

    res_out.fill_(0)
    _assert_equal(inp, expected)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_ASCEND_COMPLEX_SKIP if dtype.is_complex else ())
        for dtype in _COMPLEX_DTYPES
    ],
)
def test_view_as_real_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(_COMPLEX_TO_REAL[dtype], shape + (2,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.view_as_real_copy(inp)
    _assert_equal(res_out, ref_out)
    _assert_equal(inp, ref_inp)

    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]

    _assert_equal(res_in_grad, ref_in_grad)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_view_as_real_copy_rejects_unsupported_dtypes(dtype):
    # Values are irrelevant to dtype rejection; avoid unsupported FP8 RNG casts.
    inp = torch.empty((4, 5), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.view_as_real_copy(inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("out_dtype", _WRONG_OUT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend complex input construction is unsupported",
)
def test_view_as_real_copy_out_rejects_wrong_dtype(out_dtype):
    inp = tu.make_input(torch.complex64, (4, 5), ["-1", "1"])
    out = torch.zeros(4, 5, 2, dtype=out_dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.view_as_real_copy(inp, out=out)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("bad_input", [3.14, [1, 2, 3]])
def test_view_as_real_copy_rejects_non_tensor(bad_input):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.view_as_real_copy(bad_input)


IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

# view_as_real_copy consumes a complex tensor and produces the real tensor one
# size down: complex32 -> float16, complex64 -> float32, complex128 -> float64.
# The tested dtypes are therefore the complex counterparts of the shared float
# dtype set: `utils.FLOAT_DTYPES` supplies the precision choices and the
# float64 entry (complex128) is appended explicitly because that shared set
# stops at float32, subject to fp64 support. Every non-complex dtype (including
# all real dtypes) is rejected, so no real dtype is ever tested.
_FLOAT_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
VIEW_AS_REAL_COMPLEX_DTYPES = [
    _FLOAT_TO_COMPLEX[d] for d in utils.FLOAT_DTYPES if d in _FLOAT_TO_COMPLEX
]
if not VIEW_AS_REAL_COMPLEX_DTYPES:
    # QUICK_MODE keeps only bfloat16 in the shared set, which has no complex
    # form; fall back to the two universally supported complex types.
    VIEW_AS_REAL_COMPLEX_DTYPES = [torch.complex32, torch.complex64]
if not cfg.QUICK_MODE and utils.fp64_is_supported:
    VIEW_AS_REAL_COMPLEX_DTYPES.append(torch.complex128)

if cfg.QUICK_MODE:
    # (input_shape,) tuples of complex tensors.
    VIEW_AS_REAL_SHAPES = [
        (4,),
        (2, 5),
    ]
else:
    # (input_shape,) tuples; covers 0-dim (-> real (2,)), 1-D/2-D/3-D/4-D
    # inputs. Empty and 0-dim inputs are covered by the dedicated tests
    # below; non-contiguous inputs that need the strided kernel's stride
    # slots by test_accuracy_view_as_real_copy_high_rank_noncontiguous.
    VIEW_AS_REAL_SHAPES = [
        (8,),
        (4, 6),
        (10, 8),
        (2, 5, 7),
        (2, 3, 4),
        (2, 2, 2, 3),
        (1,),
    ]


def _assert_matches(res, ref, inp):
    # dtype -- the reinterpret must produce the mapping complex32/64/128 ->
    # float16/32/64; pinned explicitly so a wrong out dtype fails here first.
    assert res.dtype == ref.dtype, f"dtype {res.dtype} != {ref.dtype}"
    # shape -- self.shape + (2,); a wrong rank/extent is a wrong reinterpret
    # even if values match.
    assert tuple(res.shape) == tuple(ref.shape)
    # is_contiguous -- the output must be a fresh contiguous tensor like aten.
    assert res.is_contiguous()
    # _is_alias_of -- the copy must never share storage with the input;
    # gems_assert_equal below compares values only and would pass for an
    # aliased view, so this is not redundant with it.
    assert not torch._C._is_alias_of(res, inp)
    # values -- full elementwise equality against the reference.
    _assert_equal(res, ref)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("input_shape", VIEW_AS_REAL_SHAPES)
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy(input_shape, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_dtype_mapping(dtype):
    inp = torch.randn(6, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)

    expected = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[dtype]
    assert res_out.dtype == expected
    assert res_out.shape == (6, 2)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_values(dtype):
    # The reinterpret must be bit-exact: last dim index 0 = real part, index 1
    # = imag part.
    inp = torch.randn(5, 4, dtype=dtype, device=flag_gems.device)
    res = flag_gems.view_as_real_copy(inp)
    assert torch.equal(res[..., 0].to(torch.float32), inp.real.to(torch.float32))
    assert torch.equal(res[..., 1].to(torch.float32), inp.imag.to(torch.float32))


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_noncontiguous(dtype):
    # Every stride is legal for view_as_real; the strided gather path must
    # honor the real-view strides.
    pool = torch.randn(8, 6, dtype=dtype, device=flag_gems.device)
    inp = pool[::2]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_transposed(dtype):
    base = torch.randn(3, 5, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_high_rank_noncontiguous(dtype):
    # 3-D/4-D/5-D non-contiguous inputs: only non-contiguous inputs reach
    # _view_as_real_copy_strided_kernel, whose leading-dim stride slots are
    # the rank-dependent code path. Every VIEW_AS_REAL_SHAPES entry is
    # contiguous and takes the flat, rank-independent contig kernel, so this
    # is the only coverage of the strided decomposition for rank > 2 (the
    # highest rank accepted is 5, i.e. rank == _MAX_LEAD_RANK).
    base3 = torch.randn(6, 4, 8, dtype=dtype, device=flag_gems.device)
    base4 = torch.randn(4, 8, 6, 4, dtype=dtype, device=flag_gems.device)
    base5 = torch.randn(4, 6, 8, 6, 4, dtype=dtype, device=flag_gems.device)
    inputs = [
        base3[:, ::2, :],  # 3-D stride-2
        base4.transpose(1, 2),  # 4-D transposed
        base4[:, ::2, :, :],  # 4-D strided
        base5.transpose(1, 2),  # 5-D transposed (highest accepted rank)
        base5[:, ::2, :, ::2, :],  # 5-D strided
    ]
    for inp in inputs:
        assert not inp.is_contiguous()
        ref_inp = utils.to_reference(inp)
        ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
        res_out = flag_gems.view_as_real_copy(inp)
        _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_expanded(dtype):
    # Zero-stride batch dims must be materialized into concrete copies.
    base = torch.randn(1, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_storage_offset(dtype):
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[4:12]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_1d_strided(dtype):
    # A 1-D strided complex view has complex stride != 1; the real view then
    # exposes a non-contiguous layout the kernel must decompose.
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[::2]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_independent_storage(dtype):
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)
    ref = utils.to_reference(res_out.clone())

    if IS_KUNLUNXIN or flag_gems.vendor_name == "mthreads":
        inp.real.add_(1.0)
    else:
        inp.add_(1.0)
    # The copy must not change when the source is mutated in place.
    _assert_equal(res_out, ref)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_conj(dtype):
    # A lazy conjugate input must be resolved before the reinterpret.
    inp = torch.randn(4, 3, dtype=dtype, device=flag_gems.device)
    conj_inp = inp.conj()
    ref_inp = utils.to_reference(conj_inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(conj_inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
def test_accuracy_view_as_real_copy_empty():
    inp = torch.randn(0, dtype=torch.complex64, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)
    assert res_out.shape == (0, 2)
    assert res_out.dtype == torch.float32

    inp2 = torch.randn(3, 0, dtype=torch.complex64, device=flag_gems.device)
    res2 = flag_gems.view_as_real_copy(inp2)
    assert res2.shape == (3, 0, 2)
    ref2 = torch.ops.aten.view_as_real_copy(utils.to_reference(inp2))
    _assert_matches(res2, ref2, inp2)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
def test_accuracy_view_as_real_copy_zero_dim():
    inp = torch.randn((), dtype=torch.complex64, device=flag_gems.device)
    res = flag_gems.view_as_real_copy(inp)
    assert res.shape == (2,)
    ref = torch.ops.aten.view_as_real_copy(utils.to_reference(inp))
    _assert_matches(res, ref, inp)


@pytest.mark.view_as_real_copy
@_ASCEND_COMPLEX_SKIP
def test_accuracy_view_as_real_copy_error_paths():
    dev = flag_gems.device

    aten_error = (RuntimeError, ValueError) if IS_KUNLUNXIN else RuntimeError

    # Non-complex dtypes are rejected, matching PyTorch.
    for bad_dtype in (torch.float32, torch.float16, torch.int32, torch.bool):
        with pytest.raises(aten_error):
            torch.ops.aten.view_as_real_copy(
                torch.zeros(4, dtype=bad_dtype, device=dev)
            )
        with pytest.raises(aten_error):
            flag_gems.view_as_real_copy(torch.zeros(4, dtype=bad_dtype, device=dev))

    # rank > _MAX_LEAD_RANK (5): a 6-D input. The guard lives on the
    # non-contiguous branch, so use a non-contiguous 6-D input; 5-D is the
    # highest accepted rank (covered by the high-rank strided test).
    rank6 = torch.randn(*([2] * 6), dtype=torch.complex64, device=dev).transpose(1, 2)
    assert not rank6.is_contiguous()
    with pytest.raises(RuntimeError, match="at most 5 dimensions"):
        flag_gems.view_as_real_copy(rank6)
