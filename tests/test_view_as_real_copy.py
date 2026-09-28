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
from . import test_utils as tu

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


def _assert_copy_contract(res_out, ref_out, inp, ref_inp, allocating=True):
    """Native values, allocating geometry, and an input left untouched.

    Base-storage pointers are compared, so a result aliasing an offset view of
    the input is rejected as well; an empty result has no storage independence
    to observe, so that check is limited to non-empty inputs.
    """
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)
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
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_view_as_real_copy(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
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
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
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
    tu.assert_result_equal(base, ref_base)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
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
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
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
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
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
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
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
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
def test_view_as_real_copy_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_copy_contract(res_out, ref_out, inp, ref_inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", _DEFAULT_ONLY_DTYPES)
def test_view_as_real_copy_result_is_independent_storage(shape, dtype):
    # A write on either side must not reach the other one.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    inp.fill_(7)
    tu.assert_result_equal(res_out, ref_out)

    res_out.fill_(0)
    tu.assert_result_equal(inp, torch.full_like(ref_inp, 7))


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_view_as_real_copy_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(_COMPLEX_TO_REAL[dtype], shape + (2,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = flag_gems.view_as_real_copy(inp)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)

    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]

    tu.assert_result_equal(res_in_grad, ref_in_grad)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_view_as_real_copy_rejects_unsupported_dtypes(dtype):
    inp = tu.make_input(dtype, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.view_as_real_copy(inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("out_dtype", _WRONG_OUT_DTYPES)
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
