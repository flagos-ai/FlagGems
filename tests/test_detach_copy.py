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
from .conftest import QUICK_MODE

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.detach_copy


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name not in ("ascend", "kunlunxin") or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_input(dtype, shape, value_range)
    # Encode FP8 on CPU; device-side FP32-to-FP8 casts are unavailable.
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    if low == high:
        result = torch.full(shape, low, dtype=dtype, device="cpu")
    else:
        result = torch.testing.make_tensor(
            shape, dtype=dtype, device="cpu", low=low, high=high
        )
    return result.to(flag_gems.device)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name not in ("ascend", "kunlunxin") or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_special_input(dtype, scenario)
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return (
        torch.tensor(payloads[scenario], device="cpu", dtype=torch.float32)
        .to(dtype)
        .to(flag_gems.device)
    )


def _make_out_buffer(shape, dtype, device):
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and (
        torch.device(device).type == "cpu"
        or flag_gems.vendor_name in ("ascend", "kunlunxin")
    ):
        # Older CPU builds have FP8 casts but no full/fill kernel.
        return (
            torch.full(shape, 7, dtype=torch.float32, device="cpu").to(dtype).to(device)
        )
    creation_device = device
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype.is_complex:
        # Device full kernels do not cover complex payloads.
        creation_device = "cpu"
    return torch.full(shape, 7, dtype=dtype, device=creation_device).to(device)


def _comparison_to_cpu(tensor):
    fp8 = tensor.dtype in tu._FP8_DTYPES
    if tensor.layout == torch.strided and (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and tensor.device.type != "cpu"
        or (fp8 and tensor.is_neg())
    ):
        values = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            tensor.untyped_storage().cpu(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        if fp8:
            values = values.float()
        if tensor.is_conj():
            values = values.conj()
        if tensor.is_neg():
            values = torch._neg_view(values)
        return values
    values = tensor.cpu()
    return values.float() if fp8 else values


def _assert_result_equal(result, reference):
    # Validate the original device and metadata before CPU value observation.
    device = torch.device(flag_gems.device)
    if device.index is None and device.type != "cpu":
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    assert result.device == device
    assert result.dtype == reference.dtype
    assert result.layout == reference.layout
    assert result.shape == reference.shape
    if utils.TO_CPU:
        assert reference.device == torch.device("cpu")
    else:
        assert result.device == reference.device
    if result.dtype in tu._FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex()
    ):
        result, reference = _comparison_to_cpu(result), _comparison_to_cpu(reference)
    tu.assert_result_equal(result, reference)


# Copy values into independent contiguous storage; backward is unsupported.
_DETACH_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
        torch.complex64,
    ]
)

_DETACH_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_DETACH_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_DETACH_COPY_NO_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]
_DETACH_COPY_STORAGE_SHAPES = [(16, 32), (64, 128)]


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    _assert_result_equal(res_out, ref_out)
    # The input must be untouched by the copy.
    _assert_result_equal(inp, ref_inp)


_DETACH_COPY_INPUT_DTYPE_CASES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _DETACH_COPY_DTYPES
]


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _DETACH_COPY_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8 or value_range in (["-1", "1"], ["0", "max"])
]


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_detach_copy_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_detach_copy_special_values(dtype):
    inp = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_result_equal(res_out, ref_out)
    # -0.0 must copy with its sign bit intact (equal_nan-tolerant compares treat
    # -0.0 == 0.0, so pin the sign explicitly).
    # Read bits on CPU: Ascend signbit does not distinguish -0.0 from +0.0.
    assert res_out.device == inp.device
    assert torch.equal(torch.signbit(res_out.cpu()), torch.signbit(ref_out.cpu()))


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy_out(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = _make_out_buffer(shape, ref_inp.dtype, ref_inp.device)
    res_out = _make_out_buffer(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_detach_copy_out_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = _make_out_buffer(shape, ref_inp.dtype, ref_inp.device)
    res_out = _make_out_buffer(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy_non_contiguous(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        pytest.skip("Kunlunxin detach_copy cannot copy noncontiguous FP8 tensors")
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_empty(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_STORAGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_result_equal(res_out, ref_out)
    res_out.fill_(3.25)
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    _assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NO_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES, quick=utils.FLOAT_DTYPES)
)
def test_detach_copy_no_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.detach_copy(inp)
    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(res_out, inp, grad_outputs=grad)


@pytest.mark.detach_copy
def test_detach_copy_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(3.14)


@pytest.mark.detach_copy_out
def test_detach_copy_out_rejects_wrong_dtype():
    inp = _make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)
    res_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy.out(ref_inp, out=ref_out_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(inp, out=res_out_bad)


@pytest.mark.detach_copy
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype, scenario in tu.selected_cases(
            tu.special_value_cases(_DETACH_COPY_DTYPES)
        )
    ],
)
def test_detach_copy_special_scenarios(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.detach_copy(reference)
    actual = flag_gems.detach_copy(inp)
    _assert_result_equal(actual, expected)


_MASTER_SHAPES = [(16,), (1024,), (2, 3, 4), (16, 128, 64, 60)]
if QUICK_MODE:
    _MASTER_SHAPES = [(2, 3)]


def _make_master_input(shape, dtype):
    # Complex dtypes must be generated explicitly: without this branch they
    # fall through to BOOL below and never exercise the complex copy path.
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(flag_gems.device)
    # BOOL_TYPES
    return (
        torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
            flag_gems.device
        )
        if shape
        else torch.empty(shape, dtype=torch.bool, device=flag_gems.device)
    )


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _MASTER_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.FLOAT_DTYPES
    + utils.INT_DTYPES
    + utils.BOOL_TYPES
    + [torch.complex64, torch.complex128],
)
def test_accuracy_detach_copy(shape, dtype):
    inp = _make_master_input(shape, dtype)
    # Guards the generation branch above: a silent BOOL fallthrough would
    # make this parametrization pass without testing the requested dtype.
    assert inp.dtype == dtype
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == inp.shape
    assert res_out.dtype == inp.dtype
    # A copy, not a view: fresh storage.
    assert res_out.data_ptr() != inp.data_ptr()
    # Native contract: freshly allocated output is contiguous and dense.
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", [(2, 3), (4, 8, 5)])
def test_accuracy_detach_copy_non_contiguous(shape):
    # Transposed (non-contiguous) input: the strided kernel must gather
    # elements by shape/stride metadata and materialize a contiguous output,
    # matching native detach_copy.
    inp = torch.randn(shape, device=flag_gems.device).transpose(0, -1)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == inp.shape
    assert res_out.data_ptr() != inp.data_ptr()
    # Native materializes the copy contiguously for a transposed input.
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()


@pytest.mark.detach_copy
def test_accuracy_detach_copy_empty():
    # Empty 1-D tensor: grid would be (0,), short-circuited before launch.
    inp = torch.empty(0, device=flag_gems.device)
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    assert res_out.shape == (0,)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)
    # Native allocates fresh storage even for empty inputs, but an empty CUDA
    # tensor has no backing page, so both data_ptrs are 0 and the fresh-storage
    # property is not observable here (unlike non-empty cases below).


@pytest.mark.detach_copy
def test_accuracy_detach_copy_empty_non_contiguous():
    # A transposed empty tensor is "contiguous" only by ATen's degenerate
    # stride rules; exercise the strided branch's empty guard too.
    inp = torch.empty(3, 0, device=flag_gems.device).transpose(0, -1)
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    assert res_out.shape == inp.shape
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.detach_copy
def test_accuracy_detach_copy_zero_dim():
    inp = torch.randn((), device=flag_gems.device)
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    assert res_out.shape == ()
    assert res_out.numel() == 1
    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.detach_copy
def test_accuracy_detach_copy_single_element():
    inp = torch.randn(1, device=flag_gems.device)
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.detach_copy
def test_accuracy_detach_copy_autograd_contract():
    # detach_copy is a copy op: the output must never alias the input and
    # must carry fresh (grad-free) storage. Native wraps the CUDA kernel in
    # a generated autograd kernel (requires_grad propagates from the input
    # and backward raises "derivative ... not implemented"); the FlagGems
    # registration is backend-side, so the output is a plain leaf. Both
    # sides agree that the result is detached from the graph: no gradient
    # can flow from the output back into the input storage.
    inp = torch.randn(4, 5, device=flag_gems.device, requires_grad=True)
    ref_inp = inp.detach().requires_grad_(True)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    assert res_out.data_ptr() != inp.data_ptr()
    assert ref_out.data_ptr() != ref_inp.data_ptr()
    # Native: output is not a view of the input (view_assoc is unset).
    assert ref_out._base is None
    # ref_out must stay on CUDA: the native autograd wrapper (grad_fn=
    # <NotImplemented>) only exists for the CUDA native op, so ref never
    # goes through to_reference. Compare device-explicitly instead of via
    # gems_assert_equal, which requires a CPU ref under --ref=cpu.
    assert torch.equal(res_out.detach().cpu(), ref_out.detach().to("cpu"))
    # Values equal the input's current content.
    assert torch.equal(res_out.detach(), inp.detach())


@pytest.mark.detach_copy
def test_accuracy_detach_copy_dispatch_stability():
    # Repeated dispatches plus an unrelated aten op in between must not
    # corrupt the copy (no shared mutable state across calls).
    inp = torch.randn(3, 7, device=flag_gems.device)
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    for _ in range(3):
        res_out = flag_gems.detach_copy(inp)
        assert res_out.device == inp.device
        assert torch.equal(utils.to_cpu(res_out, ref_out), ref_out)
        assert torch.ops.aten.dim(inp) == 2
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.detach_copy
@pytest.mark.parametrize("layout", ["vector", "transposed", "scalar", "permuted"])
def test_accuracy_detach_copy_neg_view(layout):
    shapes = {
        "vector": (1024,),
        "transposed": (4, 8),
        "scalar": (),
        "permuted": (16, 128, 64, 60),
    }
    values = torch.randn(shapes[layout], device=flag_gems.device)
    if layout == "transposed":
        values = values.transpose(0, -1)
    elif layout == "permuted":
        values = values.permute(2, 0, 3, 1)
    inp = torch._neg_view(values)
    assert inp.is_neg()
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(inp))
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.dtype == inp.dtype
    # res_out must equal the resolved logical values, and must NOT equal the
    # raw storage of the view's base (which is what a bitwise copy yields).
    assert torch.equal(res_out, inp.resolve_neg())
    assert not torch.equal(res_out, inp.resolve_neg().neg())
    assert res_out.data_ptr() != inp.data_ptr()
    assert not res_out.is_neg()
    # Freshly allocated output is contiguous, like the native operator.
    assert res_out.is_contiguous() == ref_out.is_contiguous()


@pytest.mark.detach_copy
def test_accuracy_detach_copy_neg_view_complex():
    # The negative bit also rides on complex views; combined neg+conj views
    # resolve to -conj(x), which must not be confused with raw storage.
    inp = torch.randn(3, 4, dtype=torch.complex64, device=flag_gems.device)
    neg_conj = torch._neg_view(inp).conj()
    assert neg_conj.is_neg() and neg_conj.is_conj()
    ref_out = torch.ops.aten.detach_copy(utils.to_reference(neg_conj))
    res_out = flag_gems.detach_copy(neg_conj)
    assert res_out.device == neg_conj.device

    utils.gems_assert_equal(res_out, ref_out)
    assert torch.allclose(res_out, (-inp).conj())
    assert res_out.data_ptr() != inp.data_ptr()
    assert not res_out.is_neg() and not res_out.is_conj()


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["dense", "transposed", "scalar", "empty"])
def test_accuracy_detach_copy_complex_layouts(layout, dtype):
    if layout == "empty":
        inp = torch.empty(4, 0, dtype=dtype, device=flag_gems.device).t()
    else:
        shape = () if layout == "scalar" else (4, 6)
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if layout == "transposed":
            inp = inp.t()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)
    assert res_out.device == inp.device

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.dtype == inp.dtype
    assert res_out.shape == inp.shape
    if inp.numel() > 0:
        # Empty CUDA tensors have no backing page (data_ptr() == 0 on both
        # sides), so the fresh-storage property is only observable here.
        assert res_out.data_ptr() != inp.data_ptr()
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    assert not res_out.is_conj() and not res_out.is_neg()
