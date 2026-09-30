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

from . import test_utils as tu

# aten::_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None,
#     ScalarType? dtype=None, *, Device? device=None, Layout? layout=None) -> ()
#
# A debug metadata validator: it reads size / stride / dtype / device / layout and
# never the elements, so its only observable results are None and a
# "Tensor <kind> mismatch!" RuntimeError naming the first mismatching argument in the
# order size -> stride -> dtype -> device -> layout. There is no tensor output, so
# metadata is checked with tuple/property equality instead of the numeric
# tu.assert_result_* helpers.
#
# Not applicable: broadcast and tensor/scalar forms (one tensor operand, no scalar
# operand) and backward (nothing is returned and nothing is differentiable). The five
# value ranges are still swept, to record that every representable payload passes an
# unchanged-metadata check.

_CAPS = flag_gems.runtime.device
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# The spec's nine dtypes (the FP8 pair only where the backend can build fp8 tensors)
# plus the extra types this backend represents. float64/complex are gated on the fp64
# capability; every dtype here was probed native-valid on the active device.
_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    if dtype not in _FP8_DTYPES or _CAPS.support_fp8
]
if _CAPS.support_fp64:
    _DTYPES += [torch.float64, torch.complex128]
_DTYPES += [torch.bool]

_DTYPES += [torch.complex64]
_DTYPES = [
    dtype
    for dtype in _DTYPES
    if (dtype != torch.bfloat16 or _CAPS.support_bf16)
    and (dtype != torch.int64 or _CAPS.support_int64)
]

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# "meta" is never the device of a real input, so it mismatches on every backend; a
# literal "cpu" would match when the backend itself runs on the CPU.
_WRONG_DEVICE = torch.device("meta")
# Every input below is strided, so the sparse layout always mismatches it.
_WRONG_LAYOUT = torch.sparse_coo

# Shapes of the non-grid workloads: quick keeps the spec smoke shape, the default
# suite adds a 0-D, 1-D, 3-D and 5-D metadata vector. The grid test below already
# covers all seven spec shapes.
_METADATA_SHAPES = tu.selected_cases(
    [(), (256,), (20, 320, 15), (16, 7, 57, 32, 29)],
    quick=[(), (256,), (2, 19, 7)],
)

# Argument parsing does not depend on the tensor, so one explicit shape is enough.
_PLAIN_SHAPE = (256,)


def _metadata_of(inp):
    """Snapshot of the allocation metadata the operator has to leave untouched."""
    if inp.layout != torch.strided:
        # Sparse tensors expose neither a storage handle nor a storage offset.
        return (
            tuple(inp.size()),
            tuple(inp.stride()),
            None,
            inp.dtype,
            inp.device,
            inp.layout,
            None,
        )
    return (
        tuple(inp.size()),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.dtype,
        inp.device,
        inp.layout,
        inp.untyped_storage().data_ptr(),
    )


def _native_kwargs(inp):
    """The tensor's own metadata, i.e. the arguments that have to match.

    Taken from ``inp`` itself and never from a transferred reference copy, so the
    configured ``--ref`` device cannot change what the candidate is asked to check.
    """
    return {
        "size": list(inp.size()),
        "stride": list(inp.stride()),
        "dtype": inp.dtype,
        "device": inp.device,
        "layout": inp.layout,
    }


def _mismatch_value(name, inp):
    """A value for ``name`` that must not match this tensor's own metadata."""
    if name == "size":
        size = list(inp.size())
        return size[:-1] + [size[-1] + 1] if size else [1]
    if name == "stride":
        stride = list(inp.stride())
        return [step + 1 for step in stride] if stride else [0]
    if name == "dtype":
        return torch.float32 if inp.dtype != torch.float32 else torch.float64
    if name == "device":
        return _WRONG_DEVICE
    return _WRONG_LAYOUT


def _row_kwargs(inp, row):
    """Keyword arguments for a row of per-argument tokens.

    "ok" passes the tensor's own metadata, "bad" a value that must mismatch, "short"
    a size list of the wrong length, and None omits the argument so the schema default
    applies.
    """
    kwargs = {}
    for name, token in zip(("size", "stride", "dtype", "device", "layout"), row):
        if token is None:
            continue
        if token == "ok":
            kwargs[name] = _native_kwargs(inp)[name]
        elif token == "short":
            size = list(inp.size())
            kwargs[name] = size[:-1] if size else [1]
        else:
            kwargs[name] = _mismatch_value(name, inp)
    return kwargs


def _state_tensor(label):
    """Input states whose view and metadata shape is part of what the op checks."""
    device = flag_gems.device
    if label == "noncontiguous-2d":
        base = torch.arange(36, device=device, dtype=torch.float32).reshape(6, 6)
        return base[::2, ::3]
    if label == "storage-offset-1d":
        return torch.arange(10, device=device, dtype=torch.float32)[2:8]
    if label == "expanded-stride-0":
        return torch.zeros(1, 5, device=device).expand(4, 5)
    if label == "conjugate-complex":
        return torch.zeros(3, 3, device=device, dtype=torch.complex64).conj()
    if label == "empty-2d":
        return torch.zeros(0, 3, device=device)
    if label == "zero-dim":
        return torch.zeros((), device=device)
    if label == "single-element":
        return torch.zeros(1, device=device)
    if label == "bool-payload":
        return torch.zeros(2, 3, device=device, dtype=torch.bool)
    if label == "shared-storage-transpose":
        base = torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6)
        return base.narrow(1, 1, 4).transpose(0, 1)
    if label == "sparse-coo":
        indices = torch.zeros(2, 3, device=device, dtype=torch.int64)
        values = torch.zeros(3, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 5), device=device)
    raise AssertionError(f"unknown state label: {label}")


# One workload per state. They are as cheap as the shape cases, so quick mode keeps
# all of them.
_STATE_ROWS = [
    "noncontiguous-2d",
    "storage-offset-1d",
    "expanded-stride-0",
    "conjugate-complex",
    "empty-2d",
    "zero-dim",
    "single-element",
    "bool-payload",
    "shared-storage-transpose",
    "sparse-coo",
]

_ALL_OK = ("ok", "ok", "ok", "ok", "ok")

# Argument omission / pass combinations. Every row is cheap, including the negative
# ones, so all of them run in quick mode too.
_CALL_ROWS = [
    (None, None, None, None, None),
    ("ok", None, None, None, None),
    (None, "ok", None, None, None),
    (None, None, "ok", None, None),
    (None, None, None, "ok", None),
    (None, None, None, None, "ok"),
    ("ok", "ok", None, None, None),
    ("ok", "ok", "ok", "ok", None),
    ("ok", "ok", None, None, "ok"),
    _ALL_OK,
]

# The expected messages are the native ones; the trailing rows pin the
# size -> stride -> dtype -> device check order.
_MISMATCH_ROWS = [
    ("Tensor sizes mismatch!", ("bad", "ok", None, None, None)),
    ("Tensor sizes mismatch!", ("bad", None, None, None, None)),
    ("Tensor sizes mismatch!", ("short", None, None, None, None)),
    ("Tensor strides mismatch!", ("ok", "bad", None, None, None)),
    ("Tensor dtype mismatch!", ("ok", "ok", "bad", None, None)),
    ("Tensor device mismatch!", ("ok", "ok", "ok", "bad", None)),
    ("Tensor layout mismatch!", ("ok", "ok", "ok", "ok", "bad")),
    ("Tensor sizes mismatch!", ("bad", "bad", "bad", "bad", "bad")),
    ("Tensor strides mismatch!", ("ok", "bad", "bad", "bad", "bad")),
    ("Tensor dtype mismatch!", ("ok", "ok", "bad", "bad", "bad")),
    ("Tensor device mismatch!", ("ok", "ok", "ok", "bad", "bad")),
]

# Invalid argument types/values. The parser message varies between torch versions, so
# only the exception type is asserted; the native operator is not called in this test.
_INVALID_ARGUMENT_ROWS = [
    ("a", 5),
    ("a", None),
    ("size", 8),
    ("size", [1.5, 2.0]),
    ("dtype", "float32"),
    ("layout", "strided"),
    ("device", "not-a-device"),
]


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test__assert_tensor_metadata(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    kwargs = _row_kwargs(inp, _ALL_OK)
    before = _metadata_of(inp)
    values_before = tu.to_reference(inp)

    # The reference call establishes that this metadata is native-valid.
    torch.ops.aten._assert_tensor_metadata(inp, **kwargs)
    assert flag_gems._assert_tensor_metadata(inp, **kwargs) is None
    assert _metadata_of(inp) == before
    tu.assert_result_equal(inp, values_before)


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("shape", _METADATA_SHAPES)
@pytest.mark.parametrize("row", _CALL_ROWS)
def test__assert_tensor_metadata_call_form(shape, row):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    kwargs = _row_kwargs(inp, row)
    before = _metadata_of(inp)
    values_before = tu.to_reference(inp)

    torch.ops.aten._assert_tensor_metadata(inp, **kwargs)
    assert flag_gems._assert_tensor_metadata(inp, **kwargs) is None
    assert _metadata_of(inp) == before
    tu.assert_result_equal(inp, values_before)


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("label", _STATE_ROWS)
def test__assert_tensor_metadata_tensor_state(label):
    inp = _state_tensor(label)
    kwargs = _row_kwargs(inp, _ALL_OK)
    before = _metadata_of(inp)
    values_before = tu.to_reference(inp)

    torch.ops.aten._assert_tensor_metadata(inp, **kwargs)
    assert flag_gems._assert_tensor_metadata(inp, **kwargs) is None
    assert _metadata_of(inp) == before
    tu.assert_result_equal(inp, values_before)


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("shape", _METADATA_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test__assert_tensor_metadata_dtype_mismatch(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    kwargs = _row_kwargs(inp, ("ok", "ok", "bad", None, None))

    with pytest.raises(RuntimeError, match="Tensor dtype mismatch!"):
        flag_gems._assert_tensor_metadata(inp, **kwargs)


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("shape", _METADATA_SHAPES)
@pytest.mark.parametrize("expected,row", _MISMATCH_ROWS)
def test__assert_tensor_metadata_mismatch(shape, expected, row):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    kwargs = _row_kwargs(inp, row)
    before = _metadata_of(inp)
    values_before = tu.to_reference(inp)

    with pytest.raises(RuntimeError, match=expected):
        flag_gems._assert_tensor_metadata(inp, **kwargs)
    assert _metadata_of(inp) == before
    tu.assert_result_equal(inp, values_before)


@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize("arg,bad_value", _INVALID_ARGUMENT_ROWS)
def test__assert_tensor_metadata_invalid_argument(arg, bad_value):
    inp = tu.make_input(torch.float32, _PLAIN_SHAPE, ["-1", "1"])
    kwargs = _row_kwargs(inp, _ALL_OK)
    if arg == "a":
        inp = bad_value
    else:
        kwargs[arg] = bad_value

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._assert_tensor_metadata(inp, **kwargs)


# The op never reads elements, so a NaN/Inf payload is not a special value for it:
# these default-only cases record that a NaN/Inf carrier still passes metadata
# validation. Quick omits them, like every other positive special-value case.
@pytest.mark.assert_tensor_metadata
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[]),
)
def test__assert_tensor_metadata_special_payload(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    kwargs = _row_kwargs(inp, _ALL_OK)
    before = _metadata_of(inp)
    values_before = tu.to_reference(inp)

    torch.ops.aten._assert_tensor_metadata(inp, **kwargs)
    assert flag_gems._assert_tensor_metadata(inp, **kwargs) is None
    assert _metadata_of(inp) == before
    tu.assert_result_equal(inp, values_before)
