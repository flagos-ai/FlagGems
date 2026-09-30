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

"""Correctness tests for ``aten::get_device(Tensor self) -> int``.

The operator returns the placement index recorded in the operand's host-side
metadata (the accelerator index, ``-1`` when the placement carries none). It
reads no element data, so dtype, shape, contents and strides cannot change the
result: the checks below compare the returned Python ``int`` and verify the
operand is left untouched instead of applying a value-range grid. Broadcast,
scalar-operand and backward workloads do not apply to a schema with one Tensor
argument returning a Python int.

The oracle is the native operator applied to the tested operand itself.
``tu.to_reference`` may relocate a reference copy (``--ref cpu``) and would then
report the copy's placement, and a pure metadata read cannot perturb the operand
it is handed.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

_MAIN_RANGE = ["-1", "1"]

# bool and float64 additionally accept this metadata read; float64 is gated on
# the backend flag used throughout the suite.
_DTYPES = tu.REQUIRED_DTYPES + [torch.bool]
if utils.fp64_is_supported:
    _DTYPES = _DTYPES + [torch.float64]

_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Value ranges cannot change a metadata read, so two representative shapes
# carry the required five-range sweep (the shape level still shrinks to the
# quick shape).
_RANGE_SHAPES = tu.selected_cases(
    [(20, 320, 15), (16, 128, 64, 60)], quick=[(2, 19, 7)]
)

# Layouts whose placement must be read without materializing the tensor:
# non-contiguous views (transposed / sliced with nonzero offset), an
# overlapping expanded view, an empty tensor, a sparse COO tensor (no
# storage), and a leaf that requires grad.
_LAYOUTS = ["transposed", "sliced", "expanded", "empty", "sparse", "grad"]

# Placements that carry no device index: the schema reports -1 for both.
_NO_INDEX_PLACEMENTS = ["cpu", "meta"]

_KIND = torch.device(flag_gems.device).type


def _device_indices():
    """Explicit placement indices of the devices this backend exposes."""
    if _KIND == "cpu":
        return [None]
    return list(range(flag_gems.runtime.device.device_count))


_TARGET_INDICES = _device_indices()


def _layout_input(layout, dtype):
    if layout == "transposed":
        return tu.make_input(dtype, (4, 10), _MAIN_RANGE).t()
    if layout == "sliced":
        return tu.make_input(dtype, (4, 10), _MAIN_RANGE)[1:3, ::3]
    if layout == "expanded":
        return tu.make_input(dtype, (1, 10), _MAIN_RANGE).expand(4, 10)
    if layout == "empty":
        return tu.make_input(dtype, (0, 3), _MAIN_RANGE)
    if layout == "sparse":
        indices = torch.tensor([[0, 2, 1], [1, 1, 3]], device=flag_gems.device)
        values = tu.make_input(dtype, (3,), _MAIN_RANGE)
        return torch.sparse_coo_tensor(indices, values, (4, 4))
    if layout == "grad":
        inp = tu.make_input(dtype, (4, 10), _MAIN_RANGE)
        # Only floating dtypes can carry a grad; torch.dtype.is_floating_point
        # is a bool, unlike the bound method Tensor.is_floating_point.
        return inp.requires_grad_(inp.dtype.is_floating_point)
    raise ValueError("unknown layout: %s" % layout)


def _assert_device_index(res_index, ref_index, inp):
    """The schema returns a Python int, so compare ints directly."""
    assert type(res_index) is int, type(res_index)
    assert res_index == ref_index, (res_index, ref_index)
    if inp.device.index is not None:
        assert res_index == inp.device.index, (res_index, inp.device)


@pytest.mark.get_device
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_get_device(shape, dtype):
    inp = tu.make_input(dtype, shape, _MAIN_RANGE)

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    _assert_device_index(res_index, ref_index, inp)


@pytest.mark.get_device
@pytest.mark.parametrize("shape", _RANGE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_get_device_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    _assert_device_index(res_index, ref_index, inp)


@pytest.mark.get_device
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_get_device_layouts(layout, dtype):
    inp = _layout_input(layout, dtype)
    snapshot = (
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.dtype,
        inp.device,
    )
    # Sparse tensors have no data pointer; the remaining metadata still applies.
    ptr = inp.data_ptr() if inp.layout == torch.strided else None

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    _assert_device_index(res_index, ref_index, inp)
    # Reading the placement must not reinterpret, move or reallocate the operand.
    now = (
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.dtype,
        inp.device,
    )
    assert now == snapshot
    if ptr is not None:
        assert inp.data_ptr() == ptr


@pytest.mark.get_device
@pytest.mark.parametrize("index", _TARGET_INDICES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_get_device_reports_operand_index(index, dtype):
    # Only the placement matters here, so the allocation is never read.
    placement = torch.device(_KIND) if index is None else torch.device(_KIND, index)
    inp = torch.empty((2, 3), dtype=dtype, device=placement)

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    _assert_device_index(res_index, ref_index, inp)


@pytest.mark.get_device
@pytest.mark.parametrize("placement", _NO_INDEX_PLACEMENTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_get_device_placement_without_index(placement, dtype):
    # A CPU or meta operand carries no device index in its host-side metadata,
    # so the schema reports -1 for either placement.
    inp = torch.empty((2, 3), dtype=dtype, device=placement)

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    assert res_index == -1
    _assert_device_index(res_index, ref_index, inp)


@pytest.mark.get_device
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])
)
def test_get_device_special_values(dtype, scenario):
    # Element data is never read, so NaN/Inf operands must be reported exactly
    # like finite ones (default mode; e4m3fn only represents NaN).
    inp = tu.make_special_input(dtype, scenario)

    ref_index = torch.ops.aten.get_device(inp)
    res_index = flag_gems.get_device(inp)

    _assert_device_index(res_index, ref_index, inp)


@pytest.mark.get_device
@pytest.mark.parametrize(
    "bad_argument", [3.14, 7, "not-a-device", [0.0, 1.0], (1, 2), None]
)
def test_get_device_rejects_non_tensor(bad_argument):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.get_device(bad_argument)


@pytest.mark.get_device
def test_get_device_rejects_missing_argument():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.get_device()


@pytest.mark.get_device
def test_get_device_rejects_extra_argument():
    inp = tu.make_input(torch.float32, (2, 3), _MAIN_RANGE)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.get_device(inp, inp)
