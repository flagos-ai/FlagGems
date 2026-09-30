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


"""Correctness tests for aten::cudnn_is_acceptable.

cudnn_is_acceptable(Tensor self) -> bool is a host-side predicate, not a
numerical kernel: it answers whether cuDNN may be used for ``self`` (tensor on
the accelerator, cuDNN enabled and compiled in, dtype in {float16, float32,
float64}, numel != 0).  The candidate returns a Python bool, so results are
compared by type and value instead of through a tensor assertion.

Spec dimensions that do not apply to this schema, and the mechanism:
  * broadcast and tensor-vs-scalar: the schema takes exactly one tensor and no
    scalar operand, so there is no operand pair to broadcast.
  * backward: the result is a Python bool with no autograd history, and the
    operand is never differentiated (no numerical kernel to differentiate).
  * operator parameters: the schema has only ``self``, so there is no default
    to omit and no bool/int/float/nan boundary sweep.
  * ``.out``: ``default`` is the only registered overload.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu


def _grid_dtypes():
    """Dtypes for the value-range grid.

    Allocation capability comes from the device's static capability flags; a
    per-dtype runtime probe would conflate input construction with whether the
    operator supports the dtype.
    """
    device = flag_gems.runtime.device
    dtypes = [
        torch.float16,
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.bool,
    ]
    if device.support_fp64:
        dtypes.append(torch.float64)
    if device.support_bf16:
        dtypes.append(torch.bfloat16)
    if device.support_int64:
        dtypes.append(torch.int64)
    if device.support_fp8:
        dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return dtypes


_GRID_DTYPES = _grid_dtypes()
_FLOAT_DTYPES = [dtype for dtype in _GRID_DTYPES if dtype.is_floating_point]

# numel == 0 is rejected for every dtype.  These rows stay small, so quick mode
# keeps all of them rather than thinning dtype or shape coverage.
_EMPTY_CASES = [
    (dtype, shape) for dtype in _GRID_DTYPES for shape in ((0,), (2, 0, 3), (0, 0))
]

# NaN / Inf payloads are never inspected by the predicate; these rows record
# that floating special values do not change the answer.
_SPECIAL_CASES = tu.special_value_cases(_FLOAT_DTYPES)


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_cudnn_is_acceptable(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
    version = inp._version
    res_out = flag_gems.cudnn_is_acceptable(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp._version == version


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("dtype,shape", _EMPTY_CASES)
def test_cudnn_is_acceptable_empty(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
    res_out = flag_gems.cudnn_is_acceptable(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("dtype,scenario", tu.selected_cases(_SPECIAL_CASES, quick=[]))
def test_cudnn_is_acceptable_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
    res_out = flag_gems.cudnn_is_acceptable(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("layout", ["transposed", "strided_slice"])
@pytest.mark.parametrize("dtype", _GRID_DTYPES)
def test_cudnn_is_acceptable_non_contiguous(dtype, layout):
    # The predicate reads device, dtype and numel only; a transposed or strided
    # view must not change the answer. The oracle stays on the same device
    # even with --ref cpu, because moving devices changes this predicate.
    if layout == "transposed":
        inp = tu.make_input(dtype, (16, 8), ["-1", "1"]).t()
    else:
        inp = tu.make_input(dtype, (8, 16, 4), ["-1", "1"])[:, ::2, :]
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
    res_out = flag_gems.cudnn_is_acceptable(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("meta")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_cudnn_is_acceptable_non_accelerator_inputs(device, dtype):
    # device_check: NoCheck makes a CPU or meta tensor a valid operand, and the
    # predicate must reject it even for the dtypes it accepts on the
    # accelerator.  Both sides receive an identical same-device tensor, so no
    # transfer to the reference device is involved.
    ref_inp = torch.empty((4, 8), dtype=dtype, device=device)
    res_inp = torch.empty((4, 8), dtype=dtype, device=device)

    ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
    res_out = flag_gems.cudnn_is_acceptable(res_inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("enabled", [True, False])
def test_cudnn_is_acceptable_respects_cudnn_flag(enabled):
    # userEnabledCuDNN() is the predicate's first gate: with cuDNN disabled an
    # otherwise acceptable tensor must come back rejected.  The flags() context
    # manager restores the previous global state on exit.
    inp = tu.make_input(torch.float32, (8, 8), ["-1", "1"])
    ref_inp = inp.clone()

    with torch.backends.cudnn.flags(enabled=enabled):
        ref_out = torch.ops.aten.cudnn_is_acceptable(ref_inp)
        res_out = flag_gems.cudnn_is_acceptable(inp)
        assert type(res_out) is bool
        assert res_out == ref_out


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("bad_input", [[1, 2, 3], 3, 3.5])
def test_cudnn_is_acceptable_rejects_non_tensor(bad_input):
    # The native operator raises for arguments that are not tensors.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_is_acceptable(bad_input)


@pytest.mark.cudnn_is_acceptable
def test_cudnn_is_acceptable_requires_operand():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cudnn_is_acceptable()
