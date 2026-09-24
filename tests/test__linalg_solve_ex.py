# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register the underscore-prefixed pytest marker explicitly.
setattr(
    pytest.mark,
    "_linalg_solve_ex",
    MarkDecorator(Mark("_linalg_solve_ex", (), {}, _ispytest=True), _ispytest=True),
)

# Measured with valid rank>=2 operands on the active CUDA backend: float32,
# complex64, float64 and complex128 solve natively and keep their dtype, while
# half/bfloat16/fp8 fail with 'lu_factor_cusolver not implemented for ...' and
# every integer/bool dtype fails with 'Expected a floating point or complex
# tensor as input'. Only the 64-bit types follow the backend capability flag;
# complex64 is listed unconditionally because it does not need fp64 support.
SUPPORTED_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES += [torch.float64, torch.complex128]

_COMPLEX_SUPPORTED_DTYPES = [dtype for dtype in SUPPORTED_DTYPES if dtype.is_complex]

# The operator requires A of rank >= 2, so the spec's 0-dim and 1-dim shapes are
# not representable operands (measured: 'linalg.solve: The input tensor A must
# have at least 2 dimensions.').
_GRID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _system_shapes(shape):
    """Split a spec shape into the _linalg_solve_ex operand geometry.

    A is (batch..., n, n) with n taken from the second-to-last axis, and the
    batch axes are preserved unchanged. B takes the number of right-hand-side
    columns m from the last axis; no axis is halved, capped or dropped.
    """
    return tuple(shape[:-2]), shape[-2], shape[-1]


def _rhs_shape(batch, n, m, left):
    # AX = B takes B of shape (batch..., n, m); XA = B takes (batch..., m, n).
    return batch + (n, m) if left else batch + (m, n)


def _conditioned_pair(dtype, shape, value_range, left=True):
    """Dominant-diagonal system generated from the requested value range.

    Off-diagonal entries are the sampled range values divided by 8n. The
    diagonal is rewritten as the sampled SIGN times a magnitude between 0.5 and
    0.75 of the range's largest bound, so the diagonal strictly dominates every
    off-diagonal entry and the pivoting choice is unambiguous. Taking abs() of
    the diagonal would collapse the negative-diagonal ranges onto the positive
    family, and adding magnitude to already-large positive samples would
    overflow, which is why the magnitude is bounded by the range bound itself.
    """
    batch, n, m = _system_shapes(shape)
    a = tu.make_input(dtype, batch + (n, n), value_range)
    b = tu.make_input(dtype, _rhs_shape(batch, n, m, left), value_range)
    scale = 8 * n
    high = max(
        abs(tu.resolve_bound(value_range[0], dtype)),
        abs(tu.resolve_bound(value_range[1], dtype)),
    )
    diag = torch.diagonal(a, dim1=-2, dim2=-1)
    magnitude = (diag / high).abs()
    sign = torch.sign(diag.real if dtype.is_complex else diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    conditioned = a / scale
    torch.diagonal(conditioned, dim1=-2, dim2=-1).copy_(
        sign * (high * 0.5) * (1.0 + magnitude * 0.5)
    )
    return conditioned, b / scale


def _raw_pair(dtype, shape, value_range, left=True):
    """Unconditioned operands taken straight from the requested range.

    Nothing is rescaled here, so the requested magnitudes are preserved. On the
    two unbounded ranges the native factorization degenerates (measured on
    CUDA: a nonzero info and an all-NaN result with 1-based identity pivots for
    float32/complex64 at 2-D shapes), and the LU/pivot artifacts degenerate with
    it. The whole tuple is still compared component by component; finite and
    nonfinite entries are matched with the shared equal_nan handling instead of
    the test redefining the range or masking individual components.
    """
    batch, n, m = _system_shapes(shape)
    return (
        tu.make_input(dtype, batch + (n, n), value_range),
        tu.make_input(dtype, _rhs_shape(batch, n, m, left), value_range),
    )


def _assert_outputs(res_out, ref_out):
    """Compare all four components with the semantics each one needs.

    result/LU are computed values and use the close comparison; pivots/info are
    integer metadata and must match exactly.
    """
    tu.assert_result_close(res_out[0], ref_out[0])
    tu.assert_result_close(res_out[1], ref_out[1])
    tu.assert_result_equal(res_out[2], ref_out[2])
    tu.assert_result_equal(res_out[3], ref_out[3])


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_conditioned_grid(shape, value_range, dtype):
    a, b = _conditioned_pair(dtype, shape, value_range)
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_raw_grid(shape, value_range, dtype):
    a, b = _raw_pair(dtype, shape, value_range)
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_nontrivial_pivots(dtype):
    # Scaled anti-diagonal identity: the leading column pivot is the last row,
    # so the returned pivot vector is a real permutation rather than 1..n.
    n = 8
    a = torch.flip(torch.eye(n, dtype=dtype, device=flag_gems.device), [0]) * 3
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


_FLAG_CASES = tu.selected_cases(
    [(left, check) for left in (True, False) for check in (True, False)], quick=[]
)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("left,check_errors", _FLAG_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_flag_combinations(left, check_errors, dtype):
    a, b = _conditioned_pair(dtype, (20, 320, 15), ["-1", "1"], left=left)
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(
        ref_a, ref_b, left=left, check_errors=check_errors
    )
    res_out = flag_gems._linalg_solve_ex(a, b, left=left, check_errors=check_errors)

    _assert_outputs(res_out, ref_out)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_default_arguments(dtype):
    a, b = _conditioned_pair(dtype, (20, 320, 15), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    # Schema defaults are left=True / check_errors=False; omitting both keywords
    # must reach the same result as passing them explicitly.
    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)
    res_explicit = flag_gems._linalg_solve_ex(a, b, left=True, check_errors=False)

    _assert_outputs(res_out, ref_out)
    _assert_outputs(res_explicit, ref_out)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_mixed_batch_info(dtype):
    n, m = 3, 2
    a = tu.make_input(dtype, (3, n, n), ["-1", "1"])
    a = a + torch.eye(n, dtype=dtype, device=flag_gems.device) * 4
    a[1] = 0  # one singular lane between two healthy ones
    b = tu.make_input(dtype, (3, n, m), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    assert res_out[3].tolist() == [0, 1, 0]


_BROADCAST_ROWS = [
    ((20, 320, 320), (320, 3), True, torch.float32),
    ((16, 128, 64, 64), (64, 3), True, torch.float64),
    ((20, 320, 320), (3, 320), False, torch.float32),
    ((2, 3, 5, 5), (5, 3), True, torch.complex64),
    ((2, 1, 5, 5), (2, 3, 5, 3), True, torch.complex128),
]
_BROADCAST_CASES = tu.selected_cases(
    [row for row in _BROADCAST_ROWS if row[3] in SUPPORTED_DTYPES], quick=[]
)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("a_shape,b_shape,left,dtype", _BROADCAST_CASES)
def test__linalg_solve_ex_broadcast(a_shape, b_shape, left, dtype):
    n = a_shape[-1]
    a = tu.make_input(dtype, a_shape, ["-1", "1"])
    a = a + torch.eye(n, dtype=dtype, device=flag_gems.device) * 4
    b = tu.make_input(dtype, b_shape, ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b, left=left)
    res_out = flag_gems._linalg_solve_ex(a, b, left=left)

    _assert_outputs(res_out, ref_out)
    # info follows A's batch shape, not the broadcast result shape.
    assert res_out[3].shape == a.shape[:-2]


_BACKWARD_CASES = tu.selected_cases(
    [(True, ()), (False, ()), (True, (3,)), (False, (3,))], quick=[]
)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("left,batch", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_backward(left, batch, dtype):
    n, m = 12, 3
    a = tu.make_input(dtype, batch + (n, n), ["-1", "1"])
    a = a + torch.eye(n, dtype=dtype, device=flag_gems.device) * 6
    b = tu.make_input(dtype, _rhs_shape(batch, n, m, left), ["-1", "1"])
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    ref_a = tu.to_reference(a).detach().requires_grad_(True)
    ref_b = tu.to_reference(b).detach().requires_grad_(True)

    # Same-valued, nonconstant upstream gradient for both sides.
    upstream = tu.make_input(dtype, _rhs_shape(batch, n, m, left), ["-1", "1"])

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b, left=left)
    res_out = flag_gems._linalg_solve_ex(a, b, left=left)
    # The forward tuple must agree before the gradients are taken.
    _assert_outputs(res_out, ref_out)

    ref_grad_a, ref_grad_b = torch.autograd.grad(
        ref_out[0], (ref_a, ref_b), grad_outputs=tu.to_reference(upstream)
    )
    res_grad_a, res_grad_b = torch.autograd.grad(
        res_out[0], (a, b), grad_outputs=upstream
    )

    tu.assert_result_close(res_grad_a, ref_grad_a)
    tu.assert_result_close(res_grad_b, ref_grad_b)


# Real payloads come straight from the shared special-value generator; complex
# payloads are added explicitly because special_value_cases() only yields
# floating-point dtypes.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(SUPPORTED_DTYPES)
    + [
        (dtype, scenario)
        for dtype in _COMPLEX_SUPPORTED_DTYPES
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__linalg_solve_ex_special_dense_diagonal(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    n = payload.numel()
    a = tu.make_input(dtype, (n, n), ["-1", "1"])
    torch.diagonal(a).copy_(payload)
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__linalg_solve_ex_special_rhs_column(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    n = 4
    a = tu.make_input(dtype, (n, n), ["-1", "1"])
    a = a + torch.eye(n, dtype=dtype, device=flag_gems.device) * 5
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    b[:, 0] = payload[:n]
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


# make_special_input payloads (tests/test_utils.py) are
#   nan   -> [nan,  0,   -0,  1,  -1]
#   inf   -> [inf, -inf,  0, -0,   1]
#   mixed -> [nan,  inf, -inf, 0,  -0]
# Only these indices hold the scenario's own nonfinite values, so they are the
# ones wired into the right-hand side; the mixed scenario therefore contributes
# a nan and an inf instead of degenerating to inf only.
_RHS_POISON_SLOTS = {
    "nan": ((0, 0),),
    "inf": ((0, 1),),
    "mixed": ((0, 0), (1, 1)),
}


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__linalg_solve_ex_special_single_rhs_position(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    n = 4
    a = tu.make_input(dtype, (n, n), ["-1", "1"])
    a = a + torch.eye(n, dtype=dtype, device=flag_gems.device) * 5
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    for row, payload_index in _RHS_POISON_SLOTS[scenario]:
        b[row, 0] = payload[payload_index]
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    # Only the polluted column is contaminated; the other column stays finite.
    assert bool(torch.isfinite(res_out[0][:, 1]).all())


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__linalg_solve_ex_special_isolated_batch_lane(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    n = payload.numel()
    a = tu.make_input(dtype, (2, n, n), ["-1", "1"])
    torch.diagonal(a, dim1=-2, dim2=-1)[1].copy_(payload)
    b = tu.make_input(dtype, (2, n, 2), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    # The untouched lane must not be contaminated by the special lane.
    assert bool(torch.isfinite(res_out[0][0]).all())


_IMAGINARY_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in _COMPLEX_SUPPORTED_DTYPES
        for scenario in ("nan", "inf")
    ],
    quick=[],
)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype,scenario", _IMAGINARY_SPECIAL_CASES)
def test__linalg_solve_ex_special_imaginary_diagonal(dtype, scenario):
    # Imaginary-only payload: a real component plus 1j*inf would smear NaN
    # across the whole entry, so the real and imaginary parts are built
    # separately with known intended values.
    payload = tu.make_special_input(torch.float32, scenario)
    n = payload.numel()
    a = tu.make_input(dtype, (n, n), ["-1", "1"])
    torch.diagonal(a).copy_(torch.complex(torch.zeros_like(payload), payload).to(dtype))
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


# Zero-extent systems carry no values, so their data is fully described by the
# shape; every row is built with the shared deterministic generator.
_ZERO_EXTENT_CASES = [
    ((0, 0), (0, 3)),
    ((2, 0, 0), (2, 0, 3)),
    ((4, 4), (4, 0)),
]

# Deterministic, nonconstant right-hand side for the smallest nonempty solve.
_MINIMAL_SYSTEM_RHS = [1.0, 2.0, -3.0]


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("a_shape,b_shape", _ZERO_EXTENT_CASES)
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_zero_extents(a_shape, b_shape, dtype):
    a = tu.make_input(dtype, a_shape, ["-1", "1"])
    b = tu.make_input(dtype, b_shape, ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    assert res_out[0].shape == tuple(b_shape)
    assert res_out[3].shape == tuple(a_shape[:-2])


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_minimal_1x1_system(dtype):
    # Smallest nonempty solve. A is the fixed nonzero [[4]], so the system is
    # well posed by construction instead of being an occasional singular random
    # sample, and the nonconstant right-hand side makes the result a real
    # division rather than 0/0.
    a = torch.full((1, 1), 4, dtype=dtype, device=flag_gems.device)
    b = torch.tensor([_MINIMAL_SYSTEM_RHS], dtype=dtype, device=flag_gems.device)
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    assert res_out[0].shape == tuple(b.shape)
    assert res_out[3].shape == ()


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_singular_info(dtype):
    n = 4
    a = torch.eye(n, dtype=dtype, device=flag_gems.device)
    a[2] = 0  # singular: the third row is entirely zero
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)
    assert int(res_out[3]) > 0
    assert bool(torch.isfinite(res_out[1]).all())


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_check_errors_raises(dtype):
    n = 4
    a = torch.eye(n, dtype=dtype, device=flag_gems.device)
    a[2] = 0
    b = tu.make_input(dtype, (n, 2), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._linalg_solve_ex(a, b, check_errors=True)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_result_overload(dtype):
    a, b = _conditioned_pair(dtype, (2, 19, 7), ["-1", "1"])
    batch, n, m = _system_shapes((2, 19, 7))
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    # The .result overload is exercised through the same public operator name.
    # The output geometry is statically known, so the reference buffers are
    # allocated directly and the reference call fills them.
    ref_result = ref_a.new_empty(batch + (n, m))
    ref_lu = ref_a.new_empty(batch + (n, n))
    ref_pivots = ref_a.new_empty(batch + (n,), dtype=torch.int32)
    ref_info = ref_a.new_empty(batch, dtype=torch.int32)
    torch.ops.aten._linalg_solve_ex.result(
        ref_a,
        ref_b,
        result=ref_result,
        LU=ref_lu,
        pivots=ref_pivots,
        info=ref_info,
    )

    res_buf_result = torch.empty(batch + (n, m), dtype=dtype, device=flag_gems.device)
    res_buf_lu = torch.empty(batch + (n, n), dtype=dtype, device=flag_gems.device)
    res_buf_pivots = torch.empty(
        batch + (n,), dtype=torch.int32, device=flag_gems.device
    )
    res_buf_info = torch.empty(batch, dtype=torch.int32, device=flag_gems.device)
    res_out = flag_gems._linalg_solve_ex(
        a,
        b,
        result=res_buf_result,
        LU=res_buf_lu,
        pivots=res_buf_pivots,
        info=res_buf_info,
    )

    assert res_out[0] is res_buf_result
    assert res_out[1] is res_buf_lu
    assert res_out[2] is res_buf_pivots
    assert res_out[3] is res_buf_info
    tu.assert_result_close(res_buf_result, ref_result)
    tu.assert_result_close(res_buf_lu, ref_lu)
    tu.assert_result_equal(res_buf_pivots, ref_pivots)
    tu.assert_result_equal(res_buf_info, ref_info)


def _view_state(tensor):
    """Geometry that a resize or re-stride of the caller's view would change."""
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
    )


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", tu.selected_cases(SUPPORTED_DTYPES, quick=[]))
def test__linalg_solve_ex_result_overload_strided_views(dtype):
    a, b = _conditioned_pair(dtype, (2, 8, 4), ["-1", "1"])
    batch, n, m = _system_shapes((2, 8, 4))
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_result = ref_a.new_empty(batch + (n, m))
    ref_lu = ref_a.new_empty(batch + (n, n))
    ref_pivots = ref_a.new_empty(batch + (n,), dtype=torch.int32)
    ref_info = ref_a.new_empty(batch, dtype=torch.int32)
    torch.ops.aten._linalg_solve_ex.result(
        ref_a,
        ref_b,
        result=ref_result,
        LU=ref_lu,
        pivots=ref_pivots,
        info=ref_info,
    )

    sentinel = 7.0
    # Each buffer carries a doubled last axis so the ::2 view already has the
    # required shape: result (batch, n, m), LU (batch, n, n), pivots (batch, n)
    # and info (batch,) via a collapsed second column. The interleaved elements
    # are guard regions that must stay untouched.
    result_buf = torch.full(
        batch + (n, 2 * m), sentinel, dtype=dtype, device=flag_gems.device
    )
    lu_buf = torch.full(
        batch + (n, 2 * n), sentinel, dtype=dtype, device=flag_gems.device
    )
    pivots_buf = torch.full(
        batch + (2 * n,), 7, dtype=torch.int32, device=flag_gems.device
    )
    info_buf = torch.full(batch + (2,), 7, dtype=torch.int32, device=flag_gems.device)

    result_view = result_buf[..., ::2]
    lu_view = lu_buf[..., ::2]
    pivots_view = pivots_buf[..., ::2]
    info_view = info_buf[:, 0]
    views = (result_view, lu_view, pivots_view, info_view)
    before = [_view_state(view) for view in views]

    res_out = flag_gems._linalg_solve_ex(
        a, b, result=result_view, LU=lu_view, pivots=pivots_view, info=info_view
    )

    assert res_out[0] is result_view
    assert res_out[1] is lu_view
    assert res_out[2] is pivots_view
    assert res_out[3] is info_view
    # Writes must neither reallocate nor re-stride the caller's views.
    assert [_view_state(view) for view in views] == before
    tu.assert_result_close(result_view, ref_result)
    tu.assert_result_close(lu_view, ref_lu)
    tu.assert_result_equal(pivots_view, ref_pivots)
    tu.assert_result_equal(info_view, ref_info)
    # Interleaved guard regions stay at their sentinel values.
    for guard, sentinel_value in (
        (result_buf[..., 1::2], sentinel),
        (lu_buf[..., 1::2], sentinel),
        (pivots_buf[..., 1::2], 7),
        (info_buf[:, 1], 7),
    ):
        tu.assert_result_equal(guard, torch.full_like(guard, sentinel_value))


_NONCONTIG_KINDS = tu.selected_cases(["offset", "transposed"], quick=[])


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("kind", _NONCONTIG_KINDS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__linalg_solve_ex_noncontiguous_layouts(kind, dtype):
    n = 4
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device) * 8
    if kind == "offset":
        # A and B are views with a nonzero storage offset. The conditioning is
        # applied in place so the operand keeps its offset and stride; building
        # a fresh tensor would compact the layout. tu.to_reference gives the
        # reference the same offset/stride.
        a_base = tu.make_input(dtype, (n + 4, n + 4), ["-1", "1"])
        a = a_base[2 : n + 2, 2 : n + 2]
        a.copy_(a + eye)
        b_base = tu.make_input(dtype, (n + 4, 4), ["-1", "1"])
        b = b_base[2 : n + 2, 1:4]
    else:
        # Transposed A (column-major stride) and a stride-2 right-hand side.
        a_base = tu.make_input(dtype, (n, n), ["-1", "1"]) + eye
        a = a_base.t()
        b_base = tu.make_input(dtype, (n, 6), ["-1", "1"])
        b = b_base[:, ::2]
    ref_a, ref_b = tu.to_reference(a), tu.to_reference(b)

    ref_out = torch.ops.aten._linalg_solve_ex(ref_a, ref_b)
    res_out = flag_gems._linalg_solve_ex(a, b)

    _assert_outputs(res_out, ref_out)


# The rejected table below is a measurement, not a derived property: cuSOLVER's
# LU factorization has no half kernel and rejects every integer/bool input, and
# both errors were observed on the NVIDIA backend ('lu_factor_cusolver not
# implemented for ...' and 'Expected a floating point or complex tensor as
# input. Got Char/Byte/Int/Long'). The whole table therefore lives inside the
# nvidia scope, so no other vendor inherits a rejection that was never measured
# there; nothing is probed or skipped at runtime, and this list is purely
# collection-time metadata.
_NATIVE_REJECTED_DTYPES = []
if flag_gems.runtime.device.vendor_name == "nvidia":
    _NATIVE_REJECTED_DTYPES += [
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.bool,
    ]
    if utils.bf16_is_supported:
        _NATIVE_REJECTED_DTYPES.append(torch.bfloat16)
    if utils.fp8_is_supported:
        _NATIVE_REJECTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
    if utils.int64_is_supported:
        _NATIVE_REJECTED_DTYPES.append(torch.int64)


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("dtype", _NATIVE_REJECTED_DTYPES)
def test__linalg_solve_ex_rejects_unsupported_dtype(dtype):
    a = tu.make_input(dtype, (4, 4), ["-1", "1"])
    b = tu.make_input(dtype, (4, 2), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_solve_ex(a, b)


_NEGATIVE_SHAPE_CASES = [
    ((5,), (5,)),  # rank < 2 is rejected by the operator itself
    ((5, 5), (1, 1)),  # B incompatible with AX = B
]


@pytest.mark._linalg_solve_ex
@pytest.mark.parametrize("a_shape,b_shape", _NEGATIVE_SHAPE_CASES)
def test__linalg_solve_ex_rejects_invalid_shapes(a_shape, b_shape):
    a = tu.make_input(torch.float32, a_shape, ["-1", "1"])
    b = tu.make_input(torch.float32, b_shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._linalg_solve_ex(a, b)


@pytest.mark._linalg_solve_ex
def test__linalg_solve_ex_requires_rhs():
    a = tu.make_input(torch.float32, (4, 4), ["-1", "1"])
    # B is a required positional argument. The candidate surfaces the missing
    # operand through the operator schema, which raises RuntimeError here
    # ('missing value for argument B'); some backends report TypeError instead.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._linalg_solve_ex(a)
