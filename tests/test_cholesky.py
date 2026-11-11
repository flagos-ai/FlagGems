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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::cholesky(A, upper=False) factors a Hermitian positive-definite matrix, or
# a batch of them, into a lower (default) or upper triangular factor. Probes on
# the active backend (cuda, torch 2.8.0a0, vendor 'nvidia') fixed the domain:
#   * rank < 2 and non-square inputs raise RuntimeError,
#   * only float32, float64, complex64 and complex128 factorize,
#   * the triangle opposite to `upper` is ignored, non-finite entries included,
#   * non-positive-definite inputs, a nan diagonal entry and a -inf diagonal
#     entry raise torch._C._LinAlgError (a RuntimeError subclass),
#   * a +inf diagonal entry does not raise and yields a non-finite factor.
# CHOLESKY_DTYPES is the four implemented dtypes of the spec grid; the other spec
# dtypes stay as negative workloads below.
CHOLESKY_DTYPES = [torch.float32, torch.complex64]
if utils.fp64_is_supported:  # static capability flag, not a runtime probe
    CHOLESKY_DTYPES += [torch.float64, torch.complex128]

# The spec dtype grid minus the four dtypes above. These are measured negatives,
# scoped to the vendor they were measured on: on the active NVIDIA CUDA backend
# each raises RuntimeError "cholesky_cusolver" not implemented for '<type>'. The
# list is assembled at collection time from static capability flags only (no
# runtime dtype probing), and dtypes the backend does not advertise in the first
# place (bfloat16, int64, fp8) are included only when their flag says they exist.
UNSUPPORTED_DTYPES = []
if flag_gems.vendor_name == "nvidia":
    UNSUPPORTED_DTYPES = [
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.bool,
    ]
    if utils.bf16_is_supported:
        UNSUPPORTED_DTYPES.append(torch.bfloat16)
    if utils.int64_is_supported:
        UNSUPPORTED_DTYPES.append(torch.int64)
    if utils.fp8_is_supported:
        for _name in ("float8_e4m3fn", "float8_e5m2"):
            _dtype = getattr(torch, _name, None)
            if _dtype is not None:
                UNSUPPORTED_DTYPES.append(_dtype)


def _square(shape):
    # Square the last two dimensions without changing the rank: n is the smaller
    # of the two prescribed matrix extents, so (1024, 1024) stays 2-D and
    # (16, 128, 64, 60) becomes (16, 128, 60, 60).
    n = min(shape[-2], shape[-1])
    return shape[:-2] + (n, n)


CORE_SHAPES = [_square(shape) for shape in tu.selected_shapes() if len(shape) >= 2]
# Extra sizes, the n == 1 and empty-batch boundaries, and the empty matrices are
# additional positive workloads, so they are default-only.
EXTRA_SHAPES = tu.selected_cases([(8, 8), (64, 64), (256, 256)], quick=[])
BOUNDARY_SHAPES = tu.selected_cases([(1024, 1, 1), (0, 4, 4)], quick=[])
EMPTY_SHAPES = tu.selected_cases([(0, 0), (2, 0, 0)], quick=[])
SHAPES = CORE_SHAPES + EXTRA_SHAPES + BOUNDARY_SHAPES + EMPTY_SHAPES

# All five spec ranges for every shape. 'gram' expresses the three signed ranges
# through the factor; the two non-negative ranges use both fixtures.
GRAM_RANGES = [["-1", "1"], ["0", "1"], ["-1", "0"]]
DIAGONAL_RANGES = [["0", "1"], ["0", "max"]]
VALUE_ROWS = tu.selected_cases(
    [("gram", shape, rng) for shape in SHAPES for rng in GRAM_RANGES]
    + [("diagonal", shape, rng) for shape in SHAPES for rng in DIAGONAL_RANGES],
    quick=[("gram", _square(tu.QUICK_SHAPES[0]), tu.QUICK_RANGES[0])],
)

_REAL_OF = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def _construction_device(dtype):
    if dtype.is_complex and flag_gems.vendor_name in ("ascend", "kunlunxin"):
        return "cpu"
    return flag_gems.device


def _comparison_values(result, reference):
    device = torch.device(flag_gems.device)
    if device.index is None and device.type != "cpu":
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    assert result.device == device
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    assert result.layout == reference.layout
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else device)
    if result.is_complex() and flag_gems.vendor_name in ("ascend", "kunlunxin"):
        result, reference = _observe(result), _observe(reference)
    return result, reference


def _observe(tensor):
    if tensor.device.type == "cpu":
        return tensor
    values = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
    )
    if tensor.is_conj():
        values = values.conj()
    if tensor.is_neg():
        values = torch._neg_view(values)
    return values


def _assert_close(result, reference):
    result, reference = _comparison_values(result, reference)
    tu.assert_result_close(result, reference)


_ILUVATAR_MASTER_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="Master coverage synchronization for the Iluvatar suite",
)
_MASTER_REAL_DTYPES = [torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)
_MASTER_PD_CASES = [
    *(((n, n), False, False) for n in (2, 4, 8, 16, 32)),
    *(((n, n), True, False) for n in (2, 4, 8)),
    ((2, 4, 4), False, False),
    ((3, 8, 8), False, False),
    *(((n, n), False, True) for n in (2, 4, 8, 32)),
    *(((2, 3, n, n), False, False) for n in (2, 4, 8)),
    ((128, 128), False, False),
]


def _make_master_spd(shape, dtype):
    device = flag_gems.device
    factor = torch.randn(shape, dtype=dtype, device=device)
    return factor @ factor.transpose(-2, -1) + 0.1 * torch.eye(
        shape[-1], dtype=dtype, device=device
    )


def _check_master_factorization(inp, upper=False):
    kwargs = {"upper": True} if upper else {}
    reference = torch.cholesky(utils.to_reference(inp), **kwargs)
    result = flag_gems.cholesky(inp, **kwargs)
    actual, expected = _comparison_values(result, reference)
    utils.gems_assert_close(actual, expected, inp.dtype)


@pytest.mark.cholesky
@_ILUVATAR_MASTER_ONLY
@pytest.mark.parametrize("shape,upper,transposed", _MASTER_PD_CASES)
@pytest.mark.parametrize("dtype", _MASTER_REAL_DTYPES)
def test_cholesky_master_real_grid(shape, upper, transposed, dtype):
    inp = _make_master_spd(shape, dtype)
    if transposed:
        inp = inp.transpose(-2, -1)
    _check_master_factorization(inp, upper)


@pytest.mark.cholesky
@_ILUVATAR_MASTER_ONLY
@pytest.mark.parametrize("shape", [(2, 3, 3), (3, 8, 8), (2, 2, 4, 4)])
def test_cholesky_master_non_contiguous_batch(shape):
    inp = _make_master_spd(shape, torch.float32)
    for view in (inp.transpose(1, 2), inp[::2]):
        if view.shape[-2] == view.shape[-1]:
            _check_master_factorization(view)


@pytest.mark.cholesky
@_ILUVATAR_MASTER_ONLY
@pytest.mark.parametrize("dtype", _MASTER_REAL_DTYPES)
def test_cholesky_master_negative_definite(dtype):
    inp = -_make_master_spd((4, 4), dtype)
    with pytest.raises(torch.linalg.LinAlgError):
        flag_gems.cholesky(inp)
    with pytest.raises(torch.linalg.LinAlgError):
        torch.cholesky(utils.to_reference(inp))


@pytest.mark.cholesky
@_ILUVATAR_MASTER_ONLY
def test_cholesky_master_empty_batch():
    inp = torch.empty(0, 4, 4, device=flag_gems.device)
    result = flag_gems.cholesky(inp)
    assert result.shape == inp.shape
    assert result.device == inp.device and result.dtype == inp.dtype
    assert result is not inp


def _special_input(dtype, scenario):
    if _construction_device(dtype) == "cpu":
        return tu.make_special_input(torch.float32, scenario).cpu().to(dtype)
    return tu.make_special_input(dtype, scenario)


def _full(shape, value, dtype, device):
    creation_device = device
    if dtype.is_complex and flag_gems.vendor_name in ("ascend", "kunlunxin"):
        creation_device = "cpu"
    return torch.full(shape, value, dtype=dtype, device=creation_device).to(device)


def _skip_reference_gap(family, dtype, **params):
    vendor = flag_gems.vendor_name
    reason = None
    if family == "backward":
        if vendor in ("ascend", "kunlunxin") and dtype == torch.complex64:
            reason = f"{vendor} native complex64 cholesky backward is unsupported"
    elif family == "imaginary_diagonal":
        if vendor == "iluvatar" and dtype == torch.complex64:
            reason = "iluvatar native cholesky retains the imaginary input diagonal"
    elif family == "non_positive_definite":
        if vendor == "metax" and params["fixture"] == "mixed_batch":
            reason = "metax native cholesky does not reject the invalid middle batch"
    elif family == "special_diagonal":
        if (
            vendor in ("ascend", "iluvatar", "metax")
            and params["scenario"] in ("nan", "mixed")
            and (vendor == "metax" or dtype in (torch.float32, torch.complex64))
        ):
            reason = (
                f"{vendor} native cholesky does not reject this NaN diagonal payload"
            )
    elif family == "non_finite_rejected":
        value = params["value"]
        if (
            vendor in ("ascend", "iluvatar", "metax")
            and math.isnan(value)
            and (vendor == "metax" or dtype in (torch.float32, torch.complex64))
        ) or (
            vendor in ("ascend", "metax")
            and dtype.is_complex
            and (vendor == "metax" or dtype == torch.complex64)
            and params["placement"] == "hermitian_off_diagonal"
            and value == float("inf")
        ):
            reason = f"{vendor} native cholesky does not reject this non-finite input"
    elif family == "inf_diagonal":
        if vendor == "nvidia" and utils.TO_CPU:
            reason = "CPU and NVIDIA cholesky disagree on the +inf diagonal factor"
        elif vendor == "kunlunxin" and dtype == torch.float32:
            reason = "kunlunxin native float32 cholesky rejects a +inf diagonal"
    if reason is not None:
        pytest.skip(reason)


def _range_upper(real_dtype, value_range):
    bound = value_range[1]
    if bound == "max":
        return torch.finfo(real_dtype).max
    return float(bound)


def _make_pd_input(fixture, dtype, shape, value_range):
    # 'gram' builds A = M @ M.mH + n * I from the shared sampler. The sampled
    # range constrains the entries of the factor M, while A's entries are the Gram
    # of M plus the n-scaled diagonal: they are not themselves bounded by the
    # sampled range and grow with the matrix size and the row magnitudes of M.
    # 'diagonal' builds A = diag(upper - (upper - |sampled|) / 2): a deterministic
    # diagonal, strictly positive by construction, whose entries lie in the
    # interval [upper / 2, upper], i.e. inside the requested non-negative range
    # [0, upper]. The draw still samples the whole requested interval; only the
    # mapped magnitude is confined to its upper half, so no endpoint occurrence is
    # claimed. A diagonal entry must be strictly positive for a positive-definite
    # matrix, and the shared sampler can return an exact zero (torch.rand is a
    # finite discrete distribution), so the magnitude is mapped instead of being
    # used raw; the algebraically equivalent 0.5 * (upper + sampled) form is
    # avoided only because it overflows when sampled equals finfo.max. This is also
    # the only finite fixture for the dtype-bound range, where squaring sampled
    # magnitudes inside a Gram term overflows to inf. The complex part of a
    # diagonal sample is dropped and the magnitude is taken from its real part.
    if fixture == "diagonal":
        real_dtype = _REAL_OF.get(dtype, dtype)
        upper = _range_upper(real_dtype, value_range)
        sampled = tu.make_input(real_dtype, shape[:-1], value_range).abs()
        sampled = sampled.to(_construction_device(dtype))
        magnitude = upper - 0.5 * (upper - sampled)
        return torch.diag_embed(magnitude.to(dtype)).to(flag_gems.device)

    n = shape[-1]
    m = tu.make_input(dtype, shape, value_range)
    device = _construction_device(dtype)
    if flag_gems.vendor_name == "kunlunxin" and m.numel() == 0:
        device = "cpu"
    m = m.to(device)
    return (m @ m.mH + n * torch.eye(n, dtype=dtype, device=device)).to(
        flag_gems.device
    )


@pytest.mark.cholesky
@pytest.mark.parametrize("fixture,shape,value_range", VALUE_ROWS)
@pytest.mark.parametrize("dtype", CHOLESKY_DTYPES)
def test_cholesky_value_range(fixture, dtype, shape, value_range):
    inp = _make_pd_input(fixture, dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.cholesky(ref_inp)
    res_out = flag_gems.cholesky(inp)

    _assert_close(res_out, ref_out)


# The optional schema bool: the default value is exercised by omitting the
# argument, plus both explicit values.
UPPER_CASES = tu.selected_cases([None, False, True], quick=[])


@pytest.mark.cholesky
@pytest.mark.parametrize("upper", UPPER_CASES)
@pytest.mark.parametrize("dtype", CHOLESKY_DTYPES)
def test_cholesky_upper(dtype, upper):
    inp = _make_pd_input("gram", dtype, (1024, 1024), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    kwargs = {} if upper is None else {"upper": upper}

    ref_out = torch.ops.aten.cholesky(ref_inp, **kwargs)
    res_out = flag_gems.cholesky(inp, **kwargs)

    _assert_close(res_out, ref_out)


LAYOUT_CASES = tu.selected_cases(
    [
        (dtype, layout)
        for dtype in CHOLESKY_DTYPES
        for layout in ("strided_slice", "storage_offset", "transposed")
    ],
    quick=[],
)


def _layout_input(base, layout):
    # Returns (input view, the storage the view is part of). All three views are
    # native-valid for the factorization: a stride-(2n, 2) slice, a view with
    # strides (n+1, 1) and a nonzero storage offset, and a transposed real view or
    # a complex .mH view, which also carries the lazy conjugate bit.
    device = _construction_device(base.dtype)
    base = base.to(device)
    n = base.shape[-1]
    if layout == "strided_slice":
        storage = torch.zeros((n, 2 * n), dtype=base.dtype, device=device)
        view = storage[:, ::2]
        view.copy_(base)
    elif layout == "storage_offset":
        storage = torch.zeros((n + 1, n + 1), dtype=base.dtype, device=device)
        view = storage[1:, 1:]
        view.copy_(base)
    else:
        view, storage = (base.t() if not base.is_complex() else base.mH), base
    target_storage = storage.to(flag_gems.device)
    if target_storage is storage:
        return view, storage
    target_view = torch.as_strided(
        target_storage, view.size(), view.stride(), view.storage_offset()
    )
    if view.is_conj():
        target_view = target_view.conj()
    return target_view, target_storage


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,layout", LAYOUT_CASES)
def test_cholesky_input_layout(dtype, layout):
    base = _make_pd_input("gram", dtype, (8, 8), ["-1", "1"])
    inp, storage = _layout_input(base, layout)
    preserved = storage.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.cholesky(ref_inp)
    res_out = flag_gems.cholesky(inp)

    # The factorization is not in-place: nothing in the input storage, inside or
    # outside the view, may change.
    assert storage.device == preserved.device
    values, expected = _comparison_values(storage, tu.to_reference(preserved))
    tu.assert_result_equal(values, expected)
    _assert_close(res_out, ref_out)


OUT_CASES = tu.selected_cases(
    [
        (dtype, upper, strided, (8, 8))
        for dtype in CHOLESKY_DTYPES
        for upper in (False, True)
        for strided in (False, True)
    ]
    # the original contiguous float32 lower-triangular case at 64x64
    + [(torch.float32, False, False, (64, 64))],
    quick=[],
)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,upper,strided,shape", OUT_CASES)
def test_cholesky_out(dtype, upper, strided, shape):
    # torch.ops.aten.cholesky.out is natively callable (probed for both upper
    # modes and all four supported dtypes): it writes into the supplied buffer and
    # returns that same tensor, so the candidate is called with out= directly
    # instead of being simulated through the default overload plus copy_.
    inp = _make_pd_input("gram", dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    n = inp.shape[-1]

    if strided:
        ref_backing = _full((n, 2 * n), 7.0, ref_inp.dtype, ref_inp.device)
        ref_out = torch.as_strided(ref_backing, (n, n), (2 * n, 2))
        backing = _full((n, 2 * n), 7.0, dtype, flag_gems.device)
        out = torch.as_strided(backing, (n, n), (2 * n, 2))
    else:
        ref_out = _full(ref_inp.shape, 7.0, ref_inp.dtype, ref_inp.device)
        out = _full(inp.shape, 7.0, dtype, flag_gems.device)

    ref_ret = torch.ops.aten.cholesky.out(ref_inp, upper, out=ref_out)
    res_ret = flag_gems.cholesky(inp, upper, out=out)

    assert res_ret is out
    _assert_close(res_ret, ref_ret)
    if strided:
        # the writes must stay inside the strided view
        values = _observe(backing) if backing.is_complex() else backing
        assert (values[:, 1::2] == 7.0).all()


BACKWARD_CASES = tu.selected_cases(
    [(dtype, upper) for dtype in CHOLESKY_DTYPES for upper in (False, True)], quick=[]
)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,upper", BACKWARD_CASES)
def test_cholesky_backward(dtype, upper):
    _skip_reference_gap("backward", dtype)
    inp = _make_pd_input("gram", dtype, (4, 8, 8), ["-1", "1"])
    upstream = tu.make_input(dtype, (4, 8, 8), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_inp = tu.to_reference(inp).detach().requires_grad_(True)
    res_inp = inp.detach().requires_grad_(True)

    ref_out = torch.ops.aten.cholesky(ref_inp, upper)
    res_out = flag_gems.cholesky(res_inp, upper)
    _assert_close(res_out, ref_out)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_outputs=upstream)

    _assert_close(res_grad, ref_grad)


TRIANGLE_CASES = tu.selected_cases(
    [(dtype, upper) for dtype in CHOLESKY_DTYPES for upper in (False, True)], quick=[]
)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,upper", TRIANGLE_CASES)
def test_cholesky_opposite_triangle(dtype, upper):
    # The factorization reads only the triangle selected by upper: adding 1000 to
    # the opposite triangle leaves the native result unchanged (probed bitwise for
    # all four supported dtypes), so the perturbed matrix is a valid input and its
    # reference is the native result on that same matrix.
    inp = _make_pd_input("gram", dtype, (8, 8), ["-1", "1"])
    inp = inp.to(_construction_device(dtype))
    n = inp.shape[-1]
    ones = torch.ones((n, n), dtype=dtype, device=inp.device)
    if upper:
        perturbation = torch.tril(ones, diagonal=-1) * 1000
    else:
        perturbation = torch.triu(ones, diagonal=1) * 1000
    perturbed = (inp + perturbation).to(flag_gems.device)

    ref_inp = tu.to_reference(perturbed)
    ref_out = torch.ops.aten.cholesky(ref_inp, upper)
    res_out = flag_gems.cholesky(perturbed, upper)

    _assert_close(res_out, ref_out)


IMAGINARY_DIAGONAL_CASES = tu.selected_cases(
    [dtype for dtype in CHOLESKY_DTYPES if dtype.is_complex], quick=[]
)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype", IMAGINARY_DIAGONAL_CASES)
def test_cholesky_imaginary_diagonal(dtype):
    _skip_reference_gap("imaginary_diagonal", dtype)
    # Measured: only the real part of the diagonal is read, so a matrix with an
    # imaginary diagonal part is a valid input and the native result is the one of
    # the same matrix with a real diagonal (identical bitwise for complex64 and
    # complex128). The candidate must reproduce that native result.
    inp = _make_pd_input("gram", dtype, (8, 8), ["-1", "1"])
    inp = inp.to(_construction_device(dtype))
    inp = (inp + 1j * 5 * torch.eye(8, dtype=dtype, device=inp.device)).to(
        flag_gems.device
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.cholesky(ref_inp)
    res_out = flag_gems.cholesky(inp)

    _assert_close(res_out, ref_out)


# Negatives use their own small shapes, independent of the positive grid, and are
# collected in both modes.
INVALID_SHAPE_CASES = [
    (torch.float32, ()),
    (torch.float32, (4,)),
    (torch.float32, (3, 4)),
    (torch.float32, (2, 3, 4)),
]


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,shape", INVALID_SHAPE_CASES)
def test_cholesky_invalid_shape(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cholesky(inp)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_cholesky_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (4, 4), ["0", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.cholesky(inp)


NON_PD_FIXTURES = ["negative_definite", "singular", "late_minor", "mixed_batch"]
NON_PD_CASES = [
    (dtype, fixture) for dtype in CHOLESKY_DTYPES for fixture in NON_PD_FIXTURES
]


def _make_non_pd_input(dtype, fixture):
    # Non-positive-definite inputs. Every fixture below raised
    # torch._C._LinAlgError on the active backend for all four supported dtypes,
    # so the native behavior asserted here is rejection.
    n = 8
    if fixture == "negative_definite":
        # a diagonal of negative entries; the spec range ['min', '0'] cannot build
        # a positive-definite matrix (a diagonal entry must be > 0), so it is
        # covered here as negative-domain input instead of a positive workload
        diagonal = -tu.make_input(dtype, (n,), ["min", "0"]).real.abs()
        return torch.diag_embed(diagonal).to(dtype)
    if fixture == "singular":
        # one zero diagonal entry
        diagonal = tu.make_input(dtype, (n,), ["0", "1"]).real.abs() + 0.5
        diagonal[-1] = 0.0
        return torch.diag_embed(diagonal).to(dtype)
    if fixture == "late_minor":
        # the leading minors are positive definite, the last one is not
        diagonal = tu.make_input(dtype, (n,), ["0", "1"]).real.abs() + 0.5
        diagonal[-1] = -1.0
        return torch.diag_embed(diagonal).to(dtype)
    # a single negative-definite matrix among positive-definite ones
    m = tu.make_input(dtype, (3, n, n), ["-1", "1"])
    m = m.to(_construction_device(dtype))
    inp = m @ m.mH + n * torch.eye(n, dtype=dtype, device=m.device)
    inp[1] = -torch.eye(n, dtype=dtype, device=m.device)
    return inp.to(flag_gems.device)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,fixture", NON_PD_CASES)
def test_cholesky_non_positive_definite(dtype, fixture):
    _skip_reference_gap("non_positive_definite", dtype, fixture=fixture)
    inp = _make_non_pd_input(dtype, fixture)

    with pytest.raises(RuntimeError):
        flag_gems.cholesky(inp)


# The shared NaN/Inf/mixed payload written onto the diagonal of an otherwise
# positive-definite matrix. Measured on this backend for float32, float64,
# complex64 and complex128: every scenario raises torch._C._LinAlgError. The
# payload puts zero and negative entries on the diagonal as well as non-finite
# ones, so the matrix is not positive definite and the rejection has more than one
# cause; either way the input is valid and the workload is kept as its own family.
DIAGONAL_PAYLOAD_CASES = [
    (dtype, scenario)
    for dtype in CHOLESKY_DTYPES
    for scenario in ("nan", "inf", "mixed")
]


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,scenario", DIAGONAL_PAYLOAD_CASES)
def test_cholesky_special_values_on_diagonal(dtype, scenario):
    _skip_reference_gap("special_diagonal", dtype, scenario=scenario)
    n = 5
    inp = 4 * torch.eye(n, dtype=dtype, device=_construction_device(dtype))
    index = torch.arange(n, device=inp.device)
    inp[index, index] = _special_input(dtype, scenario)
    inp = inp.to(flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.cholesky(inp)


# Measured native behavior for one non-finite entry of an otherwise
# positive-definite matrix (A = 4 * I), identical for all four supported dtypes:
#   nan / -inf on the diagonal, nan / +inf on both off-diagonal entries
# all raise torch._C._LinAlgError. Measured separately: +inf on the diagonal does
# not raise but yields a non-finite factor, and a non-finite entry in the triangle
# the factorization ignores leaves the result unchanged.
NON_FINITE_REJECT_CASES = [
    (dtype, placement, value)
    for dtype in CHOLESKY_DTYPES
    for placement, value in (
        ("diagonal", float("nan")),
        ("diagonal", float("-inf")),
        ("hermitian_off_diagonal", float("nan")),
        ("hermitian_off_diagonal", float("inf")),
    )
]


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,placement,value", NON_FINITE_REJECT_CASES)
def test_cholesky_non_finite_rejected(dtype, placement, value):
    _skip_reference_gap("non_finite_rejected", dtype, placement=placement, value=value)
    inp = 4 * torch.eye(5, dtype=dtype, device=_construction_device(dtype))
    if placement == "diagonal":
        inp[0, 0] = value
    else:
        inp[0, 1] = value
        inp[1, 0] = value
    inp = inp.to(flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.cholesky(inp)


INF_DIAGONAL_CASES = tu.selected_cases(CHOLESKY_DTYPES, quick=[])


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype", INF_DIAGONAL_CASES)
def test_cholesky_inf_diagonal(dtype):
    _skip_reference_gap("inf_diagonal", dtype)
    # Measured: a +inf diagonal entry is not rejected. The native factor is nan at
    # that diagonal position and the unaffected factor elsewhere, reproducible
    # across repeated calls (verified with an equal_nan comparison), so the native
    # result is the reference here. The shared comparison matches the nan positions
    # and does not replace them with finite values.
    inp = 4 * torch.eye(5, dtype=dtype, device=_construction_device(dtype))
    inp[0, 0] = float("inf")
    inp = inp.to(flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.cholesky(ref_inp)
    res_out = flag_gems.cholesky(inp)

    _assert_close(res_out, ref_out)


IGNORED_TRIANGLE_CASES = tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in CHOLESKY_DTYPES
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)


@pytest.mark.cholesky
@pytest.mark.parametrize("dtype,scenario", IGNORED_TRIANGLE_CASES)
def test_cholesky_special_values_in_ignored_triangle(dtype, scenario):
    # The whole shared payload (nan / inf / mixed) is written into the strict upper
    # triangle of a diagonal matrix, so only the untouched lower triangle is read.
    # tu.make_special_input is used directly for the complex dtypes too: measured
    # for all four supported dtypes, the native result is the unchanged finite
    # factor, so a non-finite entry in the ignored triangle is a value to compare
    # rather than a rejection. The same payload on the diagonal is rejected
    # (test_cholesky_special_values_on_diagonal) and each single non-finite entry is
    # covered independently above.
    n = 5
    payload = _special_input(dtype, scenario)
    inp = 4 * torch.eye(n, dtype=dtype, device=_construction_device(dtype))
    rows, cols = torch.triu_indices(n, n, offset=1, device=inp.device)
    inp[rows, cols] = payload.repeat(2)[: rows.numel()]
    inp = inp.to(flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.cholesky(ref_inp)
    res_out = flag_gems.cholesky(inp)

    _assert_close(res_out, ref_out)
