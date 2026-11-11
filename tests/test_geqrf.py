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

# geqrf is the LAPACK Householder QR factorisation: an input of shape
# (*batch, M, N) is factored into the packed reflector matrix a (*batch, M, N)
# and the reflector scalars tau (*batch, min(M, N)). The default overload is
# out-of-place, and the a=/tau= overload writes caller-supplied buffers.
# Comments below record the native torch.ops.aten.geqrf behaviour measured on
# the nvidia CUDA target used for verification.

# Native support covers the LAPACK real/complex types only; the wide types are
# gated on the device's static fp64 capability flag.
GEQRF_DTYPES = [torch.float32, torch.complex64]
if utils.fp64_is_supported:
    GEQRF_DTYPES += [torch.float64, torch.complex128]

# Workloads that do not sweep the value ranges reuse the first shared range:
# tu.make_input only resolves the symbolic bound pairs returned by
# tu.selected_ranges(), a bare numeric bound is rejected by its resolver.
GEQRF_RANGE = tu.selected_ranges()[0]

# Native CPU and Hygon produce different nonfinite factor locations for these
# dtype-limit inputs. Default mode still checks them with a device reference.
_HYGON_CPU_EXTREME_DTYPES = {
    (256, 256): (torch.float32, torch.float64, torch.complex64, torch.complex128),
    (1024, 1024): (torch.complex64, torch.complex128),
    (512, 128): (torch.float32, torch.float64),
    (16, 128, 64, 60): (torch.float32, torch.float64),
}

# geqrf requires rank >= 2 ("torch.geqrf: input must have at least 2
# dimensions"), so the rank-0/rank-1 entries of the shared 7-shape grid cannot
# apply to it. They are replaced by two valid matrix alternatives, (256, 256)
# and (512, 128), and the (1024, 1024) 2-D entry plus the 3-D/4-D/5-D spec shapes
# are kept unchanged.
GEQRF_SHAPES = tu.selected_cases(
    [
        (256, 256),
        (1024, 1024),
        (512, 128),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ],
    quick=[(2, 19, 7)],
)

# Boundary sizes: 1x1 input, single column/row, tiny squares, 4-D batch.
# Default-only: quick keeps the value grid at (2, 19, 7).
GEQRF_BOUNDARY_SHAPES = tu.selected_cases(
    [(1, 1), (3, 1), (1, 3), (5, 3), (3, 7), (2, 3, 7, 5)],
    quick=[],
)

# Zero-sized factors are valid native inputs, e.g. (0, 0) -> a=(0, 0), tau=(0,).
# (0, 3, 4) is the distinct zero-*batch* boundary: the factorisation runs once
# per batch entry, so it must return a=(0, 3, 4) with tau=(0, 3) rather than
# collapsing the matrix axes. Default-only.
GEQRF_EMPTY_SHAPES = tu.selected_cases(
    [(0, 0), (0, 3), (3, 0), (2, 0, 3), (2, 3, 0), (0, 3, 4)],
    quick=[],
)

# Non-contiguous input views. Default-only.
GEQRF_INPUT_LAYOUTS = tu.selected_cases(["transposed", "sliced", "strided"], quick=[])

# The out overload is exercised with both a 2-D and a batched 4-D geometry, and
# each geometry with contiguous, offset and strided output buffers (the offset
# and strided layouts were measured as natively supported, so none needs an
# exemption). Default-only: quick lists only the plain value grid.
GEQRF_OUT_CASES = tu.selected_cases(
    [
        (shape, layout)
        for shape in [(16, 8), (2, 3, 7, 5)]
        for layout in ["contiguous", "offset", "strided"]
    ],
    quick=[],
)

# The three special-value scenarios of every supported dtype. Default-only.
GEQRF_SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in GEQRF_DTYPES
        for scenario in ["nan", "inf", "mixed"]
    ],
    quick=[],
)

# Isolated payload positions inside a 6x5 matrix: the packed output then has to
# propagate each payload through the following reflector updates, unlike the
# single-row degenerate matrix below.
GEQRF_PLACEMENTS = [(0, 0), (1, 2), (2, 3), (3, 1), (4, 4)]

GEQRF_PLACED_CASES = tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in GEQRF_DTYPES
        for scenario in ["nan", "inf", "mixed"]
    ],
    quick=[],
)

# Dtype rejection of the native kernel, recorded per backend: on the nvidia CUDA
# kernel these nine types fail with RuntimeError: "geqrf_cuda" not implemented
# for "<dtype>". The table is keyed by the collection-time vendor metadata
# (flag_gems.vendor_name, the same source tests/accuracy_utils.py uses) so the
# nvidia measurement is not asserted for a backend it was not taken on; such a
# backend keeps the negative families that do not depend on the dtype.
_MEASURED_DTYPE_REJECTION = {
    "nvidia": [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
}

# A tensor of these dtypes cannot even be created without the matching device
# capability (shared static flags); construction is independent of whether
# native geqrf rejects the dtype.
_DTYPE_NEEDS_CAPABILITY = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

GEQRF_UNSUPPORTED_DTYPES = [
    dtype
    for dtype in _MEASURED_DTYPE_REJECTION.get(flag_gems.vendor_name, [])
    if _DTYPE_NEEDS_CAPABILITY.get(dtype, True)
]

# geqrf has no optional parameter and no second operand, so broadcast does not
# apply (nothing to broadcast against) and backward is not differentiable: on a
# requires_grad input torch.autograd.grad raises
# NotImplementedError: the derivative for "geqrf" is not implemented. for
# float32/float64 and RuntimeError: geqrf does not support automatic
# differentiation for outputs with complex dtype. for complex64/complex128.
# The negative families tested below are an unsupported dtype and a rank below 2.
GEQRF_INVALID_RANK_SHAPES = [(), (1,), (4,)]


def make_strided_input(dtype, layout):
    # Non-contiguous views: the transposed and stride-2 batches keep storage
    # offset 0, while the sliced batch additionally starts at a non-zero offset.
    # The candidate has to read the values as they are stored.
    base_inp = tu.make_input(dtype, (4, 10, 6), GEQRF_RANGE)
    if layout == "transposed":
        return base_inp.transpose(-1, -2)
    if layout == "sliced":
        return base_inp[:, 2:8, :]
    return base_inp[:, ::2, :]


def make_out_buffers(reference, shape, layout):
    # Output buffers of the .a overload: a keeps the input shape (*batch, M, N)
    # and tau has (*batch, min(M, N)).
    batch, m, n = shape[:-2], shape[-2], shape[-1]
    inner = min(m, n)
    if layout == "contiguous":
        return reference.new_empty(shape), reference.new_empty(batch + (inner,))
    if layout == "offset":
        a_holder = reference.new_empty(batch + (m + 2, n + 2))
        tau_holder = reference.new_empty(batch + (inner + 1,))
        return a_holder[..., 1 : m + 1, :n], tau_holder[..., 1:]
    a_holder = reference.new_empty(batch + (m, 2 * n))
    tau_holder = reference.new_empty(batch + (2 * inner,))
    return a_holder[..., : 2 * n : 2], tau_holder[..., ::2]


def out_buffer_metadata(tensors):
    # Shape, strides, offset and storage of the supplied output buffers. Identity alone
    # would still hold for a candidate that resized or re-strided a correctly
    # shaped strided view onto fresh contiguous storage, so the metadata is
    # compared before and after the candidate call.
    return [
        (tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.data_ptr())
        for tensor in tensors
    ]


def make_placed_special_input(dtype, scenario):
    # The shared payload tensor is scattered with a device-side indexed
    # assignment onto isolated positions of a 6x5 matrix, so several reflector
    # rows have to be updated and the values never leave the device.
    payload = tu.make_special_input(dtype, scenario)
    rows = torch.tensor([row for row, _ in GEQRF_PLACEMENTS], device=flag_gems.device)
    cols = torch.tensor([col for _, col in GEQRF_PLACEMENTS], device=flag_gems.device)
    matrix = torch.full((6, 5), 0.5, dtype=dtype, device=flag_gems.device)
    matrix[rows, cols] = payload
    return matrix


def _assert_projected_terminal(column, a, tau):
    # Apply only the preceding reflectors to e_last. Neither terminal packed
    # value contributes to the projected coefficient used as the oracle.
    basis = torch.zeros((column.numel(), 1), dtype=column.dtype, device="cpu")
    basis[-1, 0] = 1
    q_last = torch.ormqr(a[:, :-1].cdouble(), tau[:-1].cdouble(), basis).squeeze(-1)
    eps = torch.finfo(a.real.dtype).eps
    norm = q_last.norm()
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=8 * eps, rtol=0)
    projected = torch.vdot(q_last, column)
    terminal = (1 - tau[-1].cdouble()) * a[-1, -1].cdouble()
    torch.testing.assert_close(
        terminal, projected, atol=2 * eps * column.norm().item(), rtol=0
    )


def _assert_cpu_comparison_metadata(inp, a, tau, ref_a, ref_tau):
    assert a.device == tau.device == inp.device
    assert a.dtype == tau.dtype == ref_a.dtype == ref_tau.dtype == inp.dtype
    assert a.shape == ref_a.shape == inp.shape
    assert tau.shape == ref_tau.shape == inp.shape[:-1]
    assert ref_a.device == ref_tau.device == torch.device("cpu")


def _assert_terminal_reflector_equivalent(
    inp, a, tau, ref_a, ref_tau, *, check_projection=False
):
    _assert_cpu_comparison_metadata(inp, a, tau, ref_a, ref_tau)
    a, tau = a.cpu(), tau.cpu()

    tu.assert_result_close(a[:-1], ref_a[:-1])
    tu.assert_result_close(a[-1, :-1], ref_a[-1, :-1])
    tu.assert_result_close(tau[:-1], ref_tau[:-1])

    # The final square reflector acts only on one coordinate. LAPACK and
    # Hygon may split that coordinate's phase differently between Q and R.
    phase = torch.stack((1 - tau[-1], 1 - ref_tau[-1]))
    torch.testing.assert_close(
        phase.abs(),
        torch.ones_like(phase.real),
        atol=8 * torch.finfo(phase.real.dtype).eps,
        rtol=0,
    )
    terminal = phase * torch.stack((a[-1, -1], ref_a[-1, -1]))
    assert torch.isfinite(terminal).all()
    if check_projection:
        column = inp[:, -1].cpu().cdouble()
        _assert_projected_terminal(column, a, tau)
        _assert_projected_terminal(column, ref_a, ref_tau)
    else:
        tu.assert_result_close(terminal[0], terminal[1])


def _assert_qr_factorization(inp, a, tau, ref_a, ref_tau):
    _assert_cpu_comparison_metadata(inp, a, tau, ref_a, ref_tau)
    work_dtype = torch.complex128 if inp.is_complex() else torch.float64
    original = inp.cpu().to(work_dtype)
    assert torch.isfinite(original).all()
    column_norms = torch.linalg.vector_norm(original, dim=0)
    identity = torch.eye(inp.shape[-1], dtype=original.dtype, device="cpu")
    tolerance = 32 * torch.finfo(inp.real.dtype).eps

    # Device and CPU can encode valid reflectors differently. Validate each
    # packed pair against the original matrix rather than against each other.
    for packed, reflectors in ((a, tau), (ref_a, ref_tau)):
        packed = packed.cpu().to(work_dtype)
        reflectors = reflectors.cpu().to(work_dtype)
        assert torch.isfinite(packed).all() and torch.isfinite(reflectors).all()
        tail_norm_sq = packed.tril(-1).abs().square().sum(dim=0)
        active = reflectors != 0
        assert (tail_norm_sq[~active] == 0).all()
        active_tau = reflectors[active]
        if inp.is_complex():
            # Unitarity of I - tau * v * vᴴ with an implicit leading one.
            norm_relation = active_tau.abs().square() * (1 + tail_norm_sq[active])
            expected = 2 * active_tau.real
        else:
            assert ((reflectors >= 0) & (reflectors <= 2)).all()
            norm_relation = active_tau * (1 + tail_norm_sq[active])
            expected = torch.full_like(norm_relation, 2)
        torch.testing.assert_close(norm_relation, expected, rtol=0, atol=tolerance)
        q = torch.linalg.householder_product(packed, reflectors)
        residual = torch.linalg.vector_norm(q @ packed.triu() - original, dim=0)
        assert (residual <= tolerance * column_norms).all()
        torch.testing.assert_close(q.mH @ q, identity, rtol=0, atol=tolerance)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("shape", GEQRF_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_geqrf_value_range(shape, value_range, dtype):
    if (
        flag_gems.vendor_name == "hygon"
        and utils.TO_CPU
        and value_range in (["0", "max"], ["min", "0"])
        and dtype in _HYGON_CPU_EXTREME_DTYPES.get(shape, ())
    ):
        pytest.skip("native CPU and Hygon overflow at different factor positions")

    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    snapshot = inp.clone()

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    if (
        flag_gems.vendor_name == "hygon"
        and utils.TO_CPU
        and dtype in (torch.float32, torch.complex64)
        and shape == (1024, 1024)
        and value_range == ["-1", "0"]
    ):
        _assert_qr_factorization(snapshot, res_a, res_tau, ref_a, ref_tau)
    elif (
        flag_gems.vendor_name == "hygon"
        and utils.TO_CPU
        and dtype in (torch.complex64, torch.complex128)
        and shape in ((256, 256), (1024, 1024))
        and value_range in (["-1", "1"], ["0", "1"], ["-1", "0"])
    ):
        _assert_terminal_reflector_equivalent(
            inp,
            res_a,
            res_tau,
            ref_a,
            ref_tau,
            check_projection=(
                dtype == torch.complex64
                and shape == (1024, 1024)
                and value_range == GEQRF_RANGE
            ),
        )
    else:
        tu.assert_result_close(res_a, ref_a)
        tu.assert_result_close(res_tau, ref_tau)
    # The default overload is not in-place.
    torch.testing.assert_close(inp, snapshot, atol=0, rtol=0, equal_nan=True)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("shape", GEQRF_BOUNDARY_SHAPES)
def test_geqrf_boundary_shapes(shape, dtype):
    inp = tu.make_input(dtype, shape, GEQRF_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("shape", GEQRF_EMPTY_SHAPES)
def test_geqrf_empty_shapes(shape, dtype):
    inp = tu.make_input(dtype, shape, GEQRF_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("layout", GEQRF_INPUT_LAYOUTS)
def test_geqrf_non_contiguous_input(layout, dtype):
    inp = make_strided_input(dtype, layout)
    ref_inp = tu.to_reference(inp)
    snapshot = inp.clone()

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)
    torch.testing.assert_close(inp, snapshot, atol=0, rtol=0, equal_nan=True)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("shape,layout", GEQRF_OUT_CASES)
def test_geqrf_out_buffers(shape, layout, dtype):
    inp = tu.make_input(dtype, shape, GEQRF_RANGE)
    ref_inp = tu.to_reference(inp)
    snapshot = inp.clone()
    ref_a, ref_tau = make_out_buffers(ref_inp, shape, "contiguous")
    out_a, out_tau = make_out_buffers(inp, shape, layout)
    supplied_metadata = out_buffer_metadata((out_a, out_tau))

    torch.ops.aten.geqrf.a(ref_inp, a=ref_a, tau=ref_tau)
    returned_a, returned_tau = flag_gems.geqrf(inp, a=out_a, tau=out_tau)

    # The out overload has to write the caller's buffers and hand back those
    # exact tensors, without resizing or re-striding them.
    assert returned_a is out_a
    assert returned_tau is out_tau
    assert out_buffer_metadata((out_a, out_tau)) == supplied_metadata
    tu.assert_result_close(out_a, ref_a)
    tu.assert_result_close(out_tau, ref_tau)
    torch.testing.assert_close(inp, snapshot, atol=0, rtol=0, equal_nan=True)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype,scenario", GEQRF_SPECIAL_CASES)
def test_geqrf_special_values(dtype, scenario):
    # The shared payload holds five values; reshaping it to (1, 5) keeps all of
    # them and satisfies geqrf's rank >= 2 requirement.
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = tu.to_reference(inp)

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype,scenario", GEQRF_PLACED_CASES)
def test_geqrf_special_values_placed(dtype, scenario):
    inp = make_placed_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    # A payload in an early row propagates through the later reflectors, which is
    # what distinguishes this workload from the single-row case; both packed A and
    # tau are compared with matching NaNs (equal_nan) and no masking, so a
    # candidate that computes different positions still fails.
    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_UNSUPPORTED_DTYPES)
def test_geqrf_unsupported_dtype(dtype):
    inp = torch.ones((8, 4), dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.geqrf(inp)


@pytest.mark.geqrf
@pytest.mark.parametrize("shape", GEQRF_INVALID_RANK_SHAPES)
def test_geqrf_rejects_rank_below_two(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.geqrf(inp)


_MASTER_REAL_SHAPES = [
    # degenerate
    (1, 1),
    (1, 8),
    (8, 1),
    # small path (single-program register kernel)
    (4, 4),
    (33, 33),
    (8, 3),
    (3, 8),
    # large path (blocked column-major panels)
    (256, 256),
    (300, 100),
    (100, 300),
    # batched, and batched with extra dims
    (4, 16, 16),
    (2, 3, 8, 8),
    (64, 8, 8),
    (2, 32, 8),
]
_MASTER_REAL_DTYPES = [
    dtype for dtype in GEQRF_DTYPES if dtype in (torch.float32, torch.float64)
]


def _well_conditioned(shape, dtype, device):
    # A = randn + m * I (diagonal bump over the min(m, n) diagonal): well
    # conditioned for QR, mirrors the det/slogdet/solve_ex recipe.
    m, n = shape[-2], shape[-1]
    A = torch.randn(shape, dtype=dtype, device=device)
    return A + m * torch.eye(m, n, dtype=dtype, device=device)


def _reconstruction(a, tau, A):
    """A = Q @ R from the packed (a, tau) pair; works for tall, square and
    wide inputs.  Q = H_0 H_1 ... H_{k-1} (reflectors compose right to
    left), so apply them to the identity in reverse column order."""
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = A.numel() // (m * n)
    A64 = A.detach().to("cpu").to(torch.float64).reshape(B, m, n)
    a64 = a.detach().to("cpu").to(torch.float64).reshape(B, m, n)
    tau64 = tau.detach().to("cpu").to(torch.float64).reshape(B, k)
    recon = 0.0
    for b in range(B):
        q = torch.eye(m, dtype=torch.float64)
        for j in range(k - 1, -1, -1):
            v = torch.zeros(m, dtype=torch.float64)
            v[j] = 1.0
            if j + 1 < m:
                v[j + 1 :] = a64[b, j + 1 :, j]
            tv = tau64[b, j]
            if tv != 0:
                q[:, j:] = q[:, j:] - torch.outer(v, tv * (v @ q[:, j:]))
        recon = max(recon, (q @ torch.triu(a64[b]) - A64[b]).abs().max().item())
    scale = A64.abs().max().item() + 1e-30
    return recon / scale


@pytest.mark.geqrf
@pytest.mark.parametrize("shape", _MASTER_REAL_SHAPES)
@pytest.mark.parametrize("dtype", _MASTER_REAL_DTYPES)
def test_geqrf_well_conditioned(shape, dtype):
    A = _well_conditioned(shape, dtype, flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_a, ref_tau = torch.ops.aten.geqrf.default(ref_A)
    res_a, res_tau = flag_gems.geqrf(A)

    assert res_a.device == res_tau.device == A.device
    assert res_a.dtype == res_tau.dtype == dtype
    assert res_a.shape == ref_a.shape
    assert res_tau.shape == ref_tau.shape

    # The packed reflectors and tau are implementation dependent (cuSOLVER vs
    # the Triton Householder chain produce different, equally valid
    # factorisations), so compare the deterministic reconstruction property
    # A = Q @ R instead of the factors themselves.
    max_rel = _reconstruction(res_a, res_tau, A)
    assert max_rel <= 2e-2, f"reconstruction max_rel={max_rel:.3e}"

    # R (= triu(a)) must match the reference R: for a full-rank well-
    # conditioned A the QR factor R is unique (positive/negative diagonal
    # convention may differ per column, so align signs first).
    ref_R = torch.triu(ref_a.to(torch.float64))
    res_R = torch.triu(utils.to_cpu(res_a, ref_a).to(torch.float64))
    k = min(shape[-2], shape[-1])
    sgn = torch.sign(res_R.diagonal(dim1=-2, dim2=-1))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    ref_sgn = torch.sign(ref_R.diagonal(dim1=-2, dim2=-1))
    ref_sgn = torch.where(ref_sgn == 0, torch.ones_like(ref_sgn), ref_sgn)
    # Scale each column of R by its diagonal sign: for a full-rank A the QR
    # factor R is unique up to a per-column sign.  sgn is (..., k); the wide
    # case (n > k) leaves the sign-free tail columns (j >= k) untouched, so
    # compare only the leading k columns.
    kcols = min(res_R.shape[-1], k)
    res_R_aligned = res_R[..., :kcols] * sgn.unsqueeze(-2)
    ref_R_aligned = ref_R[..., :kcols] * ref_sgn.unsqueeze(-2)
    # Both inputs are upcast to fp64 before the comparison, so the absolute
    # error here tracks the *input* precision: fp64 inputs agree with the
    # reference to ~2e-13 and fp32 inputs to ~2.4e-4 (worst over 10 seeds x
    # all shapes, sign-aligned). atol=1e-3 keeps ~4x headroom over the fp32
    # worst case while staying tight against the fp64 one; the rtol floor of
    # the fp64 resolution table (1e-7) covers the large-|R| entries.
    utils.gems_assert_close(res_R_aligned, ref_R_aligned, torch.float64, atol=1e-3)


@pytest.mark.geqrf
def test_geqrf_well_conditioned_non_contiguous():
    # transposed (non-contiguous) input must still produce the QR of the
    # logical matrix
    base = _well_conditioned((16, 8), torch.float32, flag_gems.device)
    At = base.t().t().t()[:8]  # non-contiguous view with the same rows
    ref = torch.ops.aten.geqrf.default(utils.to_reference(At.contiguous()))
    res_a, res_tau = flag_gems.geqrf(At)
    assert res_a.device == res_tau.device == At.device
    assert res_a.dtype == res_tau.dtype == At.dtype
    max_rel = _reconstruction(res_a, res_tau, At.contiguous())
    assert max_rel <= 2e-2, f"reconstruction max_rel={max_rel:.3e}"
    assert res_tau.shape == ref[1].shape


@pytest.mark.geqrf
def test_geqrf_return_structure():
    # multi-output contract: a tuple (a, tau) with the packed layout
    A = _well_conditioned((8, 8), torch.float32, flag_gems.device)
    out = flag_gems.geqrf(A)
    assert isinstance(out, tuple) and len(out) == 2
    a, tau = out
    assert a.device == tau.device == A.device
    assert a.dtype == tau.dtype == A.dtype
    assert a.shape == (8, 8) and tau.shape == (8,)
    # strict lower triangle of the first k columns holds the packed
    # reflectors with unit implicit diagonal -> a's diagonal holds R
    assert torch.all(torch.triu(a).diagonal()[1:] != 0)


@pytest.mark.geqrf
def test_geqrf_zero_and_tiny_sizes():
    # zero-size batch
    A = _well_conditioned((0, 4, 4), torch.float32, flag_gems.device)
    a, tau = flag_gems.geqrf(A)
    assert a.device == tau.device == A.device
    assert a.dtype == tau.dtype == A.dtype
    assert a.shape == (0, 4, 4) and tau.shape == (0, 4)
    # 1x1: no reflection, tau == 0
    A = torch.tensor([[3.0]], dtype=torch.float32, device=flag_gems.device)
    a, tau = flag_gems.geqrf(A)
    assert a.device == tau.device == A.device
    assert a.dtype == tau.dtype == A.dtype
    ref_a, ref_tau = torch.ops.aten.geqrf.default(utils.to_reference(A))
    assert tau.item() == ref_tau.item() == 0.0
    utils.gems_assert_close(a, ref_a, torch.float32)
