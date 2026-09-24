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
    # Shape/stride/offset of the supplied output buffers. Object identity alone
    # would still hold for a candidate that resized or re-strided a correctly
    # shaped strided view onto fresh contiguous storage, so the metadata is
    # compared before and after the candidate call.
    return [
        (tensor.shape, tensor.stride(), tensor.storage_offset()) for tensor in tensors
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


@pytest.mark.geqrf
@pytest.mark.parametrize("dtype", GEQRF_DTYPES)
@pytest.mark.parametrize("shape", GEQRF_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_geqrf_value_range(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    snapshot = inp.clone()

    ref_a, ref_tau = torch.ops.aten.geqrf(ref_inp)
    res_a, res_tau = flag_gems.geqrf(inp)

    tu.assert_result_close(res_a, ref_a)
    tu.assert_result_close(res_tau, ref_tau)
    # The default overload is not in-place.
    tu.assert_result_equal(inp, snapshot)


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
    tu.assert_result_equal(inp, snapshot)


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
    tu.assert_result_equal(inp, snapshot)


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
