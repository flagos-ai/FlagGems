# Copyright 2026 FlagOS Contributors. All rights reserved.
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
from . import kernelgen_test_utils as tu

# Native sparse log-softmax routes are unavailable on these backends. Keep
# the original cases and use CPU ATen, preserving the original target ABI before
# moving values to CPU for numerical comparison.
_CPU_REFERENCE = flag_gems.vendor_name in {"ascend", "kunlunxin"}


def _copy_dense_physical(x, device):
    """Copy raw storage, then reconstruct geometry and lazy bits; detached by design."""
    assert isinstance(x, torch.Tensor) and x.layout == torch.strided
    shape, stride, offset = tuple(x.shape), x.stride(), x.storage_offset()
    raw = torch.empty(0, dtype=torch.uint8, device=x.device).set_(
        x.untyped_storage(), 0, (x.untyped_storage().nbytes(),), (1,)
    )
    raw = raw.to(device=device).clone()
    y = raw.view(x.dtype).as_strided(shape, stride, offset)
    if x.is_conj():
        y = y.conj()
    if x.is_neg():
        y = torch._neg_view(y)
    assert y.dtype == x.dtype and y.shape == x.shape and y.stride() == stride
    assert y.storage_offset() == offset and y.is_neg() == x.is_neg()
    assert y.is_conj() == x.is_conj() and y.device == torch.device(device)
    return y


def _copy_tensor(x, device, *, preserve_requires_grad=False):
    assert isinstance(x, torch.Tensor)
    if x.layout == torch.sparse_coo:
        y = torch.sparse_coo_tensor(
            _copy_dense_physical(x._indices(), device),
            _copy_dense_physical(x._values(), device),
            tuple(x.shape),
            device=device,
            dtype=x.dtype,
            is_coalesced=x.is_coalesced(),
        )
        assert y.sparse_dim() == x.sparse_dim() and y.dense_dim() == x.dense_dim()
        assert y.is_coalesced() == x.is_coalesced()
    elif x.layout == torch.strided:
        y = _copy_dense_physical(x, device)
    else:
        raise NotImplementedError("Expected COO or dense-negative input")
    if preserve_requires_grad and x.requires_grad:
        y.requires_grad_(True)
    return y


def _to_reference(inp):
    if _CPU_REFERENCE:
        return _copy_tensor(inp, "cpu", preserve_requires_grad=True)
    return tu.to_reference(inp)


def _assert_result_close(result, reference, *, expected_device):
    assert isinstance(result, torch.Tensor)
    assert result.device == expected_device
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    if _CPU_REFERENCE:
        result = _copy_tensor(result, "cpu")
    tu.assert_result_close(result, reference)


def _assert_result_equal(result, reference, *, expected_device):
    assert isinstance(result, torch.Tensor)
    assert result.device == expected_device
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    if _CPU_REFERENCE:
        result = _copy_tensor(result, "cpu")
    tu.assert_result_equal(result, reference)


def _readonly_state(inp):
    """Snapshot the input ABI and backing bytes before invoking the candidate."""
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.sparse_coo
    outer = (
        inp.device,
        inp.dtype,
        tuple(inp.shape),
        inp.layout,
        inp.sparse_dim(),
        inp.dense_dim(),
        inp.is_coalesced(),
        inp.requires_grad,
        inp._version,
    )
    parts = []
    for part in (inp._indices(), inp._values()):
        storage = part.untyped_storage()
        metadata = (
            part.device,
            part.dtype,
            tuple(part.shape),
            part.stride(),
            part.storage_offset(),
            part.is_conj(),
            part.is_neg(),
            part.requires_grad,
            part._version,
            storage._cdata,
            storage.nbytes(),
            storage.data_ptr(),
            part.data_ptr(),
        )
        raw = torch.empty(0, dtype=torch.uint8, device=part.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        # Hold the original storage alive and save scalar pointers now: a live
        # Storage object can itself be resized/reallocated by a bad candidate.
        parts.append((storage, metadata, raw.to(device="cpu").clone()))
    return outer, parts


def _assert_readonly(inp, before):
    outer, parts = _readonly_state(inp)
    assert outer == before[0]
    for (_, metadata, raw), (_, old_metadata, old_raw) in zip(parts, before[1]):
        assert metadata == old_metadata
        assert torch.equal(raw, old_raw)


def _write_out(original_out, cpu_result):
    # Only original out-object identity is promised; native may resize/rebind storage.
    target = _copy_tensor(cpu_result, original_out.device)
    original_out.data = target
    return original_out


def backward_data(grad, output, dim, self_tensor, *, out=None):
    required_device = self_tensor.device
    if grad.device != required_device or output.device != required_device:
        raise RuntimeError(
            "original grad/output/self device mismatch before CPU transport"
        )
    if out is not None and out.device != required_device:
        raise RuntimeError("original out device mismatch before CPU transport")
    args = [_copy_tensor(x, "cpu", preserve_requires_grad=True) for x in (grad, output)]
    cpu_self = _copy_tensor(self_tensor, "cpu", preserve_requires_grad=True)
    if out is None:
        result = torch.ops.aten._sparse_log_softmax_backward_data.default(
            *args, dim, cpu_self
        )
        return _copy_tensor(result, required_device)
    cpu_out = _copy_tensor(out, "cpu", preserve_requires_grad=True)
    result = torch.ops.aten._sparse_log_softmax_backward_data.out(
        *args, dim, cpu_self, out=cpu_out
    )
    assert result is cpu_out
    return _write_out(out, result)


_cpu_oracle = backward_data


def _coalesce_input(inp):
    # Kunlunxin's native COO coalescer fails before the candidate runs. Keep
    # the original support/value dtype and the deliberately uncoalesced cases.
    if flag_gems.vendor_name == "kunlunxin":
        return _copy_tensor(_copy_tensor(inp, "cpu").coalesce(), inp.device)
    return inp.coalesce()


# The CUDA operator has float32/float64 kernels only: it reports
# "'log_softmax_backward' not implemented" for the reduced-precision and integer types
# and "'coalesce_sparse_cuda' not implemented" for fp8. float64 is gated on the static
# backend capability flag instead of being assumed.
SUPPORTED_DTYPES = [torch.float32]
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.float64)

_DEFAULT_RANGE = ["-1", "1"]

# Coordinates are always int64, independently of the value dtype (that is a separate
# requirement of a COO tensor). Correctness supports stay small; the benchmark covers
# the large sparse sizes.
_MAX_NNZ = 1024

# Support relationships, empty operands and hybrid COO layouts. Default-only: the quick
# level keeps the smoke grid and the negative cases.
_STRUCTURE_ROWS = tu.selected_cases(
    [
        # identical support
        ((6,), 1, 0, (0, 2, 4), (0, 2, 4), True),
        # same size, different support
        ((6,), 1, 0, (0, 2, 4), (1, 3, 5), True),
        # partial overlap; native keeps the output support and aligns the gradient
        ((6,), 1, 0, (0, 1, 2, 3), (1, 2, 3, 4), True),
        # disjoint support
        ((6,), 1, -1, (0, 1, 2), (3, 4, 5), True),
        # every operand empty
        ((6,), 1, 0, (), (), True),
        # empty gradient against a non-empty output and self
        ((6,), 1, 0, (), (0, 2, 4), True),
        # duplicate coordinates left uncoalesced, identical support
        ((6,), 1, 0, (0, 1, 1, 2), (0, 1, 1, 2), False),
        # duplicate coordinates left uncoalesced, differing support
        ((6,), 1, 0, (0, 1, 1, 2), (1, 1, 3, 3), False),
        # hybrid COO, reduce along the sparse dim
        ((4, 5), 1, 0, (0, 1, 2, 3), (0, 1, 2, 3), True),
        # hybrid COO, reduce along the dense dim
        ((4, 5), 1, 1, (0, 1, 2, 3), (0, 1, 2, 3), True),
        # hybrid COO, one sparse dim with two dense dims
        ((3, 4, 5), 1, 2, (0, 1, 2), (0, 1, 2), True),
        # hybrid COO, two sparse dims with one dense dim
        ((3, 4, 5), 2, 1, (0, 5, 11), (0, 5, 11), True),
        # hybrid COO whose dense block exceeds the 64-column tile
        ((3, 4, 130), 2, 0, (0, 3, 7), (0, 3, 7), True),
        ((3, 4, 130), 2, 0, (0, 3, 7), (1, 5, 7), True),
    ],
    quick=[],
)

# `dim` is swept at every rank, boundaries 0 and -1 included. The 0-dim COO tensor only
# accepts -1 (0 and -2 are out of range).
_DIM_ROWS = tu.selected_cases(
    [
        ((), -1),
        ((256,), 0),
        ((256,), -1),
        ((1024, 1024), 0),
        ((1024, 1024), 1),
        ((1024, 1024), -1),
        ((20, 320, 15), 0),
        ((20, 320, 15), -2),
        ((20, 320, 15), -1),
        ((16, 128, 64, 60), 0),
        ((16, 128, 64, 60), 2),
        ((16, 128, 64, 60), -1),
        ((16, 7, 57, 32, 29), 0),
        ((16, 7, 57, 32, 29), 3),
        ((16, 7, 57, 32, 29), -1),
    ],
    quick=[],
)

# nan only, inf only and nan+inf together, per supported float dtype, placed in either
# operand that carries values.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, slot)
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
        for slot in ("grad_output", "output")
    ],
    quick=[],
)

# Stale buffers must have their support and values rewritten, not extended. The third
# row passes an uncoalesced buffer whose coordinates are not sorted.
_OUT_BUFFER_ROWS = tu.selected_cases(
    [
        ((), True),
        ((0, 2, 4), True),
        ((4, 2, 0), False),
        ((1, 3, 5), True),
        ((2, 4, 5), True),
    ],
    quick=[],
)

# Dtypes the CUDA operator rejects (measured): "'log_softmax_backward' not implemented"
# for the reduced-precision, integer, bool and complex types and
# "'coalesce_sparse_cuda' not implemented" for fp8. Those are CUDA registrations, so the
# expectation is scoped to the vendor that reports them; a boolean log-softmax has no
# floating semantics on any backend.
_NVIDIA_REJECTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# A rejection fixture still has to be constructible, which needs the matching static
# backend capability; the flags gate construction only, so a supported dtype keeps its
# rejection coverage.
_DTYPE_CAPABILITY = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: tu.fp8_is_supported(),
    torch.float8_e5m2: tu.fp8_is_supported(),
}
_REJECTED_DTYPES = [
    dtype
    for dtype in (
        _NVIDIA_REJECTED_DTYPES
        if flag_gems.runtime.device.vendor_name == "nvidia"
        else [torch.bool]
    )
    if _DTYPE_CAPABILITY.get(dtype, True)
]

_INVALID_DIMS = [((7,), 1), ((7,), -2), ((6, 5), 2), ((6, 5), -3), ((), 1), ((), -2)]

# A mismatched pair needs two distinct float dtypes to exist, which is exactly the
# float64 capability. The row is selected in the parameter data, so the test below is an
# ordinary top-level function.
_MIXED_DTYPE_ROWS = (
    [(torch.float32, torch.float64), (torch.float64, torch.float32)]
    if utils.fp64_is_supported
    else []
)


def _coords(size, sparse_dim, dim, nnz, phase):
    """Support coordinates built on the device in O(nnz).

    A normalization group is identified by every sparse coordinate except the one along
    `dim`, which is why the group ids enumerate the other axes and the `dim` coordinate
    is inserted back at its own position. Several entries are placed in each group so
    the reduction has a within-group sum. `phase` moves the block of used groups along
    the other axes; the positive cases build all three operands with one phase, so they
    share a support and the reduction is not a single random term.
    """
    size = tuple(size)
    if sparse_dim == 0:
        # A 0-dim COO tensor carries indices of shape (0, nnz).
        return torch.empty((0, nnz), dtype=torch.long, device=flag_gems.device)
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long, device=flag_gems.device)
    device = flag_gems.device
    if dim < 0:
        dim += len(size)
    dim_size = size[dim]
    group_shape = tuple(size[a] for a in range(sparse_dim) if a != dim)
    n_groups = math.prod(group_shape) if group_shape else 1
    cols = max(1, min(dim_size, nnz))
    index = torch.arange(nnz, device=device)
    # The used groups are scattered over the other axes through `step`, and their count
    # is rounded up: with a floor count the trailing partial group can wrap back onto
    # group 0 and repeat coordinates that were meant to be distinct.
    used = max(1, -(-nnz // cols))
    step = max(1, n_groups // used)
    group_id = ((index // cols) * step + phase * max(1, n_groups // 5)) % n_groups
    dim_coord = index % cols
    parts = torch.unravel_index(group_id, group_shape) if group_shape else ()
    axes = iter(parts)
    flat = torch.stack(
        [dim_coord if a == dim else next(axes) for a in range(sparse_dim)]
    )
    return flat.to(torch.long)


def _nnz_for(size):
    return min(_MAX_NNZ, math.prod(size))


def _coo(size, sparse_dim, dim, nnz, dtype, value_range, phase):
    size = tuple(size)
    values = tu.make_input(dtype, (nnz,) + size[sparse_dim:], value_range)
    indices = _coords(size, sparse_dim, dim, nnz, phase)
    sparse = torch.sparse_coo_tensor(
        indices, values, torch.Size(size), device=flag_gems.device
    )
    return _coalesce_input(sparse)


def _coo_on_support(
    size, sparse_dim, support, dtype, value_range, coalesce, values=None
):
    size = tuple(size)
    if sparse_dim == 0:
        indices = torch.empty(
            (0, len(support)), dtype=torch.long, device=flag_gems.device
        )
    else:
        flat = torch.tensor(support, dtype=torch.long, device=flag_gems.device)
        indices = torch.stack(torch.unravel_index(flat, size[:sparse_dim]))
    if values is None:
        values = tu.make_input(dtype, (len(support),) + size[sparse_dim:], value_range)
    sparse = torch.sparse_coo_tensor(
        indices, values, torch.Size(size), device=flag_gems.device
    )
    # The rows that test duplicate coordinates must stay uncoalesced, otherwise the
    # operator never sees the duplicates.
    return _coalesce_input(sparse) if coalesce else sparse


def _coo_unit(size, sparse_dim, nnz, dtype, phase):
    # Only has to be a constructible operand for the rejection cases, so no value range
    # applies and coalesce() is skipped: fp8 has no coalesce kernel.
    size = tuple(size)
    values = torch.ones(
        (nnz,) + size[sparse_dim:], dtype=dtype, device=flag_gems.device
    )
    indices = _coords(size, sparse_dim, 0, nnz, phase)
    return torch.sparse_coo_tensor(
        indices, values, torch.Size(size), device=flag_gems.device
    )


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("size", tu.selected_shapes())
def test__sparse_log_softmax_backward_data_value_range(size, value_range, dtype):
    shape = tuple(size)
    nnz = _nnz_for(shape)
    # dim only has to be a valid group axis here; -1 exists at every rank, the 0-dim COO
    # tensor included.
    dim = -1

    # The three operands share one support, so the within-group sum of grad_output is
    # non-zero and the result exercises the reduction; they still hold independently
    # generated values. Differing supports are covered by the structure cases below.
    grad_output = _coo(shape, len(shape), dim, nnz, dtype, value_range, phase=0)
    output = _coo(shape, len(shape), dim, nnz, dtype, value_range, phase=0)
    self_ = _coo(shape, len(shape), dim, nnz, dtype, value_range, phase=0)

    expected_device = self_.device

    ref_out = torch.ops.aten._sparse_log_softmax_backward_data(
        _to_reference(grad_output),
        _to_reference(output),
        dim,
        _to_reference(self_),
    )
    res_out = flag_gems._sparse_log_softmax_backward_data(
        grad_output, output, dim, self_
    )

    # Compared as sparse COO: layout, support size and values are all part of the
    # operator's output contract.
    _assert_result_close(res_out, ref_out, expected_device=expected_device)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("size,dim", _DIM_ROWS)
def test__sparse_log_softmax_backward_data_dim(size, dim, dtype):
    shape = tuple(size)
    nnz = _nnz_for(shape)

    # Shared support, as above: the operands differ only in their values.
    grad_output = _coo(shape, len(shape), dim, nnz, dtype, _DEFAULT_RANGE, phase=0)
    output = _coo(shape, len(shape), dim, nnz, dtype, _DEFAULT_RANGE, phase=0)
    self_ = _coo(shape, len(shape), dim, nnz, dtype, _DEFAULT_RANGE, phase=0)

    expected_device = self_.device

    ref_out = torch.ops.aten._sparse_log_softmax_backward_data(
        _to_reference(grad_output),
        _to_reference(output),
        dim,
        _to_reference(self_),
    )
    res_out = flag_gems._sparse_log_softmax_backward_data(
        grad_output, output, dim, self_
    )

    _assert_result_close(res_out, ref_out, expected_device=expected_device)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "size,sparse_dim,dim,grad_coords,out_coords,coalesce", _STRUCTURE_ROWS
)
def test__sparse_log_softmax_backward_data_structure(
    size, sparse_dim, dim, grad_coords, out_coords, coalesce, dtype
):
    grad_output = _coo_on_support(
        size, sparse_dim, grad_coords, dtype, _DEFAULT_RANGE, coalesce
    )
    output = _coo_on_support(
        size, sparse_dim, out_coords, dtype, _DEFAULT_RANGE, coalesce
    )
    self_ = _coo_on_support(
        size, sparse_dim, grad_coords, dtype, _DEFAULT_RANGE, coalesce
    )

    expected_device = self_.device

    ref_out = torch.ops.aten._sparse_log_softmax_backward_data(
        _to_reference(grad_output),
        _to_reference(output),
        dim,
        _to_reference(self_),
    )
    res_out = flag_gems._sparse_log_softmax_backward_data(
        grad_output, output, dim, self_
    )

    # Operands with different supports must still yield the native sparse result,
    # support size included.
    _assert_result_close(res_out, ref_out, expected_device=expected_device)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype,scenario,slot", _SPECIAL_ROWS)
def test__sparse_log_softmax_backward_data_special_values(dtype, scenario, slot):
    # One normalization group of five entries along dim 0, so nan/inf take part in the
    # within-group reduction.
    size, sparse_dim, dim = (8,), 1, 0
    support = (0, 1, 2, 4, 6)
    payload = tu.make_special_input(dtype, scenario)
    unit = torch.ones_like(payload)

    grad_output = _coo_on_support(
        size,
        sparse_dim,
        support,
        dtype,
        _DEFAULT_RANGE,
        True,
        values=payload if slot == "grad_output" else unit,
    )
    output = _coo_on_support(
        size,
        sparse_dim,
        support,
        dtype,
        _DEFAULT_RANGE,
        True,
        values=payload if slot == "output" else unit,
    )
    self_ = _coo_on_support(
        size, sparse_dim, support, dtype, _DEFAULT_RANGE, True, values=unit
    )

    expected_device = self_.device

    ref_out = torch.ops.aten._sparse_log_softmax_backward_data(
        _to_reference(grad_output),
        _to_reference(output),
        dim,
        _to_reference(self_),
    )
    res_out = flag_gems._sparse_log_softmax_backward_data(
        grad_output, output, dim, self_
    )

    # Matching nan positions are part of the reference semantics.
    _assert_result_close(res_out, ref_out, expected_device=expected_device)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("coords,coalesce", _OUT_BUFFER_ROWS)
def test__sparse_log_softmax_backward_data_out(coords, coalesce, dtype):
    size, sparse_dim, dim = (6,), 1, 0
    support = (0, 2, 4, 5)

    grad_output = _coo_on_support(
        size, sparse_dim, support, dtype, _DEFAULT_RANGE, True
    )
    output = _coo_on_support(size, sparse_dim, support, dtype, _DEFAULT_RANGE, True)
    self_ = _coo_on_support(size, sparse_dim, support, dtype, _DEFAULT_RANGE, True)
    out_buf = _coo_on_support(size, sparse_dim, coords, dtype, _DEFAULT_RANGE, coalesce)

    expected_device = self_.device

    input_states = tuple(_readonly_state(x) for x in (grad_output, output, self_))
    ref_buf = _to_reference(out_buf)
    grad_before = _to_reference(grad_output)
    output_before = _to_reference(output)
    self_before = _to_reference(self_)

    ref_ret = torch.ops.aten._sparse_log_softmax_backward_data(
        _to_reference(grad_output),
        _to_reference(output),
        dim,
        _to_reference(self_),
        out=ref_buf,
    )
    res_ret = flag_gems._sparse_log_softmax_backward_data(
        grad_output, output, dim, self_, out=out_buf
    )

    # `.out` returns the buffer itself, so the value comparison below covers the buffer.
    assert res_ret is out_buf
    _assert_result_close(res_ret, ref_ret, expected_device=expected_device)
    # The inputs are only read: they must stay bit-identical to the snapshots taken
    # before the call, so this uses the exact comparison rather than a tolerance.
    for inp, state in zip((grad_output, output, self_), input_states):
        _assert_readonly(inp, state)
    _assert_result_equal(grad_output, grad_before, expected_device=expected_device)
    _assert_result_equal(output, output_before, expected_device=expected_device)
    _assert_result_equal(self_, self_before, expected_device=expected_device)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__sparse_log_softmax_backward_data_rejects_dtype(dtype):
    grad_output = _coo_unit((6,), 1, 3, dtype, phase=0)
    output = _coo_unit((6,), 1, 3, dtype, phase=1)
    self_ = _coo_unit((6,), 1, 3, dtype, phase=2)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems._sparse_log_softmax_backward_data(grad_output, output, 0, self_)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("size,dim", _INVALID_DIMS)
def test__sparse_log_softmax_backward_data_rejects_dim(size, dim):
    shape = tuple(size)
    nnz = _nnz_for(shape)
    grad_output = _coo(
        shape, len(shape), -1, nnz, torch.float32, _DEFAULT_RANGE, phase=0
    )
    output = _coo(shape, len(shape), -1, nnz, torch.float32, _DEFAULT_RANGE, phase=1)
    self_ = _coo(shape, len(shape), -1, nnz, torch.float32, _DEFAULT_RANGE, phase=2)
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._sparse_log_softmax_backward_data(grad_output, output, dim, self_)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_log_softmax_backward_data_rejects_dense_input(dtype):
    grad_output = tu.make_input(dtype, (6,), _DEFAULT_RANGE)
    output = _coo((6,), 1, -1, 3, dtype, _DEFAULT_RANGE, phase=1)
    self_ = _coo((6,), 1, -1, 3, dtype, _DEFAULT_RANGE, phase=2)
    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems._sparse_log_softmax_backward_data(grad_output, output, 0, self_)


@pytest.mark.sparse_log_softmax_backward_data
@pytest.mark.parametrize("grad_dtype,output_dtype", _MIXED_DTYPE_ROWS)
def test__sparse_log_softmax_backward_data_rejects_mixed_dtype(
    grad_dtype, output_dtype
):
    grad_output = _coo((6,), 1, 0, 3, grad_dtype, _DEFAULT_RANGE, phase=0)
    output = _coo((6,), 1, 0, 3, output_dtype, _DEFAULT_RANGE, phase=1)
    self_ = _coo((6,), 1, 0, 3, grad_dtype, _DEFAULT_RANGE, phase=2)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_log_softmax_backward_data(grad_output, output, 0, self_)


def test_sparse_log_softmax_backward_data_dispatch():
    """Smoke-test the dispatcher path (base and .out overloads)."""
    idx = torch.tensor([[0, 1], [0, 1]], device=flag_gems.device)
    vals = torch.randn(2, 4, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(idx, vals, (8, 8, 4), device=flag_gems.device)
    out = torch.ops.aten._sparse_log_softmax(inp, 1)
    grad = torch.sparse_coo_tensor(
        idx,
        torch.randn(2, 4, device=flag_gems.device),
        (8, 8, 4),
        device=flag_gems.device,
    )
    ref = torch.ops.aten._sparse_log_softmax_backward_data(
        utils.to_reference(grad), utils.to_reference(out), 1, utils.to_reference(inp)
    )
    o = torch.empty_like(ref)
    res = flag_gems._sparse_log_softmax_backward_data(grad, out, 1, inp)
    res_out = flag_gems._sparse_log_softmax_backward_data_out(grad, out, 1, inp, out=o)
    assert res.shape == ref.shape and res.is_coalesced() == ref.is_coalesced()
    assert res_out is o
    assert torch.equal(res._indices().cpu(), ref._indices().cpu())
    assert torch.equal(res_out._values().cpu(), ref._values().cpu())
