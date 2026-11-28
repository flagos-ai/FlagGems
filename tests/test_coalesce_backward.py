# Copyright 2026, The FlagOS Contributors.
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

# Tests call flag_gems.coalesce_backward directly instead of the
# flag_gems.use_gems() context manager: the CI kernelgen-test rules forbid a
# real use_gems() call from test files, and the wrapper is what the PR exposes.

# Value dtypes the restore streams; mirrors the sibling _coalesce sweep
# (floats, signed/unsigned integers, bool). The FLOAT_DTYPES / INT_DTYPES
# constants are expanded here because the sparse fixtures must build both the
# indices and the values with real torch.dtype objects.
_COALESCE_BACKWARD_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    *([torch.float64] if utils.fp64_is_supported else []),
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

# (shape, nnz, seed) descriptors with duplicate coordinates: coalescing merges
# the duplicates, so the backward exercises the key-match restore path.
# Small coordinate spaces keep the duplicate ratio high.
_DUPLICATE_CASES = [
    ((8,), 20, 2026),
    ((4, 4), 20, 2027),
    ((5, 5), 30, 2028),
    ((8, 8), 80, 2029),
    ((16, 16), 300, 2030),
    ((2, 3, 4), 28, 2031),
    ((3, 5, 7), 120, 2032),
    ((4, 8, 16), 600, 2033),
    ((2, 3, 4, 5), 300, 2034),
    ((2, 3, 4, 5, 6), 1000, 2035),
]

# (shape, nnz, seed) descriptors with distinct coordinates: the coalesced
# output keeps every coordinate, so the backward takes the identity fast path.
_UNIQUE_CASES = [
    ((8,), 8, 4026),
    ((4, 4), 16, 4027),
    ((5, 5), 25, 4028),
    ((8, 8), 64, 4029),
    ((2, 3, 4), 24, 4030),
    ((3, 5, 7), 105, 4031),
    ((2, 3, 4, 5), 120, 4032),
]

# Already-coalesced inputs must be restored unchanged (identity fast path).
_COALESCED_CASES = [
    ((5, 5), 30, 5026),
    ((3, 5, 7), 120, 5027),
    ((2, 3, 4, 5), 300, 5028),
]


def _make_values(nnz, dtype, gen):
    """Seeded values for one dtype; floats stay non-negative so CPU/CUDA
    accumulations remain comparable."""
    if dtype == torch.bool:
        return torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    if dtype.is_floating_point:
        return torch.rand(nnz, dtype=torch.float64, generator=gen).to(dtype)
    return torch.randint(0, 6, (nnz,), dtype=torch.int64, generator=gen).to(dtype)


def _make_sparse_input(shape, nnz, dtype, seed):
    """Seeded sparse COO fixture; coordinates may repeat within the shape."""
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    return torch.sparse_coo_tensor(
        indices, _make_values(nnz, dtype, gen), shape, device=flag_gems.device
    )


def _make_unique_sparse_input(shape, nnz, dtype, seed):
    """Sparse COO fixture with distinct coordinates (identity-path input)."""
    numel = 1
    for dim in shape:
        numel *= dim
    assert nnz <= numel
    gen = torch.Generator("cpu").manual_seed(seed)
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((len(shape), nnz), dtype=torch.long)
    for d in reversed(range(len(shape))):
        indices[d] = flat % shape[d]
        flat = flat // shape[d]
    return torch.sparse_coo_tensor(
        indices.contiguous(),
        _make_values(nnz, dtype, gen),
        shape,
        device=flag_gems.device,
    )


def _make_grad_output(coalesced, seed):
    """A seeded upstream gradient on the coalesced output's coordinates."""
    gen = torch.Generator("cpu").manual_seed(seed)
    nnz = coalesced._nnz()
    values = _make_values(nnz, coalesced._values().dtype, gen)
    return torch.sparse_coo_tensor(
        coalesced._indices(), values, coalesced.shape, device=coalesced.device
    )


def _to_device(sparse):
    """Rebuild the sparse gradient on the device under test."""
    return torch.sparse_coo_tensor(
        sparse._indices().to(flag_gems.device),
        sparse._values().to(flag_gems.device),
        sparse.shape,
        device=flag_gems.device,
    )


def _reference_backward(grad_output, self_):
    """Native-semantic reference for the merged-group gradient broadcast.

    ``_coalesce`` is differentiable (derivatives.yaml ``_coalesce ... self:
    grad``) and its autograd derivative routes every stored input position to
    the upstream value of the coalesced output group that merged it — a plain
    gather, no arithmetic, so the value dtype need not be differentiable.
    This computes that mapping directly: rank each input coordinate in the
    (ascending, unique) output key list with ``torch.searchsorted`` and copy
    the matched upstream value (zero where the forward dropped the
    coordinate, which coalesce never does but the guard keeps for safety).
    Bit-exact for every payload dtype, float or not.
    """
    out_idx = grad_output._indices()
    in_idx = self_._indices()
    n = self_._nnz()
    m = grad_output._nnz()
    shape = self_.shape
    # Row-major linear keys, matching the kernel's stride order.
    keys_out = out_idx[0].to(torch.int64)
    keys_in = in_idx[0].to(torch.int64)
    for d in range(1, out_idx.shape[0]):
        keys_out = keys_out * shape[d] + out_idx[d].to(torch.int64)
        keys_in = keys_in * shape[d] + in_idx[d].to(torch.int64)
    rank = torch.searchsorted(keys_out, keys_in)
    hit = (rank < m) & (keys_out[rank.clamp(max=m - 1)] == keys_in)
    picked = torch.where(
        hit,
        grad_output._values()[rank.clamp(max=m - 1)],
        torch.zeros(n, dtype=grad_output._values().dtype, device=grad_output.device),
    )
    return torch.sparse_coo_tensor(
        self_._indices().clone(), picked, self_.shape, is_coalesced=self_.is_coalesced()
    )


def _assert_restore(res_out, ref_out, dtype):
    """The restored gradient keeps the input index table (order included) and
    carries the upstream value of every merged group."""
    assert res_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    assert res_out._nnz() == ref_out._nnz()
    # Indices must match exactly, position by position (input table kept).
    utils.gems_assert_equal(res_out._indices(), ref_out._indices())
    # Values: tolerance for float payloads, exact for int/bool payloads.
    if dtype.is_floating_point:
        utils.gems_assert_close(res_out._values(), ref_out._values(), dtype)
    else:
        utils.gems_assert_equal(res_out._values(), ref_out._values())


@pytest.mark.coalesce_backward
@pytest.mark.parametrize("case", _DUPLICATE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_BACKWARD_DTYPES)
def test_coalesce_backward_duplicate_input(case, dtype):
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape, nnz, seed = case
    inp = _make_sparse_input(shape, nnz, dtype, seed)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_grad_output = _make_grad_output(torch.ops.aten.coalesce(ref_inp), seed + 1000)
    ref_out = _reference_backward(ref_grad_output, ref_inp)

    res_out = flag_gems.coalesce_backward(_to_device(ref_grad_output), inp)
    _assert_restore(res_out, ref_out, dtype)
    # The restore must not mutate its inputs.
    assert not inp.is_coalesced()


@pytest.mark.coalesce_backward
@pytest.mark.parametrize("case", _UNIQUE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_BACKWARD_DTYPES)
def test_coalesce_backward_identity_input(case, dtype):
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape, nnz, seed = case
    inp = _make_unique_sparse_input(shape, nnz, dtype, seed)
    ref_inp = utils.to_reference(inp)

    ref_grad_output = _make_grad_output(torch.ops.aten.coalesce(ref_inp), seed + 1000)
    ref_out = _reference_backward(ref_grad_output, ref_inp)

    grad_output = _to_device(ref_grad_output)
    res_out = flag_gems.coalesce_backward(grad_output, inp)
    _assert_restore(res_out, ref_out, dtype)
    # Duplicate-free but uncoalesced: the coalesced forward sorts the
    # coordinates, so the restored values are the upstream values permuted
    # back into the input's storage order - not a plain copy.  The exact
    # check against the key-rank reference above covers the mapping.


@pytest.mark.coalesce_backward
@pytest.mark.parametrize("case", _COALESCED_CASES)
# One float and one int payload cover the already-coalesced identity fast path;
# the full dtype sweep runs on the duplicate and unique suites above.
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_coalesce_backward_coalesced_input(case, dtype):
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape, nnz, seed = case
    inp = _make_sparse_input(shape, nnz, dtype, seed).coalesce()
    assert inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_grad_output = _make_grad_output(ref_inp, seed + 1000)
    ref_out = _reference_backward(ref_grad_output, ref_inp)

    grad_output = _to_device(ref_grad_output)
    res_out = flag_gems.coalesce_backward(grad_output, inp)
    _assert_restore(res_out, ref_out, dtype)
    # Identity fast path: the restored values equal the upstream values.
    utils.gems_assert_equal(res_out._values(), grad_output._values())


@pytest.mark.coalesce_backward
@pytest.mark.parametrize(
    "case",
    [((2, 3, 4, 5, 6), 500, 6026), ((3, 4, 5, 6, 7), 800, 6027)],
)
# float32 exercises the rank-5 stride recovery; the full dtype sweep runs above.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_coalesce_backward_duplicate_rank5(case, dtype):
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape, nnz, seed = case
    inp = _make_sparse_input(shape, nnz, dtype, seed)
    assert not inp.is_coalesced()
    ref_inp = utils.to_reference(inp)

    ref_grad_output = _make_grad_output(torch.ops.aten.coalesce(ref_inp), seed + 1000)
    ref_out = _reference_backward(ref_grad_output, ref_inp)

    res_out = flag_gems.coalesce_backward(_to_device(ref_grad_output), inp)
    _assert_restore(res_out, ref_out, dtype)


@pytest.mark.coalesce_backward
def test_coalesce_backward_empty_grad():
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape, nnz, seed = (4, 4), 10, 7026
    inp = _make_sparse_input(shape, nnz, torch.float32, seed)
    empty_idx = torch.empty((len(shape), 0), dtype=torch.long, device=flag_gems.device)
    empty_val = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.sparse_coo_tensor(empty_idx, empty_val, shape)
    res_out = flag_gems.coalesce_backward(grad_output, inp)
    assert res_out.layout == torch.sparse_coo
    assert res_out._nnz() == nnz
    # Empty upstream gradient restores zeros on the input coordinates.
    utils.gems_assert_equal(res_out._values(), torch.zeros_like(inp._values()))


@pytest.mark.coalesce_backward
def test_coalesce_backward_rejects_layout_and_shape_mismatch():
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # A dense grad_output must be rejected (layout check).
    inp = _make_sparse_input((4, 4), 8, torch.float32, 8026)
    with pytest.raises(RuntimeError):
        flag_gems.coalesce_backward(torch.randn(4, 4, device=flag_gems.device), inp)
    # A grad_output built on a different shape must be rejected (shape check).
    grad_output = _make_sparse_input((4, 4), 8, torch.float32, 8027).coalesce()
    other = _make_sparse_input((4, 8), 8, torch.float32, 8028)
    with pytest.raises(RuntimeError):
        flag_gems.coalesce_backward(grad_output, other)


@pytest.mark.coalesce_backward
def test_coalesce_backward_rejects_fp8_values():
    # This suite is validated on the device path; --ref=cpu moves the native
    # reference to CPU and the device-only restore is skipped there.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = torch.zeros(3, dtype=torch.float8_e4m3fn, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    grad_output = torch.sparse_coo_tensor(
        indices, values.clone(), (4,), device=flag_gems.device
    )
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems.coalesce_backward(grad_output, inp)
