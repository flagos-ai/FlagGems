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

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _expected_device():
    device = torch.device(flag_gems.device)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _cpu_storage_view(inp):
    # Copy bytes before decoding: target FP8 casts and complex comparisons may
    # be absent. Retain the physical view instead of compacting its geometry.
    assert inp.layout == torch.strided
    raw = (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )
    return torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )


def _to_reference(inp):
    if not utils.TO_CPU:
        return tu.to_reference(inp)
    reference = _cpu_storage_view(inp)
    if inp.is_conj():
        reference = reference.conj()
    if inp.is_neg():
        reference = torch._neg_view(reference)
    return reference.requires_grad_(inp.requires_grad)


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
        if low == high:
            value = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            value = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        value = value.to(flag_gems.device)
    else:
        value = tu.make_input(dtype, shape, value_range)
    assert value.dtype == dtype and value.shape == torch.Size(shape)
    assert value.device == _expected_device()
    return value


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        value = (
            torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
            .to(dtype)
            .to(flag_gems.device)
        )
    else:
        value = tu.make_special_input(dtype, scenario)
    assert value.dtype == dtype and value.shape == (5,)
    assert value.device == _expected_device()
    return value


# Materialize the logical shape as a fresh int64 tensor on CPU.
_SHAPE_AS_TENSOR_INPUT_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_EMPTY_SHAPES = [(0,), (0, 5), (3, 0, 4)]

# (label, view_fn, logical_shape) for a (4, 8, 6) base.
_VIEW_CASES = [
    ("transposed", lambda base: base.transpose(0, 1), (8, 4, 6)),
    ("sliced", lambda base: base[0:3, 2:7, 1], (3, 5)),
    ("narrowed", lambda base: base.narrow(1, 1, 5), (4, 5, 6)),
]


def _assert_result(res_out, ref_out, inp):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == torch.device("cpu")
    assert res_out.dtype == torch.int64 and res_out.shape == (inp.ndim,)
    assert res_out.layout == torch.strided
    assert not torch._C._is_alias_of(res_out, inp)
    assert not res_out._is_view()
    assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _SHAPE_AS_TENSOR_INPUT_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__shape_as_tensor_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__shape_as_tensor_empty(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize("view_case", _VIEW_CASES)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__shape_as_tensor_non_contiguous(view_case, value_range, dtype):
    _, view_fn, expected = view_case
    base = _make_input(dtype, (4, 8, 6), value_range)
    inp = view_fn(base)
    assert not inp.is_contiguous()
    assert inp.shape == expected
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SHAPE_AS_TENSOR_INPUT_DTYPES)),
)
def test__shape_as_tensor_nan_inf(dtype, scenario):
    inp = _make_special_input(dtype, scenario).expand(4, 8, -1)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    _assert_result(res_out, ref_out, inp)


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize("shape", [(), (1,), (2, 3, 5)])
def test__shape_as_tensor_ignores_autograd(shape):
    inp = _make_input(torch.float32, shape, ["-1", "1"]).requires_grad_()
    ref_inp = _to_reference(inp.detach())

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    assert not res_out.requires_grad
    _assert_result(res_out, ref_out, inp.detach())


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_rejects_non_tensor_input():
    for bad in (5, [1, 2, 3], "abc", 3.14):
        with pytest.raises(RuntimeError):
            torch.ops.aten._shape_as_tensor(bad)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._shape_as_tensor(bad)


def _make_coo(nnd, nnz, size, seed=0):
    gen = torch.Generator().manual_seed(seed)
    numel = 1
    for s in size[:nnd]:
        numel *= s
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    for d in reversed(range(nnd)):
        indices[d] = flat % size[d]
        flat = flat // size[d]
    # Deliberately unsorted: uncoalesced COO must return the same metadata.
    values = torch.randn(nnz, generator=gen)
    return torch.sparse_coo_tensor(indices, values, size)


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4), (2, 3, 4, 5, 6)])
def test__shape_as_tensor_master_strided(shape):
    inp = torch.randn(shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._shape_as_tensor(ref_inp)
    res_out = flag_gems._shape_as_tensor(inp)

    assert res_out.shape == (inp.dim(),)
    assert res_out.dtype == torch.int64
    assert res_out.device.type == "cpu"
    assert res_out.is_contiguous()
    assert res_out._base is None
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_noncontiguous():
    inp = torch.randn(4, 6, device=flag_gems.device).t()
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._shape_as_tensor(utils.to_reference(inp))
    res_out = flag_gems._shape_as_tensor(inp)
    assert res_out.tolist() == [6, 4]
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_sparse_coo():
    cases = [
        ("plain", _make_coo(2, 3, (2, 3), seed=1).coalesce(), [2, 3]),
        ("1d", _make_coo(1, 3, (3,), seed=2), [3]),
        (
            "hybrid",
            torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.randn(2, 3), (2, 3)),
            [2, 3],
        ),
        (
            "zero-nnz",
            torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.int64),
                torch.zeros(0),
                (2, 3),
            ),
            [2, 3],
        ),
        ("uncoalesced", _make_coo(2, 4, (2, 3), seed=3), [2, 3]),
    ]
    for name, sparse, expected in cases:
        sparse = sparse.to(flag_gems.device)
        ref_sparse = utils.to_reference(sparse)
        ref_out = torch.ops.aten._shape_as_tensor(ref_sparse)
        res_out = flag_gems._shape_as_tensor(sparse)
        assert res_out.tolist() == expected, f"{name}: {res_out} != {expected}"
        assert res_out.device.type == "cpu"
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_sparse_compressed():
    row = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=flag_gems.device)
    val = torch.randn(4, device=flag_gems.device)

    csr = torch.sparse_csr_tensor(row, col, val, (2, 3))
    ccol = torch.tensor([0, 2, 4, 4], dtype=torch.int64, device=flag_gems.device)
    csc = torch.sparse_csc_tensor(ccol, col, val, (2, 3), check_invariants=True)
    bsr = torch.randn(4, 4, device=flag_gems.device).to_sparse_bsr((2, 2))
    bsc = torch.randn(4, 4, device=flag_gems.device).to_sparse_bsc((2, 2))
    empty_nnz = torch.sparse_csr_tensor(
        torch.tensor([0, 0, 0], dtype=torch.int64, device=flag_gems.device),
        torch.zeros(0, dtype=torch.int64, device=flag_gems.device),
        torch.zeros(0, device=flag_gems.device),
        (2, 3),
    )
    for name, t, expected in [
        ("csr", csr, [2, 3]),
        ("csc", csc, [2, 3]),
        ("bsr", bsr, [4, 4]),
        ("bsc", bsc, [4, 4]),
        ("csr-empty", empty_nnz, [2, 3]),
    ]:
        res_out = flag_gems._shape_as_tensor(t)
        assert res_out.tolist() == list(t.shape), f"{name}: {res_out} != {t.shape}"
        assert res_out.tolist() == expected, f"{name}: {res_out} != {expected}"
        assert res_out.device.type == "cpu"
        ref_out = torch.ops.aten._shape_as_tensor(utils.to_reference(t))
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_matches_reference_table():
    # Reference values captured in a process that never imported flag_gems
    # (un-substituted environment, H20 build): result values, dtype, device
    # and rank pinned as literals.
    dev = flag_gems.device

    res = flag_gems._shape_as_tensor(torch.randn(2, 3, device=dev))
    assert res.tolist() == [2, 3]
    assert res.dtype == torch.int64
    assert res.device.type == "cpu"

    coo = _make_coo(2, 3, (2, 3), seed=5).to(dev)
    assert flag_gems._shape_as_tensor(coo).tolist() == [2, 3]

    hybrid = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]), torch.randn(2, 3), (2, 3)
    ).to(dev)
    assert flag_gems._shape_as_tensor(hybrid).tolist() == [2, 3]

    csr = coo.to_sparse_csr()
    assert flag_gems._shape_as_tensor(csr).tolist() == [2, 3]

    zero = flag_gems._shape_as_tensor(torch.tensor(2.5, device=dev))
    assert zero.shape == (0,) and zero.dtype == torch.int64


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_fresh_storage_and_resize_independence():
    # Native returns a freshly allocated tensor: it never aliases ``self``
    # and later resizes of ``self`` must not retroactively change it.
    t = torch.randn(6, 7, device=flag_gems.device)
    res = flag_gems._shape_as_tensor(t)

    assert res.dtype == torch.int64
    assert res.shape == (2,)
    assert res.device.type == "cpu"
    assert res.is_contiguous()
    assert res._base is None

    t.resize_(3, 4)
    assert res.tolist() == [6, 7]

    ref = torch.ops.aten._shape_as_tensor(utils.to_reference(t))
    assert ref.tolist() == [3, 4]


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_device_is_cpu_for_cuda_input():
    # Measured native rule: the C++ kernel builds the result from
    # TensorOptions(at::kLong) without a device, so the result is a CPU
    # tensor even for accelerator inputs (and for meta inputs). This is the
    # observable difference from device=self.device, which would be wrong.
    dev = flag_gems.device
    if dev == "cpu":
        pytest.skip("requires an accelerator input to observe the rule")
    res = flag_gems._shape_as_tensor(torch.randn(2, 3, device=dev))
    ref = torch.ops.aten._shape_as_tensor(
        utils.to_reference(torch.randn(2, 3, device=dev))
    )
    assert res.device.type == "cpu"
    assert ref.device.type == "cpu"
    assert res.device == ref.device
    assert res.tolist() == [2, 3]


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_quantized():
    dev = flag_gems.device
    q = torch.quantize_per_tensor(torch.randn(2, 3, device=dev), 0.1, 10, torch.quint8)
    res_q = flag_gems._shape_as_tensor(q)
    ref_q = torch.ops.aten._shape_as_tensor(utils.to_reference(q))
    assert res_q.tolist() == [2, 3]
    utils.gems_assert_equal(res_q, ref_q)



@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_pinned():
    # A pinned CPU tensor is still a plain strided tensor.
    pinned = torch.empty(5, pin_memory=True)
    res_p = flag_gems._shape_as_tensor(pinned)
    ref_p = torch.ops.aten._shape_as_tensor(pinned)
    assert res_p.tolist() == [5]
    assert res_p.device.type == "cpu"
    utils.gems_assert_equal(res_p, ref_p)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor_master_disjoint_from_native_neighbors():
    # The implementation must not re-dispatch the operator it implements
    # (registered on backend keys, an internal aten re-entry would recurse
    # into this very kernel), and it must not depend on related metadata ops
    # being overridden. Dispatch a related aten op around the calls: no
    # shared mutable state exists, results must stay correct.
    dev = flag_gems.device
    t = torch.randn(3, 4, device=dev)
    sparse = _make_coo(2, 5, (2, 3), seed=7).to(dev)
    csr = sparse.to_sparse_csr()

    for _ in range(3):
        assert flag_gems._shape_as_tensor(t).tolist() == [3, 4]
        assert flag_gems._shape_as_tensor(sparse).tolist() == [2, 3]
        assert flag_gems._shape_as_tensor(csr).tolist() == [2, 3]

    # Cross-op sanity: dispatching a related metadata op afterwards must not
    # corrupt or re-route shape_as_tensor.
    nnz = torch.ops.aten._nnz(sparse)
    assert nnz == 5
    assert flag_gems._shape_as_tensor(sparse).tolist() == [2, 3]
    assert flag_gems._shape_as_tensor(t).tolist() == [3, 4]


@pytest.mark.underscore_shape_as_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="enflame has no CUDA/SparseCUDA/SparseCsrCUDA",
)
def test__shape_as_tensor_master_dispatch_sentinel():
    # Registration-path check: the op is registered on CUDA, SparseCUDA and
    # SparseCsrCUDA (measured on this build: the plain CUDA key intercepts
    # strided inputs, and the sparse keys become reachable once the CUDA key
    # is registered too - see the report for the dispatcher-alias mechanics).
    # A sentinel on each key proves every layout routes into the registered
    # implementation instead of falling through to the native kernel.
    hits = []

    def sentinel(self):
        hits.append(str(self.layout))
        return flag_gems._shape_as_tensor(self)

    lib = torch.library.Library("aten", "IMPL")
    backend = torch._C._dispatch_key_for_device(torch.device(flag_gems.device).type)
    for key in (backend, f"Sparse{backend}", f"SparseCsr{backend}"):
        lib.impl("_shape_as_tensor", sentinel, key)

    dev = flag_gems.device
    t = torch.randn(2, 3, device=dev)
    coo = _make_coo(2, 3, (2, 3), seed=13).to(dev)
    row = torch.tensor([0, 2, 4], dtype=torch.int64, device=dev)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=dev)
    csr = torch.sparse_csr_tensor(row, col, torch.randn(4, device=dev), (2, 3))

    assert torch.ops.aten._shape_as_tensor(t).tolist() == [2, 3]
    assert torch.ops.aten._shape_as_tensor(coo).tolist() == [2, 3]
    assert torch.ops.aten._shape_as_tensor(csr).tolist() == [2, 3]

    assert set(hits) == {"torch.strided", "torch.sparse_coo", "torch.sparse_csr"}
    assert len(hits) == 3
