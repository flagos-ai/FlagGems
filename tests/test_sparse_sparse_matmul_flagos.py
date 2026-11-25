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

pytestmark = pytest.mark.sparse_sparse_matmul

_REPRO = [(0, 1, 2), (1, 0, 3)]
_ATEN = torch.ops.aten._sparse_sparse_matmul


@pytest.fixture(scope="module")
def flagos_device():
    if (
        flag_gems.vendor_name != "hygon"
        or torch._C._get_privateuse1_backend_name() != "flagos"
        or not torch.flagos.is_available()
    ):
        pytest.skip("SparsePrivateUse1 regression requires Hygon FlagOS")

    lib = torch.library.Library("aten", "IMPL")
    previous = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=lib,
            include=["_sparse_sparse_matmul", "_sparse_sparse_matmul_out"],
        )
        for op in ("aten::_sparse_sparse_matmul", "aten::_sparse_sparse_matmul.out"):
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                op, "SparsePrivateUse1"
            )
        yield torch.device("flagos", 0)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous


def _coo(indices, values, shape, coalesced):
    opts = {} if coalesced is None else {"is_coalesced": coalesced}
    return torch.sparse_coo_tensor(indices, values, shape, **opts)


def _from_entries(shape, entries, device, coalesced=None, dtype=torch.float32):
    indices = torch.tensor([[r, c] for r, c, _ in entries], dtype=torch.int64)
    indices = indices.reshape(-1, 2).T.contiguous().to(device)
    values = torch.tensor([v for _, _, v in entries], dtype=dtype).to(device)
    return _coo(indices, values, shape, coalesced)


def _assert_coo(actual, reference, device):
    assert actual.device == torch.device(device)
    assert actual.layout == reference.layout == torch.sparse_coo
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert actual.sparse_dim() == reference.sparse_dim() == 2
    assert actual.dense_dim() == reference.dense_dim() == 0
    assert actual.is_coalesced() and reference.is_coalesced()
    assert actual._nnz() == reference._nnz()
    # Compare raw coordinates, not dense values: losing stored zeros is a bug.
    torch.testing.assert_close(actual._indices().cpu(), reference._indices())
    torch.testing.assert_close(
        actual._values().cpu(), reference._values(), rtol=0, atol=0, equal_nan=True
    )


def _check_product(a_shape, a_entries, b_shape, b_entries, device, coalesced):
    a = _from_entries(a_shape, a_entries, device, coalesced)
    b = _from_entries(b_shape, b_entries, device, coalesced)
    ref_a = _from_entries(a_shape, a_entries, "cpu", coalesced)
    ref_b = _from_entries(b_shape, b_entries, "cpu", coalesced)
    _assert_coo(torch.sparse.mm(a, b), torch.sparse.mm(ref_a, ref_b), device)


def test_flagos_sparse_sparse_mm_issue_852(flagos_device):
    a = _from_entries((2, 2), _REPRO, flagos_device)
    b = _from_entries((2, 2), _REPRO, flagos_device)
    cpu = _from_entries((2, 2), _REPRO, "cpu")
    assert not a.is_coalesced() and not b.is_coalesced()
    expected = torch.sparse.mm(cpu, cpu)
    torch.testing.assert_close(expected._indices(), torch.tensor([[0, 1], [0, 1]]))
    torch.testing.assert_close(expected._values(), torch.tensor([6.0, 6.0]))
    _assert_coo(torch.sparse.mm(a, b), expected, flagos_device)


def _strided_coo(device, shape, entries, coalesced, value_layout):
    count = len(entries)
    index_storage = torch.full((count, 4), -1, dtype=torch.int64)
    index_storage[:, 1:3] = torch.tensor([[r, c] for r, c, _ in entries])
    indices = index_storage.to(device)[:, 1:3].T
    if value_layout == "expanded":
        values = torch.tensor([2.0]).to(device).expand(count)
    else:
        value_storage = torch.full((2 * count,), 99.0)
        value_storage[1::2] = torch.tensor(
            [v for _, _, v in entries], dtype=torch.float32
        )
        values = value_storage.to(device)[1::2]
    result = _coo(indices, values, shape, coalesced)
    assert result._indices().stride() == (1, 4)
    assert result._values().stride() == (0 if value_layout == "expanded" else 2,)
    return result


@pytest.mark.parametrize("coalesced", [True, False], ids=["coalesced", "duplicates"])
@pytest.mark.parametrize("value_layout", ["sliced", "expanded"])
def test_flagos_sparse_sparse_mm_strided(coalesced, value_layout, flagos_device):
    a = [(0, 1, 2), (1, 0, 3), (1, 1, -1)]
    b = [(0, 1, 3), (1, 0, 4), (1, 2, 2)]
    if not coalesced:
        a = a[::-1] + [(0, 1, -2)]
        b = b[::-1] + [(1, 2, -1)]
    operands = [
        (
            _strided_coo(device, (2, 2), a, coalesced, value_layout),
            _strided_coo(device, (2, 3), b, coalesced, value_layout),
        )
        for device in ("cpu", flagos_device)
    ]
    ref, actual = (torch.sparse.mm(*pair) for pair in operands)
    _assert_coo(actual, ref, flagos_device)


@pytest.mark.parametrize("nnz", [255, 256, 257, 1023, 1024, 1025, 4097])
def test_flagos_sparse_sparse_mm_merge_boundaries(nnz, flagos_device):
    a = [(j % 17, j % 13, j % 7 - 3) for j in reversed(range(nnz))]
    b = [(j, j % 5, j % 5 - 2) for j in range(13)]
    _check_product((17, 13), a, (13, 5), b, flagos_device, False)


def test_flagos_sparse_sparse_mm_int64_key(flagos_device):
    # Last flattened input coordinate is above INT32_MAX, with a small output.
    a = [(65535, 65535, 1), (1, 0, -2), (65535, 65535, 4)]
    b = [(65535, 0, 3), (0, 0, 7)]
    _check_product((65536, 65536), a, (65536, 1), b, flagos_device, False)


@pytest.mark.parametrize("width", [255, 256, 257])
def test_flagos_sparse_sparse_mm_fanout(width, flagos_device):
    a = [(0, 1, 2)]
    b = [(1, j, j % 9 - 4) for j in range(width)]
    _check_product((2, 2), a, (2, width), b, flagos_device, True)


@pytest.mark.parametrize("size", [1023, 1024, 1025])
@pytest.mark.parametrize("boundary", ["cols", "inner", "rows", "rhs-nnz"])
def test_flagos_sparse_sparse_mm_scan_boundaries(size, boundary, flagos_device):
    if boundary == "cols":
        a_shape, b_shape = (3, 2), (2, size)
        a = [(0, 0, 2), (1, 1, 3)]
        b = [(0, 0, 2), (0, size - 1, 7), (1, size - 1, -8)]
    elif boundary == "inner":
        a_shape, b_shape = (1, size), (size, 3)
        a = [(0, 0, 1), (0, size - 1, 9)]
        b = [(0, 0, 1), (size - 1, 1, -2)]
    elif boundary == "rows":
        a_shape, b_shape = (size, 1), (1, 3)
        a = [(0, 0, 1), (size - 1, 0, 9)]
        b = [(0, 0, -1), (0, 2, 2)]
    else:
        a_shape, b_shape = (2, 61), (61, 17)
        a = [(0, 0, 1), (0, (size - 1) // 17, -2)]
        b = [(j // 17, j % 17, j % 7 - 3) for j in range(size)]
    _check_product(a_shape, a, b_shape, b, flagos_device, True)


@pytest.mark.parametrize(
    "dtype,delta",
    [
        (torch.float16, 2**-11),
        (torch.bfloat16, 2**-8),
        (torch.float32, 2**-20),
        (torch.float64, 2**-30),
    ],
)
def test_flagos_sparse_sparse_mm_dtype(dtype, delta, flagos_device):
    a = [(0, 0, 1), (0, 0, delta), (0, 1, 1)]
    b = [(0, 0, 1), (1, 0, -1)]
    cpu_a = _from_entries((1, 2), a, "cpu", dtype=dtype)
    cpu_b = _from_entries((2, 1), b, "cpu", dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16):
        # Native CPU SpGEMM lacks these types. Round the duplicate group at
        # input precision, multiply in FP32 on CPU, then cast the result back.
        cpu_a, cpu_b = cpu_a.coalesce().float(), cpu_b.coalesce().float()
    expected = torch.sparse.mm(cpu_a, cpu_b).to(dtype)
    actual = flag_gems._sparse_sparse_matmul(
        _from_entries((1, 2), a, flagos_device, dtype=dtype),
        _from_entries((2, 1), b, flagos_device, dtype=dtype),
    )
    _assert_coo(actual, expected, flagos_device)


def test_flagos_sparse_sparse_mm_nonfinite(flagos_device):
    a = [(0, 0, 0), (1, 1, float("inf")), (2, 2, -float("inf")), (3, 3, float("nan"))]
    b = [(0, 0, float("inf")), (1, 1, 1), (2, 2, 1), (3, 3, 1)]
    _check_product((4, 4), a, (4, 4), b, flagos_device, True)


@pytest.mark.parametrize(
    "a_shape,a,b_shape,b",
    [
        ((0, 2), [], (2, 2), _REPRO),
        ((3, 0), [], (0, 2), []),
        ((2, 2), _REPRO, (2, 0), []),
        ((2, 2), [], (2, 2), _REPRO),
        ((2, 2), _REPRO, (2, 2), []),
        ((0, 0), [], (0, 0), []),
        ((2, 2), [(0, 0, 1)], (2, 2), [(1, 0, 2)]),
    ],
    ids=["zero-m", "zero-k", "zero-n", "empty-lhs", "empty-rhs", "all-zero", "no-join"],
)
def test_flagos_sparse_sparse_mm_empty(a_shape, a, b_shape, b, flagos_device):
    _check_product(a_shape, a, b_shape, b, flagos_device, False)


@pytest.mark.parametrize("mode", ["seeded", "empty", "alias-a", "alias-b", "clear"])
def test_flagos_sparse_sparse_mm_out(mode, flagos_device):
    def run(device):
        a = _from_entries((2, 2), [] if mode == "clear" else _REPRO, device)
        b = _from_entries((2, 2), _REPRO, device)
        if mode == "alias-a":
            out = a
        elif mode == "alias-b":
            out = b
        else:
            entries = [] if mode == "empty" else _REPRO + [(1, 0, -1)]
            out = _from_entries((2, 2), entries, device)
        assert _ATEN.out(a, b, out=out) is out
        return out

    _assert_coo(run(flagos_device), run("cpu"), flagos_device)


@pytest.mark.parametrize(
    "mode,error,match",
    [
        ("shape", RuntimeError, "shapes cannot be multiplied"),
        ("dtype", RuntimeError, "dtype.*does not match"),
        ("integral", NotImplementedError, "sparse_matmul.*not implemented"),
        ("rank", RuntimeError, r"2-D|dim\(\) == 2"),
        ("hybrid", RuntimeError, "scalar (sparse )?values"),
        ("out-dtype", RuntimeError, "out tensor to have dtype"),
    ],
)
def test_flagos_sparse_sparse_mm_invalid(mode, error, match, flagos_device):
    for device in ("cpu", flagos_device):
        dtype = torch.int64 if mode == "integral" else torch.float32
        a = _from_entries((2, 2), _REPRO, device, dtype=dtype)
        b = _from_entries(
            (3, 2) if mode == "shape" else (2, 2),
            _REPRO,
            device,
            dtype=torch.float64 if mode == "dtype" else dtype,
        )
        if mode in ("rank", "hybrid"):
            indices = torch.tensor([[0, 1]]).to(device)
            shape = (2,) if mode == "rank" else (2, 2)
            b = _coo(indices, torch.ones(shape).to(device), shape, False)
        if mode == "out-dtype":
            out = _from_entries((2, 2), [], device, dtype=torch.float64)
            with pytest.raises(error, match=match):
                _ATEN.out(a, b, out=out)
        else:
            with pytest.raises(error, match=match):
                _ATEN.default(a, b)
