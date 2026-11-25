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

pytestmark = pytest.mark.sparse_addmm


@pytest.fixture(scope="module")
def flagos_addmm_device():
    if (
        flag_gems.vendor_name != "hygon"
        or not hasattr(torch, "flagos")
        or not torch.flagos.is_available()
    ):
        pytest.skip("COO SparsePrivateUse1 regression requires Hygon FlagOS")
    lib = torch.library.Library("aten", "IMPL")
    previous = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(lib=lib, include=["addmm", "addmm_out"])
        for op in ("aten::addmm", "aten::addmm.out"):
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                op, "SparsePrivateUse1"
            )
        yield torch.device("flagos", 0)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous


def _coo(indices, values, shape, *, device="cpu", coalesced=False):
    return torch.sparse_coo_tensor(
        indices.to(device), values.to(device), shape, is_coalesced=coalesced
    )


def _assert_close(actual, expected, device, **tolerance):
    assert actual.device == device and actual.layout == torch.strided
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)


def test_flagos_sparse_mm_issue_852(flagos_addmm_device):
    device = flagos_addmm_device
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([2.0, 3.0], dtype=torch.float32)
    lhs = _coo(indices, values, (2, 2), device=device)
    rhs = torch.ones((2, 1), dtype=values.dtype)
    expected = torch.sparse.mm(_coo(indices, values, (2, 2)), rhs)
    torch.testing.assert_close(expected, torch.tensor([[2.0], [3.0]]))
    _assert_close(torch.sparse.mm(lhs, rhs.to(device)), expected, device)


@pytest.mark.parametrize(
    "nnz,n",
    [(e, n) for e in (3, 4, 5) for n in (63, 64, 65)]
    + [(3, n) for n in (255, 256, 257)],
)
def test_flagos_coo_addmm_tiles(nnz, n, flagos_addmm_device):
    device = flagos_addmm_device
    # The last three sizes straddle the flattened write tile on a single row.
    m = 1 if n >= 255 else 4
    indices = torch.tensor([[3, 0, 3, 1, 0], [6, 1, 6, 0, 3]])[:, :nnz].clone()
    indices[0].remainder_(m)
    values = torch.tensor([3.0, -2.0, -1.5, 0.25, 1.0])[:nnz]
    rhs = ((torch.arange(7 * n).reshape(7, n) % 31).float() - 15) / 16
    bias = rhs[0].clone()
    lhs_cpu = _coo(indices, values, (m, 7))
    lhs = _coo(indices, values, lhs_cpu.shape, device=device)
    coeffs = {"alpha": 0.75, "beta": -1.25}
    expected = torch.addmm(bias, lhs_cpu, rhs, **coeffs)
    result = torch.addmm(bias.to(device), lhs, rhs.to(device), **coeffs)
    _assert_close(result, expected, device)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("coalesced", [False, True])
def test_flagos_direct_coo_addmm_dtypes(dtype, coalesced, flagos_addmm_device):
    device = flagos_addmm_device
    if coalesced:
        indices = torch.tensor([[0, 1, 2], [1, 3, 0]])
    else:
        indices = torch.tensor([[2, 0, 2], [3, 1, 3]])
    values = torch.tensor([3.0, 0.5, -2.0], dtype=dtype)
    rhs = torch.arange(28, dtype=dtype).reshape(4, 7) / 8 - 2
    bias = torch.tensor([[1.25], [0.5], [-0.75]], dtype=dtype)
    lhs_cpu = _coo(indices, values, (3, 4), coalesced=coalesced)
    lhs = _coo(indices, values, lhs_cpu.shape, device=device, coalesced=coalesced)
    alpha, beta = 0.5, -1.0
    if dtype == torch.float64:
        alpha, beta = -0.123456789012345, 1.23456789012345
    expected = torch.ops.aten._sparse_addmm(bias, lhs_cpu, rhs, beta=beta, alpha=alpha)
    actual = flag_gems._sparse_addmm(
        bias.to(device), lhs, rhs.to(device), beta=beta, alpha=alpha
    )
    tolerance = {"rtol": 1e-12, "atol": 1e-12} if dtype == torch.float64 else {}
    _assert_close(actual, expected, device, **tolerance)


@pytest.mark.parametrize(
    "m,k,n,nnz",
    [
        (0, 0, 0, 0),
        (0, 3, 2, 0),
        (2, 0, 3, 0),
        (2, 3, 4, 0),
        (2, 3, 0, 0),
        (2, 3, 0, 1),
    ],
)
def test_flagos_coo_mm_empty(m, k, n, nnz, flagos_addmm_device):
    device = flagos_addmm_device
    indices = torch.zeros((2, nnz), dtype=torch.int64)
    values = torch.ones((nnz,))
    rhs = torch.ones((k, n))
    expected = torch.sparse.mm(_coo(indices, values, (m, k)), rhs)
    actual = torch.sparse.mm(
        _coo(indices, values, (m, k), device=device), rhs.to(device)
    )
    _assert_close(actual, expected, device)


@pytest.mark.parametrize(
    "nnz,alpha,beta,bias_value",
    [(0, 1, 0, float("nan")), (1, 1, 0, float("nan")), (1, 0, -2, 1.5), (1, 0, 0, 1.5)],
)
def test_flagos_coo_addmm_scales(nnz, alpha, beta, bias_value, flagos_addmm_device):
    device = flagos_addmm_device
    indices = torch.zeros((2, nnz), dtype=torch.int64)
    values = torch.full((nnz,), 2.0)
    bias = torch.full((2, 1), bias_value)
    rhs = torch.ones((3, 4))
    expected = torch.addmm(
        bias, _coo(indices, values, (2, 3)), rhs, beta=beta, alpha=alpha
    )
    actual = torch.addmm(
        bias.to(device),
        _coo(indices, values, (2, 3), device=device),
        rhs.to(device),
        beta=beta,
        alpha=alpha,
    )
    _assert_close(actual, expected, device, equal_nan=True)


def _strided_coo_addmm_inputs(device):
    index_storage = torch.tensor([[9, 9], [3, 6], [9, 9], [0, 1], [9, 9], [3, 6]]).to(
        device
    )
    value_storage = torch.tensor([99.0, 1.0, 99.0, -2.0, 99.0, 3.0]).to(device)
    indices, values = index_storage.T[:, 1::2], value_storage[1::2]
    lhs = _coo(indices, values, (4, 7), device=device)
    assert lhs._indices().stride() == (1, 4) and lhs._values().stride() == (2,)
    rhs = ((torch.arange(147).reshape(21, 7) % 17).float() / 8).to(device).T[:, 1::2]
    bias = (torch.arange(21).float() / 8).to(device)[1::2]
    storage = torch.full((21, 4), -17.0, device=device)
    out = storage.T[:, 1::2]
    return lhs, rhs, bias, out, storage


@pytest.mark.sparse_addmm_out
def test_flagos_coo_addmm_strided_out(flagos_addmm_device):
    device = flagos_addmm_device
    lhs_cpu, rhs_cpu, bias_cpu, ref_out, ref_storage = _strided_coo_addmm_inputs("cpu")
    lhs, rhs, bias, out, storage = _strided_coo_addmm_inputs(device)
    coeffs = {"alpha": -0.75, "beta": 2.0}
    torch.addmm(bias_cpu, lhs_cpu, rhs_cpu, out=ref_out, **coeffs)
    result = torch.addmm(bias, lhs, rhs, out=out, **coeffs)
    assert result is out and result.stride() == ref_out.stride()
    _assert_close(result, ref_out, device)
    _assert_close(storage, ref_storage, device)


@pytest.mark.sparse_addmm_out
@pytest.mark.parametrize(
    "out_shape", [(0,), (1, 1), None], ids=["empty", "undersized", "alias-bias"]
)
def test_flagos_coo_addmm_out_identity(out_shape, flagos_addmm_device):
    device = flagos_addmm_device
    indices = torch.tensor([[2, 0, 2], [1, 3, 1]])
    values = torch.tensor([3.0, 0.5, -2.0])
    rhs = torch.arange(28, dtype=values.dtype).reshape(4, 7) / 8
    bias = torch.ones((3, 7))
    bias_device = bias.to(device)
    out = bias_device if out_shape is None else torch.empty(out_shape, device=device)
    expected = torch.addmm(
        bias, _coo(indices, values, (3, 4)), rhs, beta=-1.5, alpha=0.5
    )
    actual = torch.addmm(
        bias_device,
        _coo(indices, values, (3, 4), device=device),
        rhs.to(device),
        out=out,
        beta=-1.5,
        alpha=0.5,
    )
    assert actual is out
    _assert_close(actual, expected, device)


@pytest.mark.parametrize(
    "kind", ["k", "bias-size", "bias-rank", "rhs-dtype", "out-dtype"]
)
def test_flagos_coo_addmm_invalid_arguments(kind, flagos_addmm_device):
    device = flagos_addmm_device
    lhs = _coo(torch.tensor([[0], [1]]), torch.ones(1), (2, 3), device=device)
    bias = torch.zeros(
        {"bias-size": (5,), "bias-rank": (1, 1, 1)}.get(kind, ()), device=device
    )
    rhs = torch.ones((2 if kind == "k" else 3, 4), device=device)
    if kind == "rhs-dtype":
        rhs = rhs.to(torch.float64)
    if kind == "out-dtype":
        out = torch.empty((2, 4), device=device, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="out dtype"):
            torch.addmm(bias, lhs, rhs, out=out)
        return
    message = {"k": "shapes cannot be multiplied", "rhs-dtype": "same dtype"}.get(
        kind, "broadcast"
    )
    with pytest.raises(RuntimeError, match=message):
        torch.addmm(bias, lhs, rhs, beta=0)
