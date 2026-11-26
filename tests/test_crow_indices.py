import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Core shapes exercised by BATCHED_SHAPES (mirrors worktree CI branch).
BATCHED_SHAPES = [(2, 2, 2), (2, 2, 3, 4)]


CSR_CASES = [
    ([0, 0, 0, 0, 0], [], 4, 2),
    ([0, 0, 1, 1], [0], 3, 1),
    ([0, 2, 4, 6, 8], [0, 1, 0, 1, 0, 1, 0, 1], 4, 2),
    ([0, 2, 4, 4, 5, 6], [2, 4, 1, 3, 0, 2], 5, 3),
]


def _make_csr(crow, col, nrows, ncols, device, index_dtype=torch.int64):
    values = torch.randn(len(col), dtype=torch.float32, device=device)
    return torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=index_dtype, device=device),
        torch.tensor(col, dtype=index_dtype, device=device),
        values,
        size=(nrows, ncols),
    )


@pytest.mark.crow_indices
@pytest.mark.parametrize("crow,col,nrows,ncols", CSR_CASES)
def test_crow_indices(crow, col, nrows, ncols):
    # crow_indices is a device-local accessor: the returned pointer array is a
    # view of storage that always lives on the input's device, so the CPU
    # reference under --ref=cpu is compared by value only and stays comparable.
    if utils.TO_CPU:
        pytest.skip(
            "output device follows the input device; cross-device comparison is meaningless"
        )
    res_inp = _make_csr(crow, col, nrows, ncols, flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (nrows + 1,)
    assert res_out.device == res_inp.device
    utils.gems_assert_equal(res_out, ref_inp.crow_indices())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.crow_indices
def test_crow_indices_bsr():
    # BSR tensors carry their compressed row pointers in the same buffer.
    res_inp = torch.sparse_bsr_tensor(
        torch.tensor([0, 1, 2], device=flag_gems.device),
        torch.tensor([0, 1], device=flag_gems.device),
        torch.randn(2, 2, 2, device=flag_gems.device),
        size=(4, 4),
    )
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(res_inp)
    assert res_out.dtype == torch.int64
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.crow_indices
@pytest.mark.parametrize("shape", BATCHED_SHAPES)
def test_crow_indices_batched(shape):
    # The crow buffer of a batched CSR tensor has shape
    # ``batch_dims + (nrows + 1,)``; the accessor must preserve it exactly.
    crow = torch.zeros(*shape[:-2], shape[-2] + 1, dtype=torch.int64)
    crow[..., 1:] = 1
    col = torch.zeros(*shape[:-2], shape[-1], dtype=torch.int64)
    res_inp = torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        col.to(flag_gems.device),
        torch.randn(*shape[:-2], shape[-1], device=flag_gems.device),
        shape,
    )
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch.ops.aten.crow_indices(ref_inp)
    res_out = flag_gems.crow_indices(res_inp)
    assert res_out.shape == ref_out.shape == tuple(shape[:-2]) + (shape[-2] + 1,)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.crow_indices
def test_crow_indices_int32_buffer():
    # The stored buffer's dtype is preserved on the accessor path. The
    # comparison goes through the device-reference helper, which moves both
    # sides consistently under --ref=cpu.
    res_inp = _make_csr(
        [0, 2, 4], [0, 1, 0, 1], 2, 2, flag_gems.device, index_dtype=torch.int32
    )
    ref_inp = utils.to_reference(res_inp)
    res_out = flag_gems.crow_indices(res_inp)
    ref_out = utils.to_reference(ref_inp.crow_indices())
    assert res_out.dtype == torch.int32
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.crow_indices
def test_crow_indices_alias_and_mutation():
    # The schema ``Tensor(a) self -> Tensor(a)`` makes the result an aliasing
    # view: two reads share storage, and mutating the result mutates the
    # sparse tensor's stored crow values. A copying implementation breaks both.
    res_inp = _make_csr([0, 2, 4], [0, 1, 0, 1], 4, 2, flag_gems.device)

    first = flag_gems.crow_indices(res_inp)
    second = flag_gems.crow_indices(res_inp)
    assert (
        first.data_ptr() == second.data_ptr()
    ), "crow_indices must return an aliasing view, not a fresh copy"
    assert first.data_ptr() == res_inp.crow_indices().data_ptr()

    first[0] = 77
    assert (
        res_inp.crow_indices()[0].item() == 77
    ), "mutating the returned view must be visible in the sparse tensor"


@pytest.mark.crow_indices
def test_crow_indices_dispatch_stability():
    # Repeated dispatcher-routed calls plus an unrelated sparse metadata op in
    # between must not recurse, re-route or lose the alias (no shared mutable
    # state in the implementation).
    csr = _make_csr([0, 2, 5, 7, 8], [0, 1, 0, 2, 1, 3, 2, 0], 4, 8, flag_gems.device)
    ref = utils.to_reference(csr).crow_indices()

    first = flag_gems.crow_indices(csr)
    for _ in range(3):
        again = flag_gems.crow_indices(csr)
        assert again.data_ptr() == first.data_ptr()
        assert torch.ops.aten._nnz(csr) == csr._nnz()
        utils.gems_assert_equal(again, ref)

    # The dispatcher-routed path (used once registration is active) must reach
    # the same buffer as well.
    via_dispatch = torch.ops.aten.crow_indices(csr)
    assert via_dispatch.data_ptr() == first.data_ptr()
    utils.gems_assert_equal(via_dispatch, ref)


@pytest.mark.crow_indices
def test_crow_indices_dispatch_sentinel():
    # Sentinel test: verify that the FlagGems wrapper is actually reached
    # through the Autograd-key dispatch path. Patch flag_gems.crow_indices
    # with a counting wrapper at the flag_gems namespace level and verify
    # that the call routes through the registered implementation.
    original_impl = flag_gems.crow_indices
    calls = {"count": 0}

    def counting_impl(tensor):
        calls["count"] += 1
        return original_impl(tensor)

    flag_gems.crow_indices = counting_impl
    try:
        csr = _make_csr([0, 2, 4], [0, 1, 0, 1], 3, 5, flag_gems.device)
        flag_gems.crow_indices(csr)
        assert (
            calls["count"] > 0
        ), "flag_gems.crow_indices did not reach the registered implementation"
    finally:
        flag_gems.crow_indices = original_impl
    utils.gems_assert_equal(
        flag_gems.crow_indices(csr), utils.to_reference(csr).crow_indices()
    )


@pytest.mark.crow_indices
@pytest.mark.parametrize(
    "layout", ["csc", "bsc", "coo", "strided"], ids=["csc", "bsc", "coo", "strided"]
)
def test_crow_indices_unsupported_layout_raises(layout):
    # ATen accepts only row-compressed layouts (CSR and BSR) and rejects every
    # other layout with the native message, byte-identical.
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=flag_gems.device)
    if layout == "csc":
        inp = torch.sparse_csc_tensor(indices, indptr, values, size=(4, 2))
    elif layout == "bsc":
        inp = torch.sparse_bsc_tensor(
            indices,
            indptr,
            torch.randn(2, 2, 2, device=flag_gems.device),
            size=(4, 4),
        )
    elif layout == "coo":
        inp = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]], device=flag_gems.device),
            torch.randn(2, device=flag_gems.device),
            (2, 3),
        )
    else:
        inp = torch.randn(2, 3, device=flag_gems.device)

    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.crow_indices(utils.to_reference(inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.crow_indices(inp)
    assert str(ref_err.value) == str(
        res_err.value
    ), f"{layout}: message differs: {res_err.value!r} vs {ref_err.value!r}"
