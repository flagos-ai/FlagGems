import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

EXTRA_INT_DTYPES = [torch.int8, torch.uint8]
ASCEND_UNSUPPORTED_REFERENCE_DTYPES = (torch.bfloat16, torch.float64)


def _filter_reference_supported(dtypes):
    if flag_gems.vendor_name == "ascend":
        return [
            dtype for dtype in dtypes if dtype not in ASCEND_UNSUPPORTED_REFERENCE_DTYPES
        ]
    return dtypes


NANMEDIAN_DTYPES = _filter_reference_supported(
    utils.ALL_FLOAT_DTYPES + EXTRA_INT_DTYPES + utils.ALL_INT_DTYPES
)
FLOAT_DTYPES = _filter_reference_supported(utils.ALL_FLOAT_DTYPES)
LARGE_RADIX_DTYPES = _filter_reference_supported(
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.uint8,
    ]
)


def _make_input(shape, dtype, with_nan=True):
    if dtype is torch.uint8:
        inp = torch.randint(0, 101, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES or dtype is torch.int8:
        inp = torch.randint(-100, 101, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if with_nan and inp.numel() > 0:
            inp.reshape(-1)[::7] = float("nan")
    return inp


def _assert_nanmedian_values(res, ref, dtype):
    if dtype.is_floating_point:
        utils.gems_assert_close(res, ref, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(res, ref)


def _assert_nanmedian_indices_valid(inp, values, indices, dim, keepdim, dtype):
    dim = dim % inp.ndim
    if indices.numel() == 0:
        return

    assert torch.all(indices >= 0)
    assert torch.all(indices < inp.shape[dim])

    gather_indices = indices if keepdim else indices.unsqueeze(dim)
    gathered = torch.gather(inp, dim, gather_indices)
    if not keepdim:
        gathered = gathered.squeeze(dim)

    _assert_nanmedian_values(gathered, values, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("shape", [(), (1,), (17,), (4, 33), (2, 3, 129)])
@pytest.mark.parametrize("dtype", NANMEDIAN_DTYPES)
def test_nanmedian(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp)

    _assert_nanmedian_values(res, ref, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize(
    ("shape", "dim"),
    [((7,), 0), ((4, 33), 0), ((4, 33), -1), ((2, 3, 129), 1), ((2, 3, 1031), -1)],
)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", NANMEDIAN_DTYPES)
def test_nanmedian_dim(shape, dim, keepdim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp, dim=dim, keepdim=keepdim)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=dim, keepdim=keepdim)

    _assert_nanmedian_values(res.values, ref.values, dtype)
    _assert_nanmedian_indices_valid(inp, res.values, res.indices, dim, keepdim, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", LARGE_RADIX_DTYPES)
def test_nanmedian_large_radix_path(dtype):
    inp = _make_input((4, 8192), dtype)
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp, dim=-1)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=-1)

    _assert_nanmedian_values(res.values, ref.values, dtype)
    _assert_nanmedian_indices_valid(inp, res.values, res.indices, -1, False, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nanmedian_all_nan_rows(dtype):
    inp = torch.tensor(
        [[float("nan"), float("nan")], [float("nan"), 1.0], [2.0, float("nan")]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp, dim=1)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=1)

    _assert_nanmedian_values(res.values, ref.values, dtype)
    _assert_nanmedian_indices_valid(inp, res.values, res.indices, 1, False, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", NANMEDIAN_DTYPES)
def test_nanmedian_non_contiguous(dtype):
    inp = _make_input((5, 7, 3), dtype).transpose(0, 1)
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp, dim=1)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=1)

    _assert_nanmedian_values(res.values, ref.values, dtype)
    _assert_nanmedian_indices_valid(inp, res.values, res.indices, 1, False, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", NANMEDIAN_DTYPES)
def test_nanmedian_out(dtype):
    inp = _make_input((4, 33), dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)
    torch.ops.aten.nanmedian.out(ref_inp, out=ref_out)
    out = torch.empty((), dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res = torch.ops.aten.nanmedian.out(inp, out=out)

    assert res is out
    _assert_nanmedian_values(out, ref_out, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", NANMEDIAN_DTYPES)
def test_nanmedian_dim_values(dtype):
    inp = _make_input((4, 33), dtype)
    ref_inp = utils.to_reference(inp)
    ref_values = torch.empty((4,), dtype=dtype, device=ref_inp.device)
    ref_indices = torch.empty((4,), dtype=torch.long, device=ref_inp.device)
    torch.nanmedian(ref_inp, dim=1, out=(ref_values, ref_indices))

    out_values = torch.empty((4,), dtype=dtype, device=flag_gems.device)
    out_indices = torch.empty((4,), dtype=torch.long, device=flag_gems.device)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=1, out=(out_values, out_indices))

    assert res.values is out_values
    assert res.indices is out_indices
    _assert_nanmedian_values(out_values, ref_values, dtype)
    _assert_nanmedian_indices_valid(inp, out_values, out_indices, 1, False, dtype)


@pytest.mark.nanmedian
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.uint8])
def test_nanmedian_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.nanmedian(ref_inp)

    with flag_gems.use_gems():
        res = torch.nanmedian(inp)

    _assert_nanmedian_values(res, ref, dtype)

    inp = torch.empty((2, 0), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(IndexError):
        torch.nanmedian(inp, dim=1)

    inp = torch.empty((2, 0), dtype=dtype, device=flag_gems.device)
    ref = torch.nanmedian(utils.to_reference(inp), dim=0)
    with flag_gems.use_gems():
        res = torch.nanmedian(inp, dim=0)
    _assert_nanmedian_values(res.values, ref.values, dtype)
    utils.gems_assert_equal(res.indices, ref.indices)


@pytest.mark.nanmedian
def test_nanmedian_bool_unsupported():
    inp = torch.tensor([True, False], device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(NotImplementedError):
        torch.nanmedian(inp)
