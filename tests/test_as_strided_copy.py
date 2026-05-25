import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_BASE_DTYPES = (
    utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.int8, torch.uint8, torch.bool]
)
AS_STRIDED_COPY_DTYPES = list(dict.fromkeys(_BASE_DTYPES))

AS_STRIDED_COPY_CASES = [
    ((4, 6), (2, 3), (6, 1), 0),
    ((4, 6), (2, 3), (1, 6), 0),
    ((4, 6), (2, 2), (2, 3), 1),
    ((4, 6), (2, 2), (0, 1), 0),
    ((6,), (), (), 2),
    ((4, 6), (0, 3), (6, 1), 999),
]


def _make_input(shape, dtype, device):
    numel = max(1, int(torch.tensor(shape).prod().item()))
    if dtype == torch.bool:
        values = torch.arange(numel, device="cpu") % 2 == 0
    elif dtype.is_floating_point:
        values = torch.arange(numel, dtype=torch.float32, device="cpu") - numel // 2
        values = values.to(dtype)
    else:
        values = torch.arange(numel, dtype=torch.int64, device="cpu").to(dtype)
    return values.reshape(shape).to(device)


@pytest.mark.as_strided_copy
@pytest.mark.parametrize(
    "input_shape, size, stride, storage_offset", AS_STRIDED_COPY_CASES
)
@pytest.mark.parametrize("dtype", AS_STRIDED_COPY_DTYPES)
def test_accuracy_as_strided_copy(input_shape, size, stride, storage_offset, dtype):
    inp = _make_input(input_shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.as_strided_copy(ref_inp, size, stride, storage_offset)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.as_strided_copy(inp, size, stride, storage_offset)

    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided_copy
@pytest.mark.parametrize("dtype", AS_STRIDED_COPY_DTYPES)
def test_accuracy_as_strided_copy_default_storage_offset(dtype):
    base = _make_input((16,), dtype, flag_gems.device)
    inp = base[2:]
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.as_strided_copy(ref_inp, (4,), (2,))

    with flag_gems.use_gems():
        res_out = torch.ops.aten.as_strided_copy(inp, (4,), (2,))

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided_copy_out
@pytest.mark.parametrize("dtype", AS_STRIDED_COPY_DTYPES)
def test_accuracy_as_strided_copy_out_noncontiguous(dtype):
    inp = _make_input((4, 6), dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)
    out_stride = (1, 2)
    ref_out_buf = torch.empty_strided(
        (2, 3), out_stride, dtype=dtype, device=ref_inp.device
    )
    res_out_buf = torch.empty_strided(
        (2, 3), out_stride, dtype=dtype, device=flag_gems.device
    )
    ref_out = torch.ops.aten.as_strided_copy(
        ref_inp, (2, 3), (1, 6), 0, out=ref_out_buf
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.as_strided_copy(
            inp, (2, 3), (1, 6), 0, out=res_out_buf
        )

    assert res_out.data_ptr() == res_out_buf.data_ptr()
    assert res_out.stride() == out_stride
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided_copy_out
def test_accuracy_as_strided_copy_out_resizes():
    dtype = torch.float32
    inp = _make_input((4, 6), dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out_buf = torch.empty((0,), dtype=dtype, device=ref_inp.device)
    res_out_buf = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.as_strided_copy(
        ref_inp, (2, 3), (1, 6), 0, out=ref_out_buf
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.as_strided_copy(
            inp, (2, 3), (1, 6), 0, out=res_out_buf
        )

    assert tuple(res_out.shape) == (2, 3)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided_copy_out
def test_accuracy_as_strided_copy_out_aliases_input():
    dtype = torch.float32
    inp = _make_input((8,), dtype, flag_gems.device)
    ref_base = utils.to_reference(inp.clone())
    ref_out_buf = ref_base[1:5]
    res_out_buf = inp[1:5]
    torch.ops.aten.as_strided_copy(ref_base, (4,), (1,), 0, out=ref_out_buf)

    with flag_gems.use_gems():
        torch.ops.aten.as_strided_copy(inp, (4,), (1,), 0, out=res_out_buf)

    utils.gems_assert_equal(inp, ref_base)


@pytest.mark.as_strided_copy_out
def test_accuracy_as_strided_copy_out_dtype_mismatch_raises():
    inp = _make_input((8,), torch.int64, flag_gems.device)
    out = torch.empty((4,), dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.ops.aten.as_strided_copy(inp, (4,), (1,), 0, out=out)


@pytest.mark.as_strided_copy
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not hasattr(torch, "float8_e4m3fn"),
    reason="float8_e4m3fn accuracy coverage requires CUDA and PyTorch float8 support",
)
def test_accuracy_as_strided_copy_float8_fallback():
    dtype = torch.float8_e4m3fn
    inp = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.as_strided_copy(ref_inp, (2, 3), (1, 6), 0)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.as_strided_copy(inp, (2, 3), (1, 6), 0)

    utils.gems_assert_equal(res_out, ref_out)
