import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DIAGONAL = list(zip(utils.POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))
TRIL_DTYPES = utils.FLOAT_DTYPES + [torch.int32] + utils.BOOL_TYPES
TRIL_OUT_EDGE_DTYPES = [torch.float32, torch.int64, torch.bool]

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Warning only once for all operators,  other operators may also be "
        "overridden\\.:UserWarning:torch.library"
    ),
]


def _make_tril_input(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype is torch.bool:
        inp = torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-8, 9, shape, dtype=dtype, device=flag_gems.device)
    return utils.unsqueeze_tensor(inp, 2)


def _make_sequence(shape, dtype):
    numel = 1
    for size in shape:
        numel *= size
    values = torch.arange(1, numel + 1, device=flag_gems.device).reshape(shape)
    if dtype is torch.bool:
        return values.remainder(2).to(dtype)
    return values.to(dtype)


def _assert_tril_inplace_matches_reference(inp, diagonal):
    ref_inp = utils.to_reference(inp.clone())
    ref_inp.tril_(diagonal)

    original_stride = inp.stride()
    original_data_ptr = inp.data_ptr()

    with flag_gems.use_gems():
        res = inp.tril_(diagonal)

    utils.gems_assert_equal(inp, ref_inp)
    assert res is inp
    assert inp.data_ptr() == original_data_ptr
    assert inp.stride() == original_stride


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous()


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril_noncontiguous(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype).transpose(-2, -1)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous()


@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal, dtype",
    [
        ((256, 2048), 0, torch.float32),
        ((512, 2048), -3, torch.int32),
    ],
)
def test_tril_wide_exact_row_dispatch(shape, diagonal, dtype):
    tril_mod = importlib.import_module("flag_gems.ops.tril")
    batch = 1
    for size in shape[:-2]:
        batch *= size
    assert tril_mod._use_wide_exact_row(shape[-2], shape[-1], batch)

    inp = _make_tril_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous()


@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1024, 16, 16), True),
        ((512, 32, 32), True),
        ((128, 64, 64), False),
        ((4, 16, 16), False),
    ],
)
def test_tril_tiny_batched_tile_dispatch(shape, expected):
    tril_mod = importlib.import_module("flag_gems.ops.tril")
    batch = 1
    for size in shape[:-2]:
        batch *= size

    assert tril_mod._use_tiny_batched_tile(shape[-2], shape[-1], batch) is expected


@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal, dtype",
    [
        ((1024, 16, 16), 0, torch.float32),
        ((512, 32, 32), -1, torch.int32),
        ((256, 16, 16), 1, torch.bool),
    ],
)
def test_tril_tiny_batched_tile_correctness(shape, diagonal, dtype):
    tril_mod = importlib.import_module("flag_gems.ops.tril")
    batch = 1
    for size in shape[:-2]:
        batch *= size
    assert tril_mod._use_tiny_batched_tile(shape[-2], shape[-1], batch)

    inp = _make_tril_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous()


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril_out(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.tril(ref_inp, diagonal, out=ref_out)

    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert res.data_ptr() == out.data_ptr()


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL[:3])
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril_out_resizes(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty(0, dtype=dtype, device=ref_inp.device)
    torch.tril(ref_inp, diagonal, out=ref_out)

    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert out.shape == inp.shape
    assert res.data_ptr() == out.data_ptr()


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
@pytest.mark.parametrize("dtype", TRIL_OUT_EDGE_DTYPES)
def test_tril_out_aliases_input(diagonal, dtype):
    if dtype is torch.int64 and not utils.int64_is_supported:
        # int64 is not supported on this device
        return

    inp = _make_sequence((2, 5, 7), dtype)

    ref = utils.to_reference(inp.clone())
    try:
        torch.tril(ref, diagonal, out=ref)
    except RuntimeError as exc:
        pytest.skip(f"PyTorch rejects tril out=input alias: {exc}")

    original_data_ptr = inp.data_ptr()
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=inp)

    utils.gems_assert_equal(inp, ref)
    assert inp.data_ptr() == original_data_ptr
    assert res.data_ptr() == original_data_ptr


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("diagonal", [-2, 0, 3])
@pytest.mark.parametrize("dtype", TRIL_OUT_EDGE_DTYPES)
def test_tril_out_noncontiguous_out(diagonal, dtype):
    if dtype is torch.int64 and not utils.int64_is_supported:
        # int64 is not supported on this device
        return

    inp = _make_sequence((2, 5, 7), dtype)
    out_base = torch.empty((2, 7, 5), dtype=dtype, device=flag_gems.device)
    out = out_base.transpose(-2, -1)
    assert not out.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_out_base = torch.empty((2, 7, 5), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out_base.transpose(-2, -1)
    torch.tril(ref_inp, diagonal, out=ref_out)

    original_data_ptr = out.data_ptr()
    original_stride = out.stride()
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert out.data_ptr() == original_data_ptr
    assert out.stride() == original_stride
    assert res.data_ptr() == original_data_ptr


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("diagonal", [-2, 0, 3])
@pytest.mark.parametrize("dtype", TRIL_OUT_EDGE_DTYPES)
def test_tril_out_sliced_leading_batch_out(diagonal, dtype):
    if dtype is torch.int64 and not utils.int64_is_supported:
        # int64 is not supported on this device
        return

    inp = _make_sequence((2, 5, 7), dtype)
    out_base = torch.empty((4, 5, 7), dtype=dtype, device=flag_gems.device)
    out = out_base[::2]
    assert not out.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_out_base = torch.empty((4, 5, 7), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out_base[::2]
    torch.tril(ref_inp, diagonal, out=ref_out)

    original_data_ptr = out.data_ptr()
    original_stride = out.stride()
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert out.data_ptr() == original_data_ptr
    assert out.stride() == original_stride
    assert res.data_ptr() == original_data_ptr


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("diagonal", [-99, 99])
@pytest.mark.parametrize("dtype", TRIL_OUT_EDGE_DTYPES)
def test_tril_out_extreme_diagonal_noncontiguous_out(diagonal, dtype):
    if dtype is torch.int64 and not utils.int64_is_supported:
        # int64 is not supported on this device
        return

    inp = _make_sequence((2, 5, 7), dtype)
    out_base = torch.empty((2, 7, 5), dtype=dtype, device=flag_gems.device)
    out = out_base.transpose(-2, -1)

    ref_inp = utils.to_reference(inp)
    ref_out_base = torch.empty((2, 7, 5), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out_base.transpose(-2, -1)
    torch.tril(ref_inp, diagonal, out=ref_out)

    original_data_ptr = out.data_ptr()
    original_stride = out.stride()
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal, out=out)

    utils.gems_assert_equal(out, ref_out)
    assert out.data_ptr() == original_data_ptr
    assert out.stride() == original_stride
    assert res.data_ptr() == original_data_ptr


@pytest.mark.tril
@pytest.mark.tril_out
def test_tril_out_strided_dispatch_guards():
    tril_mod = importlib.import_module("flag_gems.ops.tril")
    inp = torch.empty((2, 5, 7), dtype=torch.float32, device=flag_gems.device)

    transposed = torch.empty((2, 7, 5), dtype=inp.dtype, device=inp.device).transpose(
        -2, -1
    )
    sliced_batch = torch.empty((4, 5, 7), dtype=inp.dtype, device=inp.device)[::2]
    expanded = torch.empty((1, 5, 7), dtype=inp.dtype, device=inp.device).expand(
        2, 5, 7
    )
    overlapping = torch.as_strided(
        torch.empty((96,), dtype=inp.dtype, device=inp.device),
        inp.shape,
        (35, 1, 1),
    )

    assert tril_mod._can_use_strided_out_kernel(inp, transposed)
    assert tril_mod._can_use_strided_out_kernel(inp, sliced_batch)
    assert not tril_mod._can_use_strided_out_kernel(inp, expanded)
    assert not tril_mod._can_use_strided_out_kernel(inp, overlapping)


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril_inplace(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype)
    _assert_tril_inplace_matches_reference(inp, diagonal)


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", TRIL_DTYPES)
def test_tril_inplace_noncontiguous(shape, diagonal, dtype):
    inp = _make_tril_input(shape, dtype).transpose(-2, -1)
    _assert_tril_inplace_matches_reference(inp, diagonal)


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_inplace_strided_batched_view(diagonal, dtype):
    base = _make_sequence((2, 8, 10), dtype)
    inp = base[:, ::2, 1::2]
    _assert_tril_inplace_matches_reference(inp, diagonal)


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("diagonal", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_inplace_expanded_view(diagonal, dtype):
    base = _make_sequence((1, 4), dtype)
    ref_base = utils.to_reference(base.clone())
    inp_base = base.clone()

    ref = ref_base.expand(3, 4)
    inp = inp_base.expand(3, 4)
    ref.tril_(diagonal)

    original_stride = inp.stride()
    original_data_ptr = inp.data_ptr()
    with flag_gems.use_gems():
        res = inp.tril_(diagonal)

    utils.gems_assert_equal(inp, ref)
    assert res is inp
    assert inp.data_ptr() == original_data_ptr
    assert inp.stride() == original_stride


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("diagonal", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_inplace_overlapping_as_strided_view(diagonal, dtype):
    base = _make_sequence((8,), dtype)
    ref_base = utils.to_reference(base.clone())
    inp_base = base.clone()

    ref = torch.as_strided(ref_base, (3, 3), (1, 1))
    inp = torch.as_strided(inp_base, (3, 3), (1, 1))
    ref.tril_(diagonal)

    original_stride = inp.stride()
    original_data_ptr = inp.data_ptr()
    with flag_gems.use_gems():
        res = inp.tril_(diagonal)

    utils.gems_assert_equal(inp, ref)
    assert res is inp
    assert inp.data_ptr() == original_data_ptr
    assert inp.stride() == original_stride


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0), (2, 0, 7), (2, 5, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_empty(shape, dtype):
    inp = _make_tril_input(shape, dtype)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, -1)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, -1)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == inp.shape


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0), (2, 0, 7), (2, 5, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_inplace_empty(shape, dtype):
    inp = _make_tril_input(shape, dtype)
    _assert_tril_inplace_matches_reference(inp, -1)


@pytest.mark.tril
@pytest.mark.tril_
@pytest.mark.parametrize("diagonal", [-99, 99])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_tril_inplace_extreme_diagonal(diagonal, dtype):
    inp = _make_tril_input((2, 5, 7), dtype)
    _assert_tril_inplace_matches_reference(inp, diagonal)


@pytest.mark.tril
def test_tril_invalid_rank():
    inp = torch.tensor(1.0, device=flag_gems.device)

    with (
        flag_gems.use_gems(),
        pytest.raises(
            RuntimeError, match="tril: input tensor must have at least 2 dimensions"
        ),
    ):
        torch.tril(inp)


@pytest.mark.tril
@pytest.mark.tril_
def test_tril_inplace_invalid_rank():
    inp = torch.tensor(1.0, device=flag_gems.device)

    with (
        flag_gems.use_gems(),
        pytest.raises(
            RuntimeError, match="tril: input tensor must have at least 2 dimensions"
        ),
    ):
        inp.tril_()
