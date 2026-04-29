import pytest
import torch

import flag_gems
from flag_gems.ops.median import median_dim as gems_median_dim

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DTYPES = [torch.float32, torch.int32]
    NO_DIM_SHAPES = [(5,), (3, 11)]
    DIM_CASES = [((3, 17), 1), ((3, 0, 5), 0)]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DTYPES = FLOAT_DTYPES + utils.ALL_INT_DTYPES
    NO_DIM_SHAPES = [(1,), (3,), (10,), (3, 4), (2, 5, 9), (131,), (256,)]
    DIM_CASES = [
        ((9,), 0),
        ((10,), -1),
        ((2, 5), 0),
        ((19, 23), 0),
        ((19, 23), 1),
        ((4, 6, 9), 1),
        ((4, 6, 9), -1),
        ((3, 0, 5), 0),
        ((0, 4), 1),
    ]
    KEEPDIM = [True, False]


def _make_input(shape, dtype):
    if dtype in utils.ALL_INT_DTYPES or dtype in (torch.int8, torch.uint8):
        numel = 1
        for size in shape:
            numel *= size
        if numel == 0:
            return torch.empty(shape, dtype=dtype, device=flag_gems.device)
        vals = torch.arange(numel, device=flag_gems.device, dtype=torch.int64)
        vals = (vals * 37) % numel - numel // 2
        return vals.reshape(shape).to(dtype)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _has_unique_median_indices(inp, dim, values, keepdim):
    if inp.ndim == 0:
        return True

    dim = dim % inp.ndim
    compare_values = values if keepdim else values.unsqueeze(dim)
    if inp.dtype.is_floating_point:
        value_matches = (inp == compare_values) | (
            torch.isnan(inp) & torch.isnan(compare_values)
        )
    else:
        value_matches = inp == compare_values
    return bool(torch.all(torch.count_nonzero(value_matches, dim=dim) == 1).item())


def _assert_indices_select_values(inp, dim, values, indices, *, keepdim, equal_nan):
    if inp.ndim == 0:
        flag_gems.testing.assert_equal(indices, torch.zeros_like(indices))
        return

    dim = dim % inp.ndim
    gather_indices = indices if keepdim else indices.unsqueeze(dim)
    gathered = torch.gather(inp, dim, gather_indices)
    expected = values if keepdim else values.unsqueeze(dim)
    flag_gems.testing.assert_equal(gathered, expected, equal_nan=equal_nan)


def _assert_median_dim_equal(
    res,
    ref,
    dtype,
    *,
    equal_nan=False,
    exact_indices=None,
    inp=None,
    dim=None,
    keepdim=False,
):
    res_values, res_indices = res
    ref_values, ref_indices = ref
    if exact_indices is None:
        exact_indices = res_indices.device == ref_indices.device
    if exact_indices and inp is not None and dim is not None:
        exact_indices = _has_unique_median_indices(inp, dim, res_values, keepdim)
    utils.gems_assert_equal(res_values, ref_values, equal_nan=equal_nan)
    if inp is not None and dim is not None:
        _assert_indices_select_values(
            inp, dim, res_values, res_indices, keepdim=keepdim, equal_nan=equal_nan
        )
    if exact_indices:
        utils.gems_assert_equal(res_indices, ref_indices)
    assert tuple(res_values.shape) == tuple(ref_values.shape)
    assert tuple(res_indices.shape) == tuple(ref_indices.shape)


@pytest.mark.median
@pytest.mark.parametrize("shape", NO_DIM_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_median_no_dim(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("shape, dim", DIM_CASES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    _assert_median_dim_equal(res_out, ref_out, dtype, inp=inp, dim=dim, keepdim=keepdim)


@pytest.mark.median
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_nan(dtype):
    inp = torch.tensor(
        [
            [float("nan"), -2.0, 5.0, 1.0, 8.0],
            [4.0, 3.0, 2.0, float("nan"), 1.0],
            [float("nan"), float("nan"), 9.0, 7.0, 6.0],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_no_dim = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_no_dim = torch.median(inp)
    utils.gems_assert_equal(res_no_dim, ref_no_dim, equal_nan=True)

    ref_dim = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_dim = torch.median(inp, dim=1)
    _assert_median_dim_equal(res_dim, ref_dim, dtype, equal_nan=True, inp=inp, dim=1)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(65,), (129,), (257,), (16, 64)])
def test_median_no_dim_lastdim_sort(dtype, shape):
    numel = 1
    for size in shape:
        numel *= size
    inp = torch.arange(numel, dtype=torch.float32, device=flag_gems.device)
    inp = ((inp * 17) % numel - numel // 2).reshape(shape).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [65, 257, 1024])
def test_median_no_dim_lastdim_sort_nan(dtype, width):
    inp = torch.randn((width,), dtype=dtype, device=flag_gems.device)
    inp[0] = float("nan")
    inp[width // 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.isnan(res_out).item()


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [64, 256])
def test_median_no_dim_direct_flat_nan(dtype, width):
    inp = torch.randn((width,), dtype=dtype, device=flag_gems.device)
    inp[0] = float("nan")
    inp[width - 1] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.isnan(res_out).item()


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_tie_lower_median(dtype, keepdim):
    inp = torch.tensor(
        [[5, 1, 5, 9, 2], [4, 4, 4, 1, 8], [7, 0, 7, 0, 7]],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        exact_indices=False,
        inp=inp,
        dim=1,
        keepdim=keepdim,
    )


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32]
)
def test_median_direct_duplicate_indices_select_value(dtype):
    base = torch.tensor(
        [15, *range(15), 15, *range(16, 31)],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = base[:, None].repeat(1, 4)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=0
    )


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32]
)
@pytest.mark.parametrize("reduction_size", [32, 64, 65, 128, 144, 256, 257])
def test_median_reduction_boundary_dim0(dtype, reduction_size):
    inp = _make_input((reduction_size, 33), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0)

    _assert_median_dim_equal(res_out, ref_out, dtype, inp=inp, dim=0)


@pytest.mark.median
@pytest.mark.parametrize(
    "shape, keepdim",
    [
        ((144, 8192), False),
        ((32, 384, 128), True),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_median_direct_public_shapes_dim0(dtype, shape, keepdim):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0, keepdim=keepdim)

    _assert_median_dim_equal(res_out, ref_out, dtype, inp=inp, dim=0, keepdim=keepdim)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction_size", [32, 144, 256])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_direct_nan_first_index_dim0(dtype, reduction_size, keepdim):
    inp = torch.randn((reduction_size, 4), dtype=dtype, device=flag_gems.device)
    expected_indices = torch.tensor(
        [0, reduction_size // 2, reduction_size - 1, 0], device=flag_gems.device
    )
    inp[0, 0] = float("nan")
    inp[min(7, reduction_size - 1), 0] = float("nan")
    inp[reduction_size // 2, 1] = float("nan")
    inp[reduction_size - 1, 2] = float("nan")
    inp[:, 3] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        exact_indices=False,
        inp=inp,
        dim=0,
        keepdim=keepdim,
    )
    flag_gems.testing.assert_equal(res_out.indices.reshape(-1), expected_indices)
    assert torch.all(torch.isnan(res_out.values)).item()


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
)
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_direct_non_contiguous_dim0(dtype, keepdim):
    base = _make_input((5, 33, 144), dtype)
    inp = base.permute(2, 0, 1)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0, keepdim=keepdim)

    _assert_median_dim_equal(res_out, ref_out, dtype, inp=inp, dim=0, keepdim=keepdim)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_direct_named_dim0_preserves_names(keepdim):
    inp = torch.randn((32, 7), dtype=torch.float32, device=flag_gems.device)
    inp = inp.refine_names("reduce", "feature")
    ref_inp = utils.to_reference(inp.rename(None))

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    res_out = gems_median_dim(inp, dim="reduce", keepdim=keepdim)

    expected_names = ("reduce", "feature") if keepdim else ("feature",)
    assert res_out.values.names == expected_names
    assert res_out.indices.names == expected_names
    _assert_median_dim_equal(
        (res_out.values.rename(None), res_out.indices.rename(None)),
        ref_out,
        torch.float32,
        inp=inp.rename(None),
        dim=0,
        keepdim=keepdim,
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_median_empty_no_dim(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=dtype.is_floating_point)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float64, torch.int8, torch.uint8])
def test_median_extra_no_dim_dtypes(dtype):
    inp = _make_input((9,), dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
def test_median_bool_no_dim():
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("bool median no-dim is CUDA-specific in native PyTorch")

    inp = torch.tensor([True, False, True], device=flag_gems.device)
    ref_out = torch.median(inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    assert res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device
    assert res_out.item() == ref_out.item()


@pytest.mark.median
def test_median_empty_complex_no_dim():
    inp = torch.empty((0,), dtype=torch.complex64, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    assert res_out.dtype == torch.complex64
    assert torch.isnan(res_out.real)
    assert res_out.imag == 0


@pytest.mark.median
def test_median_complex_nonempty_errors():
    inp = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64, device=flag_gems.device)

    with pytest.raises((RuntimeError, NotImplementedError)):
        with flag_gems.use_gems():
            torch.median(inp)

    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.median(inp, dim=0)


@pytest.mark.median
@pytest.mark.parametrize("shape, dim", [((3, 0, 5), 1), ((0, 4), 0)])
def test_median_empty_reduced_dim_raises(shape, dim):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        with flag_gems.use_gems():
            torch.median(inp, dim=dim)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
@pytest.mark.parametrize("shape, dim", [((3, 0, 5), 0), ((3, 0, 5), 2)])
def test_median_empty_output_unsupported_dtype(shape, dim, dtype):
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("empty-output bool/complex median semantics differ on CPU")

    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim)

    assert tuple(res_out.values.shape) == tuple(ref_out.values.shape)
    assert tuple(res_out.indices.shape) == tuple(ref_out.indices.shape)
    assert res_out.values.dtype == ref_out.values.dtype
    assert res_out.indices.dtype == ref_out.indices.dtype
    assert res_out.values.device == ref_out.values.device
    assert res_out.indices.device == ref_out.indices.device


@pytest.mark.median
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_scalar_dim(keepdim):
    inp = torch.tensor(5.0, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out, ref_out, torch.float32, inp=inp, dim=0, keepdim=keepdim
    )


@pytest.mark.median
def test_median_non_contiguous():
    base = torch.randn((5, 4, 6), dtype=torch.float32, device=flag_gems.device)
    inp = base.transpose(0, 2)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_no_dim = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_no_dim = torch.median(inp)
    utils.gems_assert_equal(res_no_dim, ref_no_dim)

    ref_dim = torch.median(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_dim = torch.median(inp, dim=0)
    _assert_median_dim_equal(res_dim, ref_dim, torch.float32, inp=inp, dim=0)


@pytest.mark.median
@pytest.mark.parametrize("width", [257, 5001])
def test_median_large_width(width):
    inp = torch.randn((2, width), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(res_out, ref_out, torch.float32, inp=inp, dim=1)


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.float16, 65),
        (torch.float16, 129),
        (torch.bfloat16, 65),
        (torch.bfloat16, 129),
    ],
)
def test_median_lastdim_sort_unique_exact_index(dtype, width):
    rank = (width - 1) // 2
    first = torch.arange(width, dtype=dtype, device=flag_gems.device)
    second = torch.arange(width, dtype=dtype, device=flag_gems.device).roll(7)
    inp = torch.stack((first, second))
    expected_indices = torch.tensor([rank, (rank + 7) % width], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=True, inp=inp, dim=1
    )
    flag_gems.testing.assert_equal(res_out.indices, expected_indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_lastdim_sort_1536_unique_exact_index(dtype, keepdim):
    width = 1536
    rank = (width - 1) // 2
    first = torch.arange(width, dtype=dtype, device=flag_gems.device)
    second = torch.arange(width, dtype=dtype, device=flag_gems.device).roll(13)
    inp = torch.stack((first, second))
    expected_indices = torch.tensor(
        [rank, (rank + 13) % width], device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        exact_indices=True,
        inp=inp,
        dim=1,
        keepdim=keepdim,
    )
    flag_gems.testing.assert_equal(res_out.indices.reshape(-1), expected_indices)


@pytest.mark.median
@pytest.mark.parametrize("width", [257, 258, 640, 1023, 1024, 1025, 1536, 2048, 2049])
def test_median_fp32_key_select_boundaries(width):
    vals = torch.arange(4 * width, dtype=torch.int64, device=flag_gems.device)
    vals = ((vals * 7919) % (3 * width)) - width
    inp = vals.reshape(4, width).to(torch.float32)
    inp[0, 0] = float("-inf")
    inp[0, width - 1] = float("inf")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, torch.float32, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("width", [257, 258, 640, 1023, 1024, 1536, 2048])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_fp32_key_select_unique_exact_index(width, keepdim):
    rank = (width - 1) // 2
    first = torch.arange(width, dtype=torch.float32, device=flag_gems.device)
    second = torch.arange(width, dtype=torch.float32, device=flag_gems.device).roll(23)
    inp = torch.stack((first, second))
    expected_indices = torch.tensor(
        [rank, (rank + 23) % width], device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        torch.float32,
        exact_indices=True,
        inp=inp,
        dim=1,
        keepdim=keepdim,
    )
    flag_gems.testing.assert_equal(res_out.indices.reshape(-1), expected_indices)


@pytest.mark.median
@pytest.mark.parametrize("width", [640, 1536])
def test_median_fp32_key_select_nan_first_index(width):
    inp = torch.randn((4, width), dtype=torch.float32, device=flag_gems.device)
    expected_indices = torch.tensor(
        [0, 511, min(512, width - 1), width - 1], device=flag_gems.device
    )
    inp[0, 0] = float("nan")
    inp[0, min(width - 1, 17)] = float("nan")
    inp[1, 511] = float("nan")
    inp[2, min(512, width - 1)] = float("nan")
    inp[3, width - 1] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        torch.float32,
        equal_nan=True,
        exact_indices=False,
        inp=inp,
        dim=1,
    )
    flag_gems.testing.assert_equal(res_out.indices, expected_indices)
    assert torch.all(torch.isnan(res_out.values)).item()


@pytest.mark.median
def test_median_fp32_key_select_duplicates_infinities_and_zeros():
    width = 640
    rows = []

    duplicate = torch.arange(width, dtype=torch.float32, device=flag_gems.device)
    duplicate[0] = 319
    duplicate[511] = 319
    rows.append(duplicate)

    infinities = torch.linspace(
        -10, 10, width, dtype=torch.float32, device=flag_gems.device
    )
    infinities[:40] = float("-inf")
    infinities[-40:] = float("inf")
    rows.append(infinities.roll(71))

    zeros = torch.arange(width, dtype=torch.float32, device=flag_gems.device) - 320
    zeros[0] = -0.0
    zeros[1] = 0.0
    rows.append(zeros.roll(9))

    inp = torch.stack(rows)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, torch.float32, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("width", [640, 1536])
def test_median_fp32_key_select_signed_zero_bits(width):
    neg_balanced = torch.full(
        (width,), -0.0, dtype=torch.float32, device=flag_gems.device
    )
    neg_balanced[width // 2 :] = 0.0
    pos_biased = torch.full((width,), 0.0, dtype=torch.float32, device=flag_gems.device)
    pos_biased[: width // 2 - 1] = -0.0
    inp = torch.stack((neg_balanced.roll(37), pos_biased.roll(53)))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, torch.float32, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [512, 513, 640, 1024, 1025, 1536, 2048, 2049])
def test_median_f16_key_select_boundaries(dtype, width):
    vals = torch.arange(4 * width, dtype=torch.int64, device=flag_gems.device)
    vals = ((vals * 6151) % (3 * width)) - width
    inp = vals.reshape(4, width).to(dtype)
    inp[0, 0] = float("-inf")
    inp[0, width - 1] = float("inf")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("width", [640, 1025, 1536, 2048])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_fp16_key_select_unique_exact_index(width, keepdim):
    rank = (width - 1) // 2
    first = torch.arange(width, dtype=torch.float16, device=flag_gems.device)
    second = torch.arange(width, dtype=torch.float16, device=flag_gems.device).roll(29)
    inp = torch.stack((first, second))
    expected_indices = torch.tensor(
        [rank, (rank + 29) % width], device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        torch.float16,
        exact_indices=True,
        inp=inp,
        dim=1,
        keepdim=keepdim,
    )
    flag_gems.testing.assert_equal(res_out.indices.reshape(-1), expected_indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [640, 1025, 1536, 2048])
def test_median_f16_key_select_nan_first_index(dtype, width):
    inp = torch.randn((5, width), dtype=dtype, device=flag_gems.device)
    expected_indices = torch.tensor(
        [0, 511, min(1024, width - 1), width - 1, 0], device=flag_gems.device
    )
    inp[0, 0] = float("nan")
    inp[0, min(width - 1, 17)] = float("nan")
    inp[1, 511] = float("nan")
    inp[2, min(1024, width - 1)] = float("nan")
    inp[3, width - 1] = float("nan")
    inp[4].fill_(float("nan"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        exact_indices=False,
        inp=inp,
        dim=1,
    )
    flag_gems.testing.assert_equal(res_out.indices, expected_indices)
    assert torch.all(torch.isnan(res_out.values)).item()


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_f16_key_select_width640_keepdim(dtype, keepdim):
    width = 640
    vals = torch.arange(5 * width, dtype=torch.int64, device=flag_gems.device)
    vals = ((vals * 3203) % (4 * width)) - 2 * width
    inp = vals.reshape(5, width).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=-1, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=-1, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        exact_indices=False,
        inp=inp,
        dim=-1,
        keepdim=keepdim,
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_median_f16_key_select_duplicates_infinities_and_zeros(dtype):
    width = 1536
    rows = []

    duplicate = torch.arange(width, dtype=torch.float32, device=flag_gems.device)
    duplicate[0] = 767
    duplicate[1111] = 767
    rows.append(duplicate.to(dtype))

    infinities = torch.linspace(
        -10, 10, width, dtype=torch.float32, device=flag_gems.device
    )
    infinities[:96] = float("-inf")
    infinities[-96:] = float("inf")
    rows.append(infinities.roll(97).to(dtype))

    zeros = torch.arange(width, dtype=torch.float32, device=flag_gems.device) - 768
    zeros[0] = -0.0
    zeros[1] = 0.0
    rows.append(zeros.roll(31).to(dtype))

    inp = torch.stack(rows)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [640, 1536])
def test_median_f16_key_select_signed_zero_index_bits(dtype, width):
    neg_balanced = torch.full((width,), -0.0, dtype=dtype, device=flag_gems.device)
    neg_balanced[width // 2 :] = 0.0
    pos_biased = torch.full((width,), 0.0, dtype=dtype, device=flag_gems.device)
    pos_biased[: width // 2 - 1] = -0.0
    inp = torch.stack((neg_balanced.roll(37), pos_biased.roll(53)))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )
    gathered = torch.gather(inp, 1, res_out.indices.unsqueeze(1)).reshape(-1)
    flag_gems.testing.assert_equal(
        torch.signbit(gathered), torch.signbit(res_out.values)
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_median_f16_key_select_width640_nonlast_and_no_dim(dtype):
    width = 640
    inp_dim0 = torch.randn((width, 3), dtype=dtype, device=flag_gems.device)
    ref_dim0 = torch.median(utils.to_reference(inp_dim0), dim=0)
    with flag_gems.use_gems():
        res_dim0 = torch.median(inp_dim0, dim=0)
    _assert_median_dim_equal(
        res_dim0, ref_dim0, dtype, exact_indices=False, inp=inp_dim0, dim=0
    )

    inp_flat = torch.randn((width,), dtype=dtype, device=flag_gems.device)
    ref_flat = torch.median(utils.to_reference(inp_flat))
    with flag_gems.use_gems():
        res_flat = torch.median(inp_flat)
    utils.gems_assert_equal(res_flat, ref_flat)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [65, 257, 1536, 5001])
def test_median_large_width_nan_first_index(dtype, width):
    inp = torch.randn((3, width), dtype=dtype, device=flag_gems.device)
    expected_indices = torch.tensor([0, width // 2, width - 1], device=flag_gems.device)
    inp[0, 0] = float("nan")
    inp[0, min(width - 1, 17)] = float("nan")
    inp[1, width // 2] = float("nan")
    inp[2, width - 1] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, equal_nan=True, exact_indices=False, inp=inp, dim=1
    )
    flag_gems.testing.assert_equal(res_out.indices, expected_indices)
    assert torch.all(torch.isnan(res_out.values)).item()


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.float32, 257),
        (torch.float32, 4096),
        (torch.float16, 1025),
        (torch.float16, 4096),
        (torch.bfloat16, 1025),
        (torch.bfloat16, 4096),
    ],
)
def test_median_no_dim_fallback_nan(dtype, width):
    inp = torch.randn((width,), dtype=dtype, device=flag_gems.device)
    inp[0] = float("nan")
    inp[width // 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert torch.isnan(res_out).item()


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_non_lastdim_fallback_nan_first_index(dtype, keepdim):
    width = 257
    inp = torch.randn((width, 3), dtype=dtype, device=flag_gems.device)
    expected_indices = torch.tensor([0, width // 2, width - 1], device=flag_gems.device)
    inp[0, 0] = float("nan")
    inp[17, 0] = float("nan")
    inp[width // 2, 1] = float("nan")
    inp[width - 1, 2] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=0, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0, keepdim=keepdim)

    _assert_median_dim_equal(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        exact_indices=False,
        inp=inp,
        dim=0,
        keepdim=keepdim,
    )
    observed_indices = res_out.indices.reshape(-1)
    flag_gems.testing.assert_equal(observed_indices, expected_indices)
    assert torch.all(torch.isnan(res_out.values)).item()


@pytest.mark.median
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32]
)
def test_median_large_width_duplicate_indices_select_value(dtype):
    width = 257
    midpoint = width // 2
    row = torch.arange(width, dtype=torch.int64, device=flag_gems.device)
    row[midpoint + 1] = midpoint
    inp = torch.stack((row, row.flip(0))).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("width", [65, 640, 1536, 2048, 2049])
def test_median_int_lastdim_select_boundaries(dtype, width):
    vals = torch.arange(4 * width, dtype=torch.int64, device=flag_gems.device)
    vals = ((vals * 9973) % (3 * width)) - width
    inp = vals.reshape(4, width).to(dtype)
    info = torch.iinfo(dtype)
    inp[0, 0] = info.min
    inp[0, width - 1] = info.max
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("width", [65, 1536])
def test_median_int_lastdim_select_unique_exact_index(dtype, width):
    rank = (width - 1) // 2
    first = torch.arange(width, dtype=dtype, device=flag_gems.device)
    second = torch.arange(width, dtype=dtype, device=flag_gems.device).roll(7)
    inp = torch.stack((first, second))
    expected_indices = torch.tensor([rank, (rank + 7) % width], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=True, inp=inp, dim=1
    )
    flag_gems.testing.assert_equal(res_out.indices, expected_indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
@pytest.mark.parametrize("width", [65, 2048])
def test_median_int_lastdim_select_all_equal(dtype, width):
    inp = torch.full((3, width), 17, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    _assert_median_dim_equal(
        res_out, ref_out, dtype, exact_indices=False, inp=inp, dim=1
    )


@pytest.mark.median
def test_median_out():
    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_buf = torch.empty((1,), dtype=inp.dtype, device=ref_inp.device)
    out = torch.empty((1,), dtype=inp.dtype, device=flag_gems.device)
    ref_result = torch.ops.aten.median.out(ref_inp, out=ref_buf)
    with flag_gems.use_gems():
        res_result = torch.ops.aten.median.out(inp, out=out)

    utils.gems_assert_equal(res_result, ref_result)
    utils.gems_assert_equal(out, ref_buf)


@pytest.mark.median
def test_median_out_error_paths():
    inp = torch.randn((7,), dtype=torch.float32, device=flag_gems.device)
    bad_dtype = torch.empty((), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ops.aten.median.out(inp, out=bad_dtype)

    if torch.device(flag_gems.device).type == "cuda":
        cpu_out = torch.empty((), dtype=inp.dtype, device="cpu")
        with pytest.raises(RuntimeError):
            with flag_gems.use_gems():
                torch.ops.aten.median.out(inp, out=cpu_out)


@pytest.mark.median
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_median_dim_values_out(keepdim):
    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_values = torch.empty((1,), dtype=inp.dtype, device=ref_inp.device)
    ref_indices = torch.empty((1,), dtype=torch.int64, device=ref_inp.device)
    values = torch.empty((1,), dtype=inp.dtype, device=flag_gems.device)
    indices = torch.empty((1,), dtype=torch.int64, device=flag_gems.device)

    ref_result = torch.ops.aten.median.dim_values(
        ref_inp, 1, keepdim, values=ref_values, indices=ref_indices
    )
    with flag_gems.use_gems():
        res_result = torch.ops.aten.median.dim_values(
            inp, 1, keepdim, values=values, indices=indices
        )

    _assert_median_dim_equal(
        res_result, ref_result, torch.float32, inp=inp, dim=1, keepdim=keepdim
    )
    utils.gems_assert_equal(values, ref_values)
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.median
def test_median_dim_values_out_python_api():
    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_values = torch.empty((1,), dtype=inp.dtype, device=ref_inp.device)
    ref_indices = torch.empty((1,), dtype=torch.int64, device=ref_inp.device)
    values = torch.empty((1,), dtype=inp.dtype, device=flag_gems.device)
    indices = torch.empty((1,), dtype=torch.int64, device=flag_gems.device)

    ref_result = torch.median(ref_inp, dim=1, out=(ref_values, ref_indices))
    with flag_gems.use_gems():
        res_result = torch.median(inp, dim=1, out=(values, indices))

    _assert_median_dim_equal(res_result, ref_result, torch.float32, inp=inp, dim=1)
    utils.gems_assert_equal(values, ref_values)
    utils.gems_assert_equal(indices, ref_indices)


@pytest.mark.median
def test_median_dim_values_out_wrong_device():
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("device mismatch path requires a CUDA input and CPU out tensor")

    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    values = torch.empty((7,), dtype=inp.dtype, device="cpu")
    indices = torch.empty((7,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.median(inp, dim=1, out=(values, indices))


@pytest.mark.median
def test_median_error_paths():
    bool_inp = torch.tensor([True, False, True], device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.median(bool_inp, dim=0)

    inp = torch.randn((2, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(IndexError):
        with flag_gems.use_gems():
            torch.median(inp, dim=3)
