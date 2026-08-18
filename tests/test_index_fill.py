import os
import subprocess
import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

INDEX_FILL_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4, 8), (2, 3, 5)]
DIM_LIST = [1] if QUICK_MODE else [0, -1]
INDEX_CASES = ["normal", "negative", "scalar"]
_INDEX_FILL_DTYPES = utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
INDEX_FILL_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.device == "npu" and dtype == torch.int16,
            reason="torch_npu does not support int16 index_fill reference",
        ),
    )
    for dtype in _INDEX_FILL_DTYPES
]
INDEX_FILL_OPS = [
    "index_fill_scalar",
    "index_fill_scalar_",
    "index_fill_scalar_out",
    "index_fill_tensor",
    "index_fill_tensor_",
    "index_fill_tensor_out",
]
INDEX_FILL_OOB_PATHS = ("python_contiguous", "python_strided")


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=flag_gems.device)


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return -3.5
    return -3


def _make_index(dim_size, case):
    if case == "normal":
        values = [0, dim_size - 1] if dim_size > 1 else [0]
        return torch.tensor(values, dtype=torch.long, device=flag_gems.device)
    if case == "negative":
        return torch.tensor([-1], dtype=torch.long, device=flag_gems.device)
    if case == "scalar":
        return torch.tensor(0, dtype=torch.long, device=flag_gems.device)
    raise ValueError(f"Unknown index case: {case}")


def _to_ref_value(value):
    if isinstance(value, torch.Tensor):
        return utils.to_reference(value, False)
    return value


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend bounds-check dispatch is only used on NPU",
)
@pytest.mark.parametrize(
    ("index_numel", "expected_host_check"),
    ((512, True), (2048, True), (4096, True), (8192, False)),
)
def test_index_fill_ascend_host_bounds_check_threshold(
    index_numel, expected_host_check
):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    index = torch.empty(index_numel, dtype=torch.long, device=flag_gems.device)
    assert (
        ascend_index_fill._should_use_ascend_host_index_check(index)
        is expected_host_check
    )


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend bounds-check cache is only used on NPU",
)
def test_index_fill_ascend_bounds_check_cache_revalidates_mutated_index(monkeypatch):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    inp = torch.empty((4, 8), dtype=torch.float16, device=flag_gems.device)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    original_aminmax = ascend_index_fill.torch.aminmax
    calls = 0

    def counted_aminmax(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_aminmax(*args, **kwargs)

    ascend_index_fill._ascend_index_validation_cache.clear()
    monkeypatch.setattr(ascend_index_fill.torch, "aminmax", counted_aminmax)

    dim, prepared, bounds_checked, has_negative, _ = (
        ascend_index_fill._prepare_ascend_index(inp, 0, index)
    )
    assert (dim, prepared is index, bounds_checked, has_negative) == (
        0,
        True,
        True,
        True,
    )
    ascend_index_fill._prepare_ascend_index(inp, 0, index)
    assert calls == 1

    index.copy_(torch.tensor([0, 4], dtype=torch.long, device=flag_gems.device))
    with pytest.raises(IndexError, match="index out of range in self"):
        ascend_index_fill._prepare_ascend_index(inp, 0, index)
    assert calls == 2
    ascend_index_fill._ascend_index_validation_cache.clear()


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend membership cache is only used on NPU",
)
def test_index_fill_ascend_membership_cache_revalidates_mutated_index():
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    out = torch.empty((1024, 4096), dtype=torch.float16, device=flag_gems.device)
    index = torch.tensor([0, 2, 7], dtype=torch.long, device=flag_gems.device)

    ascend_index_fill._ascend_membership_cache.clear()
    ascend_index_fill._ascend_membership_cache_bytes = 0
    first = ascend_index_fill._build_contiguous_membership_mask(
        out, index, has_negative=False, dim_size=4096
    )
    second = ascend_index_fill._build_contiguous_membership_mask(
        out, index, has_negative=False, dim_size=4096
    )
    assert second is first

    index.copy_(torch.tensor([1, 3, 9], dtype=torch.long, device=flag_gems.device))
    third = ascend_index_fill._build_contiguous_membership_mask(
        out, index, has_negative=False, dim_size=4096
    )
    assert third is not first
    expected = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    assert torch.equal(third.cpu()[:10], expected)
    ascend_index_fill._ascend_membership_cache.clear()
    ascend_index_fill._ascend_membership_cache_bytes = 0


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend transpose-fill dispatch is only used on NPU",
)
@pytest.mark.parametrize(
    ("index_numel", "expected_transpose_fill"),
    ((512, False), (1024, True)),
)
def test_index_fill_ascend_transpose_fill_program_threshold(
    index_numel, expected_transpose_fill
):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    inp = torch.empty((200, 40999, 3), dtype=torch.float16, device=flag_gems.device)
    index = torch.empty(index_numel, dtype=torch.long, device=flag_gems.device)
    assert (
        ascend_index_fill._can_use_contiguous_high_density_transpose_fill(
            inp, 1, index, value_is_tensor=False, bounds_checked=True
        )
        is expected_transpose_fill
    )


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend transpose-fill dispatch is only used on NPU",
)
@pytest.mark.parametrize(
    ("index_numel", "expected_transpose_fill"),
    ((2047, False), (2048, True)),
)
def test_index_fill_ascend_transpose_fill_high_density_inner1(
    index_numel, expected_transpose_fill
):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    inp = torch.empty((4096, 4096), dtype=torch.float16, device=flag_gems.device)
    index = torch.empty(index_numel, dtype=torch.long, device=flag_gems.device)
    assert (
        ascend_index_fill._can_use_contiguous_high_density_transpose_fill(
            inp, 1, index, value_is_tensor=False, bounds_checked=True
        )
        is expected_transpose_fill
    )


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend transpose-fill dispatch is only used on NPU",
)
@pytest.mark.parametrize(
    ("index_numel", "expected_transpose_fill"),
    ((128, False), (256, True)),
)
def test_index_fill_ascend_transpose_fill_small_full_dim(
    index_numel, expected_transpose_fill
):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    inp = torch.empty((4096, 256), dtype=torch.float16, device=flag_gems.device)
    index = torch.empty(index_numel, dtype=torch.long, device=flag_gems.device)
    assert (
        ascend_index_fill._can_use_contiguous_high_density_transpose_fill(
            inp, 1, index, value_is_tensor=False, bounds_checked=True
        )
        is expected_transpose_fill
    )


@pytest.mark.index_fill
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend full-coverage fill dispatch is only used on NPU",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_index_fill_ascend_full_coverage_fill_and_duplicate_fallback(dtype):
    from flag_gems.runtime.backend._ascend.ops import index_fill as ascend_index_fill

    shape = (16, 64, 3)
    value = 3.14159
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index = torch.randperm(shape[1], device=flag_gems.device)
    index[::7] -= shape[1]

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)
    actual = ascend_index_fill.index_fill_scalar(inp, 1, index, value)
    utils.gems_assert_equal(actual, ref_out)

    inplace = inp.clone()
    ref_inplace = utils.to_reference(inplace, False)
    ref_inplace.index_fill_(1, ref_index, value)
    ascend_index_fill.index_fill_scalar_(inplace, 1, index, value)
    utils.gems_assert_equal(inplace, ref_inplace)

    duplicate = torch.randint(0, shape[1] // 2, (shape[1],), device=flag_gems.device)
    ref_duplicate = ref_inp.index_fill(1, utils.to_reference(duplicate, False), value)
    duplicate_actual = ascend_index_fill.index_fill_scalar(inp, 1, duplicate, value)
    utils.gems_assert_equal(duplicate_actual, ref_duplicate)


@pytest.mark.index_fill
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(dim, index, value)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out is not inp


@pytest.mark.index_fill_
@pytest.mark.parametrize("shape", INDEX_FILL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("index_case", INDEX_CASES)
def test_index_fill_scalar_(shape, dim, dtype, index_case):
    inp = _make_input(shape, dtype)
    dim = dim % inp.ndim
    index = _make_index(inp.size(dim), index_case)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(dim, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill_(dim, index, value)

    assert res_out is inp
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
@pytest.mark.parametrize("value_device", ["device", "cpu"])
def test_index_fill_tensor_value(dtype, value_device):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([1, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(
        _scalar_value(dtype),
        dtype=dtype,
        device=flag_gems.device if value_device == "device" else "cpu",
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_fill_
def test_index_fill_duplicate_index():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.tensor([1, 1, -1], dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_empty_index_noop():
    inp = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    index = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_index = utils.to_reference(index, False)
    ref_inp.index_fill_(1, ref_index, -7.0)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        inp.index_fill_(1, index, -7.0)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_fill_
def test_index_fill_noncontiguous_view():
    base = torch.arange(12, dtype=torch.float32, device=flag_gems.device).reshape(3, 4)
    ref_base = utils.to_reference(base.clone(), False)
    res_base = base.clone()
    ref_view = ref_base.t()
    res_view = res_base.t()
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    ref_index = utils.to_reference(index, False)

    ref_view.index_fill_(1, ref_index, -8.0)
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = res_view.index_fill_(1, index, -8.0)

    assert res is res_view
    utils.gems_assert_equal(res_base, ref_base)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_fill_contiguous_inner3_fast_path(dtype):
    inp = _make_input((4, 17, 3), dtype)
    index = torch.tensor([0, 1, 8, -1], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res_out = inp.index_fill(1, index, value)
        inplace = inp.clone()
        res_inplace = inplace.index_fill_(1, index, value)

    assert res_out is not inp
    assert res_inplace is inplace
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend high-density transpose fill is only used on NPU",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_index_fill_high_density_inner1_transpose_path(dtype):
    inp = _make_input((256, 4096), dtype)
    index = torch.cat(
        (
            torch.tensor([-1, -1], dtype=torch.long, device=flag_gems.device),
            torch.arange(2046, dtype=torch.long, device=flag_gems.device),
        )
    )
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        result = inplace.index_fill_(1, index, value)

    assert actual is not inp
    assert result is inplace
    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu",
    reason="Ascend small full-dim transpose fill is only used on NPU",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_index_fill_small_full_dim_transpose_path(dtype):
    inp = _make_input((4096, 256), dtype)
    index = torch.cat(
        (
            torch.tensor([-1, -1], dtype=torch.long, device=flag_gems.device),
            torch.arange(254, dtype=torch.long, device=flag_gems.device),
        )
    )
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        result = inplace.index_fill_(1, index, value)

    assert actual is not inp
    assert result is inplace
    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.parametrize("value_is_tensor", [False, True])
def test_index_fill_large_contiguous_membership_functional(value_is_tensor):
    inp = _make_input((1024, 1024), torch.float16)
    index = torch.arange(16, dtype=torch.long, device=flag_gems.device)
    value = (
        torch.tensor(-3.5, dtype=inp.dtype, device=flag_gems.device)
        if value_is_tensor
        else -3.5
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(1, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)

    assert actual is not inp
    utils.gems_assert_equal(actual, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
def test_index_fill_large_contiguous_membership_duplicate_index():
    inp = _make_input((1024, 1024), torch.float16)
    base_index = torch.arange(127, dtype=torch.long, device=flag_gems.device)
    index = torch.cat(
        (
            base_index,
            base_index,
            torch.tensor([-1], dtype=torch.long, device=flag_gems.device),
        )
    )
    value = -3.5

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        inplace.index_fill_(1, index, value)

    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("shape", ((64, 257), (32, 17, 3)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("value_is_tensor", (False, True))
def test_index_fill_dim0_row_path_negative_duplicate(shape, dtype, value_is_tensor):
    inp = _make_input(shape, dtype)
    index = torch.tensor([0, 7, 7, -1, 31], dtype=torch.long, device=flag_gems.device)
    value = (
        torch.tensor(-3.5, dtype=dtype, device=flag_gems.device)
        if value_is_tensor
        else -3.5
    )

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_value = _to_ref_value(value)
    ref_out = ref_inp.index_fill(0, ref_index, ref_value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(0, index, value)
        inplace = inp.clone()
        inplace.index_fill_(0, index, value)

    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "shape",
    (
        (32, 64, 2),
        (9, 37, 2),
        (5, 37, 2, 1),
        (32, 64, 3),
        (9, 37, 3),
        (5, 37, 3, 1),
        (32, 64, 4),
        (9, 37, 4),
        (5, 37, 2, 2),
    ),
)
def test_index_fill_small_inner_blocked_updates(dtype, shape):
    inp = _make_input(shape, dtype)
    base_index = torch.arange(30, dtype=torch.long, device=flag_gems.device)
    index = torch.cat(
        (
            base_index,
            base_index[:1],
            torch.tensor([-1], dtype=torch.long, device=flag_gems.device),
        )
    )
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        inplace.index_fill_(1, index, value)

    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("index_value", (0, -1))
def test_index_fill_single_index_small_inner_updates(dtype, index_value):
    inp = _make_input((32, 64), dtype)
    index = torch.tensor([index_value], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)

    ref_inp = utils.to_reference(inp, False)
    ref_index = utils.to_reference(index, False)
    ref_out = ref_inp.index_fill(1, ref_index, value)

    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        actual = inp.index_fill(1, index, value)
        inplace = inp.clone()
        inplace.index_fill_(1, index, value)

    utils.gems_assert_equal(actual, ref_out)
    utils.gems_assert_equal(inplace, ref_out)


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_scalar_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = _scalar_value(dtype)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Scalar_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        value,
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Scalar_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_out
@pytest.mark.parametrize("dtype", INDEX_FILL_DTYPES)
def test_index_fill_tensor_out(dtype):
    inp = _make_input((3, 4), dtype)
    index = torch.tensor([0, -1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor(_scalar_value(dtype), dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_out = torch.empty_like(utils.to_reference(inp, False))

    ref = torch.ops.aten.index_fill.int_Tensor_out(
        utils.to_reference(inp, False),
        1,
        utils.to_reference(index, False),
        utils.to_reference(value, False),
        out=ref_out,
    )
    with flag_gems.use_gems(include=INDEX_FILL_OPS):
        res = torch.ops.aten.index_fill.int_Tensor_out(inp, 1, index, value, out=out)

    assert res is out
    utils.gems_assert_equal(res, ref)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_dtype():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.int32, device=flag_gems.device)
    with (
        flag_gems.use_gems(include=INDEX_FILL_OPS),
        pytest.raises(IndexError, match="Expected dtype int64"),
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_index_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([[1]], dtype=torch.long, device=flag_gems.device)
    with (
        flag_gems.use_gems(include=INDEX_FILL_OPS),
        pytest.raises(IndexError, match="Index is supposed to be a vector"),
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill
@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device not in ("cuda", "npu"),
    reason="out-of-range behavior is backend-specific",
)
@pytest.mark.parametrize("op_name", ("index_fill", "index_fill_"))
@pytest.mark.parametrize("execution_path", INDEX_FILL_OOB_PATHS)
def test_index_fill_out_of_range_index_device_assert(op_name, execution_path):
    if execution_path == "python_strided":
        input_setup = (
            "inp = torch.zeros((3, 4), device=flag_gems.device).t()\n"
            "dim = 1\n"
            "index_value = 3"
        )
    else:
        input_setup = (
            "inp = torch.zeros((3, 4), device=flag_gems.device)\n"
            "dim = 1\n"
            "index_value = 4"
        )
    operation = (
        "inp = inp.index_fill(dim, index, 1.0)"
        if op_name == "index_fill"
        else "inp.index_fill_(dim, index, 1.0)"
    )

    source_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    child_code = f"""
import sys
sys.path.insert(0, {source_dir!r})

import torch
import flag_gems
from flag_gems.runtime import torch_device_fn

{input_setup}
index = torch.tensor([index_value], dtype=torch.long, device=flag_gems.device)
ops = {INDEX_FILL_OPS!r}
try:
    with flag_gems.use_gems(include=ops):
        {operation}
        torch_device_fn.synchronize()
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (
        "device-side assert" in output
        or "index out of bounds" in output
        or "index out of range" in output
    )


@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device != "npu", reason="NPU validates index bounds on the host"
)
def test_index_fill_out_of_range_index():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([4], dtype=torch.long, device=flag_gems.device)
    with (
        flag_gems.use_gems(include=INDEX_FILL_OPS),
        pytest.raises(IndexError, match="index out of range"),
    ):
        inp.index_fill_(1, index, -1.0)


@pytest.mark.index_fill_
def test_index_fill_invalid_tensor_value_ndim():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device=flag_gems.device)
    value = torch.tensor([1.0], device=flag_gems.device)
    with (
        flag_gems.use_gems(include=INDEX_FILL_OPS),
        pytest.raises(RuntimeError, match="0-dimensional value tensor"),
    ):
        inp.index_fill_(1, index, value)


@pytest.mark.index_fill_
@pytest.mark.skipif(
    flag_gems.device == "cpu", reason="device mismatch requires device backend"
)
def test_index_fill_cpu_index_rejected():
    inp = torch.randn((3, 4), device=flag_gems.device)
    index = torch.tensor([1], dtype=torch.long, device="cpu")
    with (
        flag_gems.use_gems(include=INDEX_FILL_OPS),
        pytest.raises(RuntimeError, match="same device"),
    ):
        inp.index_fill_(1, index, -1.0)
