import importlib
import inspect
import random

import pytest
import torch

import flag_gems

hopper_mul = importlib.import_module("flag_gems.runtime.backend._nvidia.hopper.ops.mul")


def _cuda_available():
    return torch.cuda.is_available() and flag_gems.device == "cuda"


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA-like Hopper mul tests require CUDA backend"
)


def _assert_both_paths_match_torch(a, b, *, equal_nan=False):
    expected = torch.mul(a, b)
    actual_direct = flag_gems.mul(a, b)
    with flag_gems.use_gems():
        actual_dispatch = torch.mul(a, b)
    torch.testing.assert_close(actual_direct, expected, equal_nan=equal_nan)
    torch.testing.assert_close(actual_dispatch, expected, equal_nan=equal_nan)


def test_hopper_mul_does_not_use_pointwise_dynamic():
    source = inspect.getsource(hopper_mul)
    assert "pointwise_dynamic" not in source


def test_no_dead_8d_kernels():
    source = inspect.getsource(hopper_mul)
    assert "mul_generic_8d_kernel" not in source
    assert "mul_complex_generic_8d_kernel" not in source


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((4, 1), (4, 8)),
        ((1, 8), (4, 8)),
        ((2, 1, 4), (1, 3, 4)),
        ((), (3, 5)),
    ],
)
def test_mul_tensor_broadcast_matches_torch(a_shape, b_shape):
    a = torch.randn(a_shape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(b_shape, device="cuda", dtype=torch.bfloat16)
    _assert_both_paths_match_torch(a, b)


def test_mul_qwen_hot_column_broadcast_matches_torch():
    a = torch.randn((512, 1), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 2048), device="cuda", dtype=torch.bfloat16)
    _assert_both_paths_match_torch(a, b)


def test_mul_tensor_scalar_and_scalar_tensor_match_torch():
    x = torch.randn((7, 11), device="cuda", dtype=torch.bfloat16)
    _assert_both_paths_match_torch(x, 0.5)
    _assert_both_paths_match_torch(0.5, x)


def test_mul_non_contiguous_broadcast_matches_torch():
    a = torch.randn((8, 4), device="cuda", dtype=torch.float32).t()
    b = torch.randn((1, 8), device="cuda", dtype=torch.float32)
    _assert_both_paths_match_torch(a, b)


def test_mul_complex_matches_torch():
    a = torch.randn((3, 1), device="cuda", dtype=torch.complex64)
    b = torch.randn((3, 5), device="cuda", dtype=torch.complex64)
    _assert_both_paths_match_torch(a, b)
    _assert_both_paths_match_torch(a, 1.5 - 0.25j)


def test_mul_inplace_broadcast_matches_torch():
    x_expected = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    x_actual = x_expected.clone()
    y = torch.randn((1, 8), device="cuda", dtype=torch.float32)
    x_expected.mul_(y)
    flag_gems.mul_(x_actual, y)
    torch.testing.assert_close(x_actual, x_expected)


def test_mul_inplace_dispatch_matches_torch():
    x_expected = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    x_actual = x_expected.clone()
    y = torch.randn((1, 8), device="cuda", dtype=torch.float32)
    x_expected.mul_(y)
    with flag_gems.use_gems():
        x_actual.mul_(y)
    torch.testing.assert_close(x_actual, x_expected)


@pytest.mark.parametrize(
    "dtype,scalar",
    [
        (torch.bool, True),
        (torch.int32, 3),
        (torch.int64, -7),
        (torch.float64, -0.25),
    ],
)
def test_mul_dtype_matrix_matches_torch(dtype, scalar):
    if dtype == torch.bool:
        a = torch.tensor([[True, False], [False, True]], device="cuda", dtype=dtype)
        b = torch.tensor([[True, True], [False, False]], device="cuda", dtype=dtype)
    elif dtype.is_floating_point:
        a = torch.randn((2, 3), device="cuda", dtype=dtype)
        b = torch.randn((2, 3), device="cuda", dtype=dtype)
    else:
        a = torch.randint(-8, 8, (2, 3), device="cuda", dtype=dtype)
        b = torch.randint(-8, 8, (2, 3), device="cuda", dtype=dtype)
    _assert_both_paths_match_torch(a, b)
    _assert_both_paths_match_torch(a, scalar)


def test_mul_empty_tensor_matches_torch():
    a = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    b = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    _assert_both_paths_match_torch(a, b)


def test_mul_special_values_match_torch():
    a = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 3.0],
        device="cuda",
        dtype=torch.float32,
    )
    b = torch.tensor([2.0, 0.0, -1.0, 5.0, float("nan")], device="cuda")
    _assert_both_paths_match_torch(a, b, equal_nan=True)


def _make_random_broadcast_case(rng: random.Random):
    ndim = rng.randint(1, 4)
    out_shape = [rng.randint(1, 5) for _ in range(ndim)]

    def make_shape():
        rank = rng.randint(0, ndim)
        tail = []
        for dim in out_shape[ndim - rank :]:
            tail.append(1 if rng.random() < 0.4 else dim)
        return tuple(tail)

    a_shape = make_shape()
    b_shape = make_shape()
    if a_shape == ():
        a = torch.tensor(rng.uniform(-2.0, 2.0), device="cuda", dtype=torch.float32)
    else:
        a = torch.randn(a_shape, device="cuda", dtype=torch.float32)
    if b_shape == ():
        b = torch.tensor(rng.uniform(-2.0, 2.0), device="cuda", dtype=torch.float32)
    else:
        b = torch.randn(b_shape, device="cuda", dtype=torch.float32)

    if isinstance(a, torch.Tensor) and a.ndim >= 1 and a.numel() > 0 and rng.random() < 0.5:
        a = torch.randn((*a.shape, 2), device="cuda", dtype=torch.float32)[..., 0]
    if isinstance(b, torch.Tensor) and b.ndim >= 1 and b.numel() > 0 and rng.random() < 0.5:
        b = torch.randn((*b.shape, 2), device="cuda", dtype=torch.float32)[..., 0]
    return a, b


def test_mul_random_broadcast_property_matches_torch():
    rng = random.Random(0)
    for _ in range(40):
        a, b = _make_random_broadcast_case(rng)
        _assert_both_paths_match_torch(a, b)


def test_mul_high_rank_broadcast_matches_torch():
    a = torch.randn((1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), device="cuda")
    b = torch.randn((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3), device="cuda")
    _assert_both_paths_match_torch(a, b)


def test_mul_high_rank_complex_matches_torch():
    a = torch.randn(
        (1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        device="cuda",
        dtype=torch.complex64,
    )
    b = torch.randn(
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        device="cuda",
        dtype=torch.complex64,
    )
    _assert_both_paths_match_torch(a, b)


def test_mul_invalid_broadcast_raises_runtime_error():
    a = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    b = torch.randn((4, 5), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems.mul(a, b)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.mul(a, b)


def test_mul_inplace_dtype_mismatch_raises_runtime_error():
    a = torch.ones((2, 3), device="cuda", dtype=torch.int32)
    b = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        flag_gems.mul_(a, b)


def test_mul_dispatch_out_matches_torch():
    a = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    b = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    expected = torch.mul(a, b, out=torch.empty_like(a))
    with flag_gems.use_gems():
        actual = torch.mul(a, b, out=torch.empty_like(a))
    torch.testing.assert_close(actual, expected)
