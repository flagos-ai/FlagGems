import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="nonzero_static Triton-Ascend implementation requires NPU",
)

DTYPES = [
    torch.bool,
    torch.int32,
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

CASES = [
    ((), torch.float32, 1.0, 4, -1),
    ((0,), torch.float32, 0.0, 4, 7),
    ((8,), torch.float32, 0.0, 4, -1),
    ((8,), torch.float32, 1.0, 4, -1),
    ((8,), torch.float32, 0.5, 0, -1),
    ((8,), torch.float32, 0.5, 16, 7),
    ((16385,), torch.float32, 0.1, 128, -1),
    ((262144,), torch.float32, 0.01, 128, -1),
    ((262144,), torch.bfloat16, 0.0001, 128, 7),
    ((512, 512), torch.float32, 0.01, 128, -1),
    ((1048577,), torch.float32, 0.001, 1024, -1),
    ((2, 3, 4), torch.float32, 0.1, 16, -1),
    ((2, 3, 4, 5), torch.float32, 0.1, 32, -1),
    ((2, 2, 2, 2, 2, 1024), torch.float32, 0.01, 128, -1),
]


def make_input(shape, dtype, nnz_ratio):
    if shape == ():
        return torch.tensor(
            1 if nnz_ratio >= 0.5 else 0,
            dtype=dtype,
        )
    mask = torch.rand(shape) < nnz_ratio
    x = torch.zeros(shape, dtype=dtype)
    x[mask] = 1
    return x


def assert_matches(shape, dtype, nnz_ratio, size, fill_value):
    torch.manual_seed(0)
    x_cpu = make_input(shape, dtype, nnz_ratio)
    expected = torch.nonzero_static(
        utils.to_reference(x_cpu), size=size, fill_value=fill_value
    )
    actual = flag_gems.nonzero_static(
        x_cpu.to(flag_gems.device),
        size=size,
        fill_value=fill_value,
    )
    assert actual.dtype == torch.int64
    assert tuple(actual.shape) == (size, len(shape))
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_static_ascend_dtypes(dtype):
    assert_matches((32, 128), dtype, 0.1, 128, -1)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("shape,dtype,nnz_ratio,size,fill_value", CASES)
def test_nonzero_static_ascend_cases(shape, dtype, nnz_ratio, size, fill_value):
    assert_matches(shape, dtype, nnz_ratio, size, fill_value)


@pytest.mark.nonzero_static
def test_nonzero_static_ascend_non_contiguous():
    torch.manual_seed(1)
    x_cpu = make_input((16, 32), torch.float32, 0.2).t()
    expected = torch.nonzero_static(utils.to_reference(x_cpu), size=128, fill_value=7)
    actual = flag_gems.nonzero_static(
        x_cpu.to(flag_gems.device), size=128, fill_value=7
    )
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_ascend_registered():
    torch.manual_seed(2)
    x_cpu = make_input((4, 5), torch.float32, 0.4)
    x_npu = x_cpu.to(flag_gems.device)
    expected = torch.nonzero_static(utils.to_reference(x_cpu), size=16, fill_value=-1)
    with flag_gems.use_gems(include=["nonzero_static"]):
        actual = torch.nonzero_static(x_npu, size=16, fill_value=-1)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_ascend_out():
    torch.manual_seed(3)
    x_cpu = make_input((4, 5), torch.float32, 0.4)
    x_npu = x_cpu.to(flag_gems.device)
    expected = torch.nonzero_static(utils.to_reference(x_cpu), size=16, fill_value=7)
    out = torch.empty((1, 1), device=flag_gems.device, dtype=torch.int64)
    with flag_gems.use_gems(include=["nonzero_static_out"]):
        actual = torch.nonzero_static(x_npu, size=16, fill_value=7, out=out)
    assert actual is out
    assert tuple(actual.shape) == (16, 2)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_ascend_sparse_group_fallback():
    x_cpu = torch.zeros(8193, dtype=torch.bfloat16)
    x_cpu[[1, 2, 3, 40, 41, 2050]] = 1
    expected = torch.nonzero_static(utils.to_reference(x_cpu), size=4, fill_value=-1)
    actual = flag_gems.nonzero_static(x_cpu.to(flag_gems.device), size=4, fill_value=-1)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_ascend_bfloat16_special_values():
    x_cpu = torch.zeros(8193, dtype=torch.bfloat16)
    x_cpu[:7] = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0],
        dtype=torch.bfloat16,
    )
    expected = torch.nonzero_static(utils.to_reference(x_cpu), size=128, fill_value=-1)
    actual = flag_gems.nonzero_static(
        x_cpu.to(flag_gems.device), size=128, fill_value=-1
    )
    utils.gems_assert_equal(actual, expected)
