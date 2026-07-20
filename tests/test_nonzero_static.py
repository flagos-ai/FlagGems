import pytest
import torch

import flag_gems
from flag_gems.ops.nonzero_static import nonzero_static, nonzero_static_out

from . import accuracy_utils as utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="nonzero_static Triton implementation requires CUDA",
)


# Cover bool, integer, float, and bf16 dispatch without exploding runtime.
DTYPES = [
    torch.bool,
    torch.int32,
    torch.float16,
    torch.float32,
    torch.bfloat16,
]
COMPLEX_DTYPES = [torch.complex64, torch.complex128]  # complex dtype coverage

CASES = [
    ((), torch.float32, 1.0, 4, -1),
    ((0,), torch.float32, 0.0, 4, 7),
    ((8,), torch.float32, 0.0, 4, -1),
    ((8,), torch.float32, 1.0, 4, -1),
    ((8,), torch.float32, 0.5, 0, -1),
    ((8,), torch.float32, 0.5, 16, 7),
    ((1024,), torch.float32, 0.01, 128, -1),
    ((4, 5), torch.float32, 0.5, 16, 7),
    ((2, 3, 4), torch.float32, 0.1, 16, -1),
    ((2, 3, 4, 5), torch.float32, 0.1, 32, -1),
    ((1, 2, 1, 2, 3), torch.float32, 0.5, 16, -3),
    ((1, 1, 2, 1, 2, 1), torch.int32, 1.0, 2, -1),
]

TARGETED_PATH_CASES = [
    # small-counts path: count kernel + write_small_counts kernel.
    ((16385,), torch.float32, 0.1, 128, -1),
    ((262144,), torch.float32, 0.001, 1024, -1),
    ((32, 1024), torch.float32, 0.1, 1024, -1),
    # cumsum path: num_blocks > SMALL_COUNTS_MAX_BLOCKS.
    ((1048577,), torch.float32, 0.001, 1024, -1),
    ((1048577,), torch.float32, 0.1, 4096, -1),
    # fill_tail path: output padding tail exceeds write kernel coverage.
    ((20000,), torch.float32, 0.001, 30000, -1),
    # multi-block generic path: ndim > 4 and numel > SINGLE_BLOCK_MAX_NUMEL.
    ((2, 2, 2, 2, 2, 1024), torch.float32, 0.01, 128, -1),
    # complex multi-block path.
    ((20000,), torch.complex64, 0.01, 128, -1),
    # padding with zero fill_value.
    ((32, 128), torch.float32, 0.1, 128, 0),
]


def make_input(shape, dtype, nnz_ratio, device):
    if shape == ():
        value = nnz_ratio >= 0.5
        if dtype.is_complex:
            return torch.tensor(1 + 0j if value else 0j, device=device, dtype=dtype)
        if dtype == torch.bool:
            return torch.tensor(value, device=device, dtype=dtype)
        return torch.tensor(1 if value else 0, device=device, dtype=dtype)

    mask = torch.rand(shape, device=device) < nnz_ratio
    x = torch.zeros(shape, device=device, dtype=dtype)
    if dtype.is_complex:
        x[mask] = 1 + 0j
    else:
        x[mask] = 1
    return x


def assert_nonzero_static_matches(shape, dtype, nnz_ratio, size, fill_value):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this CUDA device")

    torch.manual_seed(0)
    x_cpu = make_input(shape, dtype, nnz_ratio, "cpu")
    x_gpu = x_cpu.cuda()

    actual = nonzero_static(x_gpu, size=size, fill_value=fill_value)
    ref_x = utils.to_reference(x_gpu)
    expected = torch.nonzero_static(ref_x, size=size, fill_value=fill_value)
    expected = utils.to_reference(expected)

    assert actual.dtype == torch.int64
    assert tuple(actual.shape) == (size, x_gpu.dim())
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_static_dtypes(dtype):
    assert_nonzero_static_matches((32, 128), dtype, 0.1, 128, -1)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("shape,dtype,nnz_ratio,size,fill_value", CASES)
def test_nonzero_static_cases(shape, dtype, nnz_ratio, size, fill_value):
    assert_nonzero_static_matches(shape, dtype, nnz_ratio, size, fill_value)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("shape,dtype,nnz_ratio,size,fill_value", TARGETED_PATH_CASES)
def test_nonzero_static_targeted_paths(shape, dtype, nnz_ratio, size, fill_value):
    assert_nonzero_static_matches(shape, dtype, nnz_ratio, size, fill_value)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_nonzero_static_complex(dtype):
    x_cpu = torch.zeros((3, 4), dtype=dtype)
    x_cpu[0, 1] = 1 + 0j
    x_cpu[2, 3] = 0 + 2j
    x_gpu = x_cpu.cuda()

    actual = nonzero_static(x_gpu, size=4, fill_value=9)
    ref_x = utils.to_reference(x_gpu)
    expected = torch.nonzero_static(ref_x, size=4, fill_value=9)
    expected = utils.to_reference(expected)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_non_contiguous_transpose():
    torch.manual_seed(1)
    x_cpu_base = make_input((16, 32), torch.float32, 0.2, "cpu")
    x_gpu_base = x_cpu_base.cuda()

    x_gpu_view = x_gpu_base.t()

    actual = nonzero_static(x_gpu_view, size=128, fill_value=-1)
    ref_view = utils.to_reference(x_gpu_view)
    expected = torch.nonzero_static(ref_view, size=128, fill_value=-1)
    expected = utils.to_reference(expected)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_non_contiguous_slice():
    torch.manual_seed(2)
    x_cpu_base = make_input((16, 32), torch.float32, 0.2, "cpu")
    x_gpu_base = x_cpu_base.cuda()

    x_gpu_view = x_gpu_base[:, ::2]

    actual = nonzero_static(x_gpu_view, size=128, fill_value=7)
    ref_view = utils.to_reference(x_gpu_view)
    expected = torch.nonzero_static(ref_view, size=128, fill_value=7)
    expected = utils.to_reference(expected)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_argument_errors():
    x = torch.ones((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(TypeError):
        nonzero_static(x, 4)

    with pytest.raises(TypeError, match="fill_value"):
        nonzero_static(x, size=4, fill_value=1.5)

    with pytest.raises(RuntimeError, match="size must be non-negative"):
        nonzero_static(x, size=-1, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_bool_size():
    x = torch.ones((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(TypeError, match="size"):
        nonzero_static(x, size=True, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_bool_fill_value():
    x = torch.ones((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(TypeError, match="fill_value"):
        nonzero_static(x, size=4, fill_value=True)


@pytest.mark.nonzero_static
def test_nonzero_static_registered_with_use_gems():
    torch.manual_seed(3)
    x_cpu = make_input((4, 5), torch.float32, 0.4, "cpu")
    x_gpu = x_cpu.cuda()
    ref_x = utils.to_reference(x_gpu)
    expected = torch.nonzero_static(ref_x, size=16, fill_value=-1)
    expected = utils.to_reference(expected)

    with flag_gems.use_gems(include=["nonzero_static"]):
        actual = torch.nonzero_static(x_gpu, size=16, fill_value=-1)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_out():
    torch.manual_seed(4)
    x_cpu = make_input((4, 5), torch.float32, 0.4, "cpu")
    x_gpu = x_cpu.cuda()

    expected_out = torch.empty((1, 1), device=x_gpu.device, dtype=torch.int64)
    expected = torch.nonzero_static(x_gpu, size=16, fill_value=7, out=expected_out)
    expected = utils.to_reference(expected)

    actual_out = torch.empty((1, 1), device=x_gpu.device, dtype=torch.int64)
    actual = nonzero_static_out(x_gpu, size=16, fill_value=7, out=actual_out)

    assert actual is actual_out
    assert actual.dtype == torch.int64
    assert tuple(actual.shape) == tuple(expected.shape)
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
def test_nonzero_static_out_registered_with_use_gems():
    torch.manual_seed(5)
    x_cpu = make_input((4, 5), torch.float32, 0.4, "cpu")
    x_gpu = x_cpu.cuda()

    expected_out = torch.empty((1, 1), device=x_gpu.device, dtype=torch.int64)
    expected = torch.nonzero_static(x_gpu, size=16, fill_value=-1, out=expected_out)
    expected = utils.to_reference(expected)

    actual_out = torch.empty((1, 1), device=x_gpu.device, dtype=torch.int64)
    with flag_gems.use_gems(include=["nonzero_static_out"]):
        actual = torch.nonzero_static(x_gpu, size=16, fill_value=-1, out=actual_out)

    assert actual is actual_out
    assert tuple(actual.shape) == tuple(expected.shape)
    utils.gems_assert_equal(actual, expected)
