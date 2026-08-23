# Copyright 2026, The FlagOS Contributors.
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

from . import accuracy_utils as utils
from . import conftest as cfg

IS_ASCEND = flag_gems.vendor_name == "ascend"
# Nonzero_static is not supported for cuda <= 11.4
COMPOSED_REFERENCE_VENDORS = ("ascend", "hygon", "iluvatar")
CUDA_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="nonzero_static complex and CUDA kernel paths require NVIDIA",
)
ASCEND_ONLY = pytest.mark.skipif(
    not IS_ASCEND,
    reason="nonzero_static Ascend paths require NPU",
)
FULL_ONLY = pytest.mark.skipif(
    cfg.QUICK_MODE,
    reason="large nonzero_static path cases are excluded from quick mode",
)

BASE_DTYPES = [torch.bool, torch.int32]  # Cover boolean and integer dispatch.
DTYPES = (
    [torch.float32]
    if cfg.QUICK_MODE
    else list(dict.fromkeys([*BASE_DTYPES, *utils.FLOAT_DTYPES]))
)
CUDA_COMPLEX_DTYPES = [  # Complex input is supported by the CUDA path.
    torch.complex64,
    torch.complex128,
]

QUICK_CASES = [
    ((), torch.float32, 1.0, 4, -1),
    ((0,), torch.float32, 0.0, 4, 7),
    ((8,), torch.float32, 1.0, 4, -1),
    ((8,), torch.float32, 0.5, 0, -1),
    ((8,), torch.float32, 0.5, 16, 7),
    ((2, 3, 4), torch.float32, 0.1, 16, -1),
    ((1, 2, 1, 2, 3), torch.float32, 0.5, 16, -3),
]

FULL_CASES = [
    ((8,), torch.float32, 0.0, 4, -1),
    ((1024,), torch.float32, 0.01, 128, -1),
    ((4, 5), torch.float32, 0.5, 16, 7),
    ((2, 3, 4, 5), torch.float32, 0.1, 32, -1),
    ((1, 1, 2, 1, 2, 1), torch.int32, 1.0, 2, -1),
    ((2, 2, 2, 2, 2, 1024), torch.float32, 0.01, 128, -1),
    ((32, 128), torch.float32, 0.1, 128, 0),
]

CASES = QUICK_CASES if cfg.QUICK_MODE else QUICK_CASES + FULL_CASES

CUDA_PATH_CASES = [
    ((16385,), torch.float32, 0.1, 128, -1),
    ((262144,), torch.float32, 0.001, 1024, -1),
    ((32, 1024), torch.float32, 0.1, 1024, -1),
    ((1048577,), torch.float32, 0.001, 1024, -1),
    ((1048577,), torch.float32, 0.1, 4096, -1),
    ((20000,), torch.float32, 0.001, 30000, -1),
    ((20000,), torch.complex64, 0.01, 128, -1),
]

ASCEND_PATH_CASES = [
    ((1048577,), torch.float32, 0.001, 1024, -1),
    ((128, 4096), torch.float32, 0.01, 4096, -1),
]


def make_input(shape, dtype, nnz_ratio):
    device = flag_gems.device
    if shape == ():
        value = nnz_ratio >= 0.5
        if dtype.is_complex:
            return torch.tensor(1 + 0j if value else 0j, device=device, dtype=dtype)
        if dtype == torch.bool:
            return torch.tensor(value, device=device, dtype=dtype)
        return torch.tensor(1 if value else 0, device=device, dtype=dtype)

    mask = torch.rand(shape, device=device) < nnz_ratio
    input = torch.zeros(shape, device=device, dtype=dtype)
    input[mask] = 1 + 0j if dtype.is_complex else 1
    return input


def _composed_nonzero_static_reference(input, size, fill_value):
    out = torch.full(
        (size, input.dim()),
        fill_value,
        dtype=torch.int64,
        device=input.device,
    )
    if size == 0 or input.dim() == 0:
        return out

    indices = torch.nonzero(input)
    copy_len = min(size, indices.shape[0])
    if copy_len > 0:
        out[:copy_len].copy_(indices[:copy_len])
    return out


def assert_nonzero_static_matches(input, size, fill_value):
    ref_input = utils.to_reference(input)
    if flag_gems.vendor_name in COMPOSED_REFERENCE_VENDORS:
        expected = _composed_nonzero_static_reference(ref_input, size, fill_value)
    else:
        expected = torch.nonzero_static(
            ref_input,
            size=size,
            fill_value=fill_value,
        )

    with flag_gems.use_gems(include=["nonzero_static"]):
        actual = torch.nonzero_static(
            input,
            size=size,
            fill_value=fill_value,
        )

    assert actual.dtype == torch.int64
    assert tuple(actual.shape) == (size, input.dim())
    utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_static_dtypes(dtype):
    torch.manual_seed(0)
    input = make_input((32, 128), dtype, 0.1)
    assert_nonzero_static_matches(input, size=128, fill_value=-1)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("shape,dtype,nnz_ratio,size,fill_value", CASES)
def test_nonzero_static_cases(shape, dtype, nnz_ratio, size, fill_value):
    torch.manual_seed(0)
    input = make_input(shape, dtype, nnz_ratio)
    assert_nonzero_static_matches(input, size=size, fill_value=fill_value)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("view_kind", ["transpose", "slice"])
def test_nonzero_static_non_contiguous(view_kind):
    torch.manual_seed(1)
    base = make_input((16, 32), torch.float32, 0.2)
    input = base.t() if view_kind == "transpose" else base[:, ::2]
    fill_value = -1 if view_kind == "transpose" else 7
    assert_nonzero_static_matches(input, size=128, fill_value=fill_value)


@pytest.mark.nonzero_static
def test_nonzero_static_argument_errors():
    input = torch.ones((8,), device=flag_gems.device, dtype=torch.float32)

    with pytest.raises(TypeError):
        flag_gems.nonzero_static(input, 4)

    with pytest.raises(TypeError, match="fill_value"):
        flag_gems.nonzero_static(input, size=4, fill_value=1.5)

    with pytest.raises(RuntimeError, match="size must be non-negative"):
        flag_gems.nonzero_static(input, size=-1, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_bool_arguments():
    input = torch.ones((8,), device=flag_gems.device, dtype=torch.float32)

    with pytest.raises(TypeError, match="size"):
        flag_gems.nonzero_static(input, size=True, fill_value=-1)

    with pytest.raises(TypeError, match="fill_value"):
        flag_gems.nonzero_static(input, size=4, fill_value=True)


@pytest.mark.nonzero_static_out
def test_nonzero_static_out():
    torch.manual_seed(2)
    input = make_input((4, 5), torch.float32, 0.4)
    assert_nonzero_static_matches(input, size=16, fill_value=-7)


@CUDA_ONLY
@FULL_ONLY
@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", CUDA_COMPLEX_DTYPES)
def test_nonzero_static_cuda_complex(dtype):
    input = torch.zeros((3, 4), dtype=dtype, device=flag_gems.device)
    input[0, 1] = 1 + 0j
    input[2, 3] = 0 + 2j
    assert_nonzero_static_matches(input, size=4, fill_value=9)


@CUDA_ONLY
@FULL_ONLY
@pytest.mark.nonzero_static
@pytest.mark.parametrize(
    "shape,dtype,nnz_ratio,size,fill_value",
    CUDA_PATH_CASES,
)
def test_nonzero_static_cuda_paths(shape, dtype, nnz_ratio, size, fill_value):
    torch.manual_seed(3)
    input = make_input(shape, dtype, nnz_ratio)
    assert_nonzero_static_matches(input, size=size, fill_value=fill_value)


@ASCEND_ONLY
@FULL_ONLY
@pytest.mark.nonzero_static
@pytest.mark.parametrize(
    "shape,dtype,nnz_ratio,size,fill_value",
    ASCEND_PATH_CASES,
)
def test_nonzero_static_ascend_paths(shape, dtype, nnz_ratio, size, fill_value):
    torch.manual_seed(4)
    input = make_input(shape, dtype, nnz_ratio)
    assert_nonzero_static_matches(input, size=size, fill_value=fill_value)


@ASCEND_ONLY
@FULL_ONLY
@pytest.mark.nonzero_static
def test_nonzero_static_ascend_sparse_group_fallback():
    input = torch.zeros(8193, dtype=torch.bfloat16, device=flag_gems.device)
    input[[1, 2, 3, 40, 41, 2050]] = 1
    assert_nonzero_static_matches(input, size=4, fill_value=-1)


@ASCEND_ONLY
@FULL_ONLY
@pytest.mark.nonzero_static
def test_nonzero_static_ascend_bfloat16_special_values():
    input = torch.zeros(8193, dtype=torch.bfloat16, device=flag_gems.device)
    input[:7] = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0],
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )
    assert_nonzero_static_matches(input, size=128, fill_value=-1)
