# Copyright 2026 FlagOS Contributors
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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KEEPDIM_DIMS = [(True, 1)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KEEPDIM_DIMS = list(zip([True, False] * 2, DIMS_LIST))


@pytest.mark.mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mean
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.mean
def test_mean_empty_explicit_dtype():
    inp = torch.empty((0,), dtype=torch.float16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dtype=torch.float32)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dtype=torch.float32)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.mean
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_mean_empty_rejects_non_floating_input(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.mean(inp)


@pytest.mark.mean
def test_mean_empty_rejects_non_floating_dtype():
    inp = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.mean(inp, dtype=torch.int32)


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_EMPTY_DIM_CASES = [
    ((2, 0), 1),
    ((0, 3), 0),
    ((0, 3), 1),
    ((2, 0), -1),
    ((2, 0, 3), [0, 1]),
    ((2, 0, 3), [1, 2]),
    ((0, 2, 3), [1, 2]),
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape, dim", MEAN_EMPTY_DIM_CASES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_empty(shape, dim, keepdim, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.mean_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_empty_explicit_dtype():
    inp = torch.empty((2, 0), dtype=torch.float16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim=1, dtype=torch.float32)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim=1, dtype=torch.float32)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


# Shapes where K (product of dims after the reduction axis) exceeds the CUDA
# grid-Y limit of 65535, which used to trigger "Triton Error [CUDA]: invalid
# argument" before the mean_heur_tile_k grid-overflow fix.
MEAN_LARGE_K_SHAPES = [
    (1, 8, 256, 256),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1, 4, 512, 512),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_K_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_large_k(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_LARGE_INNERDIM_SHAPES = [
    (1024, 1024, 1024),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1024, 2048, 1024),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_INNERDIM_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_large_innerdim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)
