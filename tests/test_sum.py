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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


@pytest.mark.sum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    res_out = flag_gems.sum(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_large_ascend_reduction_tile(use_out):
    if flag_gems.device != "npu":
        pytest.skip("Ascend local-buffer reduction limit")

    # The old sqrt-based heuristic selected 32768, overflowing Ascend's UB.
    # At BLOCK_SIZE=4096 this also exceeds coreDim and needs multiple mid tiles.
    inp = torch.randn(2**28 + 97, dtype=torch.float32, device=flag_gems.device)
    if use_out:
        result = torch.empty((), dtype=inp.dtype, device=inp.device)
        flag_gems.sum_out(inp, out=result)
    else:
        result = flag_gems.sum(inp)
    expected = torch.sum(inp)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-2)


@pytest.mark.sum
def test_sum_ascend_grid_stride(monkeypatch):
    if flag_gems.device != "npu":
        pytest.skip("Ascend launch-grid limit")

    sum_ops = importlib.import_module(flag_gems.sum.__module__)
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 1)
    monkeypatch.setattr(sum_ops, "_ASCEND_SUM_MAX_BLOCK_SIZE", 128)
    inp = torch.randn(65537, dtype=torch.float32, device=flag_gems.device)
    expected = torch.sum(inp)
    result = flag_gems.sum(inp)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-3)


INCLUDE_0_SHAPES = [(1, 0, 128, 512), (4096, 1, 256, 0), (200, 10, 0, 3)]


@pytest.mark.sum_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + INCLUDE_0_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    # Direct GEMS call: the dispatcher lifts an int dim to IntList, so pass a
    # list here to match sum_dim's own contract.
    dims = [dim] if isinstance(dim, int) else list(dim)
    res_out = flag_gems.sum_dim(inp, dim=dims, keepdim=keepdim)

    dim_list = [d % inp.ndim for d in dims]
    _dim = 1
    for d in dim_list:
        _dim *= shape[d]
    if dim_list == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.sum_dim_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim_out(shape, dim, keepdim, dtype):
    # Regression test: sum_dim_out must resize external out tensor and skip squeeze.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_result = torch.sum(ref_inp, dim=dim, keepdim=keepdim)

    # Pre-allocate out tensor with wrong shape to test resize logic
    out = torch.empty((1,), dtype=dtype, device=flag_gems.device)
    # Direct GEMS call: pass dim as a list (the dispatcher would lift int -> IntList).
    dims = [dim] if isinstance(dim, int) else list(dim)
    res_result = flag_gems.sum_dim_out(inp, dim=dims, keepdim=keepdim, out=out)

    dim_list = [d % inp.ndim for d in dims]
    _dim = 1
    for d in dim_list:
        _dim *= shape[d]
    if dim_list == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_result, ref_result, dtype, reduce_dim=_dim)
    utils.gems_assert_close(out, ref_result, dtype, reduce_dim=_dim)


@pytest.mark.sum_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)

    out = torch.empty([], dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.sum_out(inp, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=inp.numel())
