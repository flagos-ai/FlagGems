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
import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_zeros_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.zeros_like(ref_inp)

    res_out = flag_gems.zeros_like(inp)

    utils.gems_assert_equal(res_out, ref_out)


# Ascend coreDim and capped-grid regressions for zeros_like.
BLOCK_SIZE = 32768
CORE_DIM_LIMIT = 65535


@pytest.fixture
def zeros_like_ops():
    ops = importlib.import_module(flag_gems.zeros_like.__module__)
    assert "_ascend" in ops.__name__
    return ops


@pytest.fixture
def checked_zeros_like_launches(zeros_like_ops, monkeypatch):
    kernel = zeros_like_ops.zeros_kernel
    launches = []

    class CheckedKernel:
        def __getitem__(self, grid):
            def checked_grid(meta):
                resolved = grid(meta) if callable(grid) else grid
                assert 0 < math.prod(resolved) <= CORE_DIM_LIMIT
                launches.append((resolved, meta["BLOCK_SIZE"], meta["BLOCK_SIZE_SUB"]))
                return resolved

            return kernel[checked_grid]

    monkeypatch.setattr(zeros_like_ops, "zeros_kernel", CheckedKernel())
    return launches


@pytest.mark.zeros_like
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("logical_blocks", [65534, 65535, 65536, 131072])
def test_zeros_like_launch_core_dim_limit(zeros_like_ops, monkeypatch, logical_blocks):
    # Check launch planning without allocating a multi-GiB tensor.
    numel = logical_blocks * BLOCK_SIZE

    class MetadataOnlyTensor:
        device = torch.device(flag_gems.device)

        def numel(self):
            return numel

    tensor = MetadataOnlyTensor()
    launches = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def capture(out, size, **meta):
                assert out is tensor and size == numel
                launches.append(grid(meta) if callable(grid) else grid)

            return capture

    monkeypatch.setattr(zeros_like_ops, "zeros_kernel", CaptureKernel())
    monkeypatch.setattr(zeros_like_ops.torch, "empty_like", lambda *a, **kw: tensor)
    result = zeros_like_ops.zeros_like(tensor, dtype=torch.bool)
    assert result is tensor
    assert launches
    for grid in launches:
        assert (
            0 < math.prod(grid) <= CORE_DIM_LIMIT
        ), f"zeros_like: numel={numel}, grid={grid}, coreDim must be <= {CORE_DIM_LIMIT}"


@pytest.mark.zeros_like
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
def test_zeros_like_factory_capped_grid_exact(
    zeros_like_ops, checked_zeros_like_launches, monkeypatch
):
    # _zeros_grid resolves its cap from the zeros module shared by zeros_like.
    zeros_ops = importlib.import_module(flag_gems.zeros.__module__)
    monkeypatch.setattr(zeros_ops, "_MAX_CORE_DIM", 2)
    shape = (9 * BLOCK_SIZE + 17,)
    original_empty_like = torch.empty_like

    def poisoned_empty_like(*args, **kwargs):
        out = original_empty_like(*args, **kwargs)
        out.fill_(7)
        return out

    inp = torch.empty(shape, device=flag_gems.device, dtype=torch.float32)
    with monkeypatch.context() as patch:
        patch.setattr(zeros_like_ops.torch, "empty_like", poisoned_empty_like)
        result = flag_gems.zeros_like(inp, device=inp.device, dtype=inp.dtype)
    torch.npu.synchronize()
    assert result.device == inp.device
    torch.testing.assert_close(
        result.cpu(), torch.zeros(shape, dtype=inp.dtype), rtol=0, atol=0
    )
    assert checked_zeros_like_launches
    assert all(grid == (2,) for grid, _, _ in checked_zeros_like_launches)
