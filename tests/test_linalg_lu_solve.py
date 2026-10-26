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

SHAPES = [(4, 4), (16, 1), (33, 16), (64, 64), (100, 40)]
DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])


def _cpu_lu_solve(lu, pivots, rhs, *, left=True, adjoint=False):
    """CPU ATen first, with a NumPy solve when this Torch has no LAPACK."""
    assert lu.device.type == pivots.device.type == rhs.device.type == "cpu"
    try:
        return torch.linalg.lu_solve(lu, pivots, rhs, left=left, adjoint=adjoint)
    except RuntimeError as exc:
        if "requires compiling PyTorch with LAPACK" not in str(exc):
            raise
    import numpy as np

    lower = torch.tril(lu, diagonal=-1) + torch.eye(lu.shape[-1], dtype=lu.dtype)
    matrix = lower @ torch.triu(lu)
    # LAPACK pivots encode sequential row swaps: undo them to recover A.
    flat = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
    pivot_rows = pivots.reshape(-1, pivots.shape[-1])
    for batch, swaps in zip(flat, pivot_rows):
        for i in range(swaps.numel() - 1, -1, -1):
            j = int(swaps[i]) - 1
            batch[[i, j]] = batch[[j, i]]
    array = matrix.numpy()
    if adjoint:
        array = array.swapaxes(-1, -2).conj()
    right = rhs.numpy()
    if left:
        output = np.linalg.solve(array, right)
    else:
        output = np.linalg.solve(array.swapaxes(-1, -2), right.swapaxes(-1, -2)).swapaxes(-1, -2)
    return torch.from_numpy(np.array(output, copy=True)).to(dtype=rhs.dtype)


def _make_inputs(batch_shape, n, k, dtype):
    # PPU device-side lu_factor aborts; factor the same recipe on CPU instead.
    factory_device = "cpu" if flag_gems.vendor_name == "thead" else flag_gems.device
    matrix = torch.randn(*batch_shape, n, n, dtype=dtype, device=factory_device)
    matrix = matrix @ matrix.mT + torch.eye(n, dtype=dtype, device=factory_device) * n
    rhs = torch.randn(*batch_shape, n, k, dtype=dtype, device=factory_device)
    lu, pivots = torch.linalg.lu_factor(matrix)
    return tuple(value.to(flag_gems.device) for value in (lu, pivots, rhs))


@pytest.mark.linalg_lu_solve
@pytest.mark.parametrize("batch_shape", [(), (4,)])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg_lu_solve(batch_shape, shape, dtype):
    n, k = shape
    lu, pivots, rhs = _make_inputs(batch_shape, n, k, dtype)
    ref_lu = utils.to_reference(lu, True)
    ref_pivots = utils.to_reference(pivots)
    ref_rhs = utils.to_reference(rhs, True)
    reference = (
        _cpu_lu_solve(ref_lu, ref_pivots, ref_rhs)
        if ref_lu.device.type == "cpu"
        else torch.linalg.lu_solve(ref_lu, ref_pivots, ref_rhs)
    )
    result = flag_gems.linalg_lu_solve(lu, pivots, rhs)
    assert result.device == rhs.device
    assert result.dtype == rhs.dtype
    assert result.shape == rhs.shape
    assert result is not lu
    assert result is not rhs
    if result.numel():
        assert result.data_ptr() != lu.data_ptr()
        assert result.data_ptr() != rhs.data_ptr()
    utils.gems_assert_close(result, reference, dtype)
