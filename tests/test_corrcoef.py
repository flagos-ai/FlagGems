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

# corrcoef treats rows as variables and columns as observations, so the
# input must be a 2D matrix. These shapes cover a single-variable degenerate
# case, square matrices, and tall/wide matrices.
CORRCOEF_SHAPES = [
    (1, 8),
    (2, 4),
    (4, 16),
    (8, 64),
    (16, 256),
    (32, 1024),
    (64, 128),
    (128, 512),
]


@pytest.mark.corrcoef
@pytest.mark.parametrize("shape", CORRCOEF_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    res_out = flag_gems.corrcoef(inp)

    ref_out = torch.corrcoef(ref_inp).to(dtype)

    # The correlation matrix is NxN; use a larger reduce_dim tolerance since
    # the matrix product accumulates over n_cols observations.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[1])


@pytest.mark.corrcoef
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef_1d(dtype):
    # A 1D input represents a single variable; the correlation coefficient of a
    # variable with itself is 1.0.
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.corrcoef(inp)

    ref_out = torch.corrcoef(utils.to_reference(inp, upcast=True)).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.corrcoef
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef_non_contiguous(dtype):
    # Verify corrcoef handles non-contiguous inputs by working on a strided
    # slice of a larger matrix.
    base = torch.randn(64, 256, dtype=dtype, device=flag_gems.device)
    inp = base[::2, ::2]
    ref_inp = utils.to_reference(inp, upcast=True)

    res_out = flag_gems.corrcoef(inp)

    ref_out = torch.corrcoef(ref_inp).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.shape[1])
