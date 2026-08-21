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

# norm_except_dim keeps one dim and reduces the rest, so rank >= 1 is required
# (a 0-dim input has no "other" dims). Shapes exercise the dim==0 contiguous
# path, the dim==last column-reduction path, and a middle-dim path.
NORM_EXCEPT_DIM_SHAPES = [
    (4, 256, 3),
    (4096, 256),
    (64, 64),
    (32, 1024),
    (1, 2),
]


@pytest.mark.norm_except_dim
@pytest.mark.parametrize("shape", NORM_EXCEPT_DIM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_norm_except_dim(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.norm_except_dim(ref_inp, 2, 0)
    with flag_gems.use_gems():
        res_out = torch.norm_except_dim(res_inp, 2, 0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.norm_except_dim
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_norm_except_dim_dim(dim, dtype):
    # A rank-3 shape so dim in {0, 1, 2, -1} each keep a distinct axis.
    shape = (8, 33, 17)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.norm_except_dim(ref_inp, 2, dim)
    with flag_gems.use_gems():
        res_out = torch.norm_except_dim(res_inp, 2, dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.norm_except_dim
@pytest.mark.parametrize("pow", [1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_norm_except_dim_pow(pow, dtype):
    # A rank-2 shape keeping dim 0; vary the norm order p.
    shape = (128, 512)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.norm_except_dim(ref_inp, pow, 0)
    with flag_gems.use_gems():
        res_out = torch.norm_except_dim(res_inp, pow, 0)

    utils.gems_assert_close(res_out, ref_out, dtype)
