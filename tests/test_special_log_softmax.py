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


@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log_softmax(dtype):
    # Test with dim=1, which is the most common case
    x = torch.randn(32, 64, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.log_softmax(ref_x, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=1)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log_softmax_large_n(dtype):
    # Test large N cases
    x = torch.randn(1, 8192, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.log_softmax(ref_x, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=1)
    utils.gems_assert_close(res_out, ref_out, dtype)


# Reducing over a dim that is not the innermost one leaves K > 1 elements
# between neighbouring values of the reduced axis. Both the small-N and the
# large-N path must stride by K instead of assuming the row is contiguous.
@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((32, 64), 0),  # small N, K = 64
        ((4, 8, 16), 0),  # small N, K = 128
        ((4, 8, 16), 1),  # small N, K = 16
        ((17, 33, 5), 1),  # small N, non-power-of-2 K
        ((2, 3, 4, 5), 2),  # small N, 4-D
        ((2048, 4), 0),  # large N, K = 4
        ((4, 2048, 3), 1),  # large N, K = 3
    ],
)
def test_special_log_softmax_non_inner_dim(shape, dim, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.log_softmax(ref_x, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=dim)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_special_log_softmax_negative_dim(dim, dtype):
    x = torch.randn(4, 8, 16, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.log_softmax(ref_x, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=dim)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=x.shape[dim])
