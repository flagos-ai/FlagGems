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

device = flag_gems.device


@pytest.mark.eye
@pytest.mark.parametrize(
    "shape",
    [
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
    ]
    + [(2**d, 2**d) for d in range(7, 13)],
)
@pytest.mark.parametrize(
    "dtype", utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES + utils.BOOL_TYPES
)
def test_eye(shape, dtype):
    n, m = shape

    # test eye(n, m) without dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, m, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, m, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )

    # test eye(n)
    with flag_gems.use_gems():
        res_out = torch.eye(n, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )


@pytest.mark.eye_m
@pytest.mark.parametrize(
    "shape",
    [
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
    ]
    + [(2**d, 2**d) for d in range(7, 13)],
)
@pytest.mark.parametrize(
    "dtype", utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES + utils.BOOL_TYPES
)
def test_eye_m(shape, dtype):
    n, m = shape

    # test eye(n, m) without dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, m, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, m, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )
