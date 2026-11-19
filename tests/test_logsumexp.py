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

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    gems_log_logger,
    gems_log_prefix,
    to_reference,
)
from .conftest import QUICK_MODE

DIM_LIST = [0] if QUICK_MODE else [0, 1]


@pytest.mark.logsumexp
@pytest.mark.skipif(
    flag_gems.logsumexp.__module__ != "flag_gems.ops.logsumexp",
    reason="covers the generic logsumexp implementation",
)
@pytest.mark.parametrize(
    "shape, dim, dtype",
    [
        ((0,), 0, torch.float32),
        ((2, 0, 3), -2, torch.float32),
        ((0, 2), 1, torch.float32),
        ((2, 3, 0), 1, torch.float32),
        ((0,), 0, torch.int64),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_empty(shape, dim, dtype, keepdim, caplog):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.logsumexp(to_reference(inp), dim=dim, keepdim=keepdim)
    with caplog.at_level("DEBUG", logger=gems_log_logger(flag_gems.logsumexp)):
        res_out = flag_gems.logsumexp(inp, dim=dim, keepdim=keepdim)
    assert f"{gems_log_prefix(flag_gems.logsumexp)} LOGSUMEXP" in caplog.text
    torch.testing.assert_close(res_out, ref_out.to(res_out.device), rtol=0, atol=0)


@pytest.mark.logsumexp
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
def test_accuracy_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.logsumexp(inp, dim=dim, keepdim=keepdim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim], atol=5e-3)
