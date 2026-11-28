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


@pytest.mark.randn_like
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randn_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.randn_like(x)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.randn_like
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_randn_like_complex(dtype):
    x = torch.empty((128, 1024), device=flag_gems.device).T
    actual = flag_gems.randn_like(x, dtype=dtype)
    assert actual.dtype == dtype
    assert actual.shape == x.shape
    assert actual.stride() == x.stride()
    parts = torch.view_as_real(actual.cpu()).flatten(0, -2)
    assert torch.all(parts.mean(dim=0).abs() < 0.015)
    assert torch.all((parts.var(dim=0) - 0.5).abs() < 0.02)

    empty = flag_gems.randn_like(x[:0], dtype=dtype)
    assert empty.dtype == dtype
    assert empty.shape == x[:0].shape
