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

import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [(1, 1, 32), (16, 16, 64)]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
        (15, 160, 1024),
        (128, 256, 512),
        (256, 256, 256),
        (495, 5333, 71),
        (1024, 1024, 128),
    ]


@pytest.mark.addbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm(M, N, K, scalar, dtype):
    torch.manual_seed(0)
    if flag_gems.vendor_name == "mthreads" and dtype in (torch.float16, torch.bfloat16):
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    try:
        batch1 = torch.randn((4, M, K), dtype=dtype, device=flag_gems.device)
        batch2 = torch.randn((4, K, N), dtype=dtype, device=flag_gems.device)
        self = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
        original = self.clone()
        ref_result = torch.addbmm(
            utils.to_reference(self, True),
            utils.to_reference(batch1, True),
            utils.to_reference(batch2, True),
            alpha=scalar,
            beta=scalar,
        )
        with flag_gems.use_gems():
            result = torch.addbmm(self, batch1, batch2, alpha=scalar, beta=scalar)
        utils.gems_assert_close(result, ref_result, dtype, reduce_dim=K)
        assert self.device == original.device
    assert self.dtype == original.dtype
    assert self.shape == original.shape
    utils.gems_assert_equal(self.cpu(), original.cpu())
    finally:
        if flag_gems.vendor_name == "mthreads":
            os.environ.pop("MUSA_ENABLE_SQMMA", None)


@pytest.mark.addbmm
def test_addbmm_autograd():
    torch.manual_seed(0)
    dtype = torch.float32
    self = torch.randn((3, 5), dtype=dtype, device=flag_gems.device, requires_grad=True)
    batch1 = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device, requires_grad=True)
    batch2 = torch.randn((2, 4, 5), dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inputs = tuple(utils.to_reference(value, True).requires_grad_(True) for value in (self, batch1, batch2))
    ref_result = torch.addbmm(*ref_inputs)
    with flag_gems.use_gems():
        result = torch.addbmm(self, batch1, batch2)
    grad = torch.randn_like(result)
    ref_grads = torch.autograd.grad(ref_result, ref_inputs, utils.to_reference(grad, True))
    grads = torch.autograd.grad(result, (self, batch1, batch2), grad)
    for actual, expected in zip(grads, ref_grads):
        utils.gems_assert_close(actual, expected, dtype, reduce_dim=4)


@pytest.mark.addbmm
def test_addbmm_rejects_incompatible_shapes():
    torch.manual_seed(0)
    self = torch.randn((3, 5), device=flag_gems.device)
    batch1 = torch.randn((2, 3, 4), device=flag_gems.device)
    batch2 = torch.randn((2, 6, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.addbmm(self, batch1, batch2)
