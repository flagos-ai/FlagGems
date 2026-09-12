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

# Square-matrix shapes for slogdet: a batched small case covers the (*, n, n)
# interface, and 4x4 through 32x32 cover the small/medium matrices targeted
# by this single-program LU implementation.
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
# torch.slogdet (and the FlagGems kernel) only support float32 on CUDA.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_slogdet(shape, dtype):
    """Test slogdet accuracy against PyTorch reference."""
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.slogdet(ref_A)

    res_sign, res_logabsdet = flag_gems.slogdet(A)

    # Compare sign
    utils.gems_assert_close(res_sign, ref_out.sign, dtype)

    # Compare logabsdet (more tolerant for floating point)
    utils.gems_assert_close(
        res_logabsdet, ref_out.logabsdet, dtype, reduce_dim=shape[-1]
    )


@pytest.mark.slogdet
@pytest.mark.parametrize("n", [3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_slogdet_singular(n, dtype):
    """A singular matrix should yield sign == 0 and logabsdet == -inf."""
    A = torch.zeros((n, n), dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.slogdet(ref_A)
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    assert torch.equal(res_sign, torch.zeros((), dtype=dtype, device=A.device))
    assert torch.all(torch.isinf(res_logabsdet))
    # Reference: sign == 0, logabsdet == -inf
    assert torch.equal(ref_out.sign, torch.zeros((), dtype=dtype, device=ref_A.device))
    assert torch.all(torch.isinf(ref_out.logabsdet))


@pytest.mark.slogdet
@pytest.mark.parametrize("n", [3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_slogdet_identity(n, dtype):
    """The identity matrix has sign == 1 and logabsdet == 0."""
    A = torch.eye(n, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)

    ref_out = torch.slogdet(ref_A)
    res_sign, res_logabsdet = flag_gems.slogdet(A)

    utils.gems_assert_close(res_sign, ref_out.sign, dtype)
    utils.gems_assert_close(res_logabsdet, ref_out.logabsdet, dtype, reduce_dim=n)
