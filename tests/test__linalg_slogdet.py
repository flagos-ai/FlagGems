# Copyright 2026, The FlagOS Contributors.
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

# Define shapes for _linalg_slogdet (square matrices)
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.underscore_linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
# _linalg_slogdet generated kernel only supports float32 on CUDA.
@pytest.mark.parametrize("dtype", [torch.float32])
def test__linalg_slogdet(shape, dtype):
    """Test aten::_linalg_slogdet accuracy against PyTorch reference.

    aten::_linalg_slogdet returns (sign, logabsdet, LU, pivots). The LU/pivots
    outputs follow the LAPACK factorization and are not bit-reproducible by the
    Gaussian-elimination kernel, so only the well-defined (sign, logabsdet) pair
    is compared -- matching torch.linalg.slogdet semantics.
    """
    # Ensure we have a square matrix
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    # Create a well-conditioned input tensor
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    ref_A = utils.to_reference(A)

    # Compute reference via the public 2-output slogdet.
    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_A)

    # Compute with FlagGems via the aten 4-output overload (directly, no use_gems).
    res_sign, res_logabsdet, _res_lu, _res_pivots = torch.ops.aten._linalg_slogdet(A)

    # Compare sign
    utils.gems_assert_close(res_sign, ref_sign, dtype)

    # Compare logabsdet (more tolerant for floating point)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype, reduce_dim=shape[-1])
