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

# Define shapes for linalg_det (square matrices)
DET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", DET_SHAPES)
# _linalg_det generated kernel only supports float32 on CUDA.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_det(shape, dtype):
    """Test aten::_linalg_det accuracy against PyTorch reference.

    aten::_linalg_det returns (result, LU, pivots). The LU/pivots outputs follow
    the LAPACK factorization and are not bit-reproducible by the
    Gaussian-elimination kernel, so only the well-defined det result is
    compared -- matching torch.linalg.det semantics.
    """
    # Ensure we have a square matrix
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    # Create a well-conditioned input tensor: identity plus a small perturbation.
    # This keeps the det value O(1) so it never overflows float32 (a plain
    # ``randn + eye*n`` construction pushes det ~ n**n past 3.4e38 by n=32).
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    if len(shape) > 2:
        eye = eye.expand(shape).contiguous()
    A = eye + torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.05

    ref_A = utils.to_reference(A)

    # Compute reference via the public det.
    ref_result = torch.linalg.det(ref_A)

    # Compute with FlagGems via the aten 3-output overload.
    with flag_gems.use_gems():
        res_result, _res_lu, _res_pivots = torch.ops.aten._linalg_det(A)

    # Compare det (tolerant for floating-point accumulation over n pivots).
    utils.gems_assert_close(res_result, ref_result, dtype, reduce_dim=shape[-1])
