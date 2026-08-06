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


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)])
# _linalg_eigh requires float32 for cuSOLVER eigenvalue computation
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("UPLO", ["L", "U"])
def test_linalg_eigh(shape, dtype, UPLO):
    """Test _linalg_eigh accuracy against PyTorch reference."""
    # Create a symmetric matrix
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = (base + base.transpose(-2, -1)) / 2
    ref_inp = utils.to_reference(inp)

    ref_w, ref_v = torch.ops.aten._linalg_eigh.default(ref_inp, UPLO, True)
    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, UPLO, True)

    # Compare eigenvalues
    utils.gems_assert_close(res_w, ref_w, dtype)

    # Eigenvectors are only defined up to sign/basis rotation
    # Validate via reconstruction: A = V diag(w) V^T
    # Tolerance is 1e-2 because the reconstruction matmuls run under TF32 on
    # newer NVIDIA GPUs (CI default), which accumulates ~1e-3 error on 20x20.
    A = inp.float()
    V = res_v.float()
    W = res_w.float()
    recon = V @ torch.diag_embed(W) @ V.transpose(-2, -1)
    assert torch.allclose(recon, A, atol=1e-2, rtol=1e-2), "Reconstruction failed"

    # Check V is orthonormal: V^T V = I
    eye = torch.eye(shape[-1], dtype=V.dtype, device=V.device)
    VtV = V.transpose(-2, -1) @ V
    assert torch.allclose(
        VtV, eye, atol=1e-2, rtol=1e-2
    ), "Eigenvectors not orthonormal"
