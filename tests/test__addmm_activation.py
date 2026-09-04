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
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.addmm_activation
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("use_gelu", [False, True])
def test_addmm_activation(M, N, K, scalar, dtype, use_gelu):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch._addmm_activation(
        ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta, use_gelu=use_gelu
    )
    with flag_gems.use_gems():
        res_out = torch._addmm_activation(
            bias, mat1, mat2, alpha=alpha, beta=beta, use_gelu=use_gelu
        )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm_activation_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("use_gelu", [False, True])
def test_addmm_activation_out(M, N, K, scalar, dtype, use_gelu):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    alpha = beta = scalar

    torch.ops.aten._addmm_activation.out(
        ref_bias,
        ref_mat1,
        ref_mat2,
        alpha=alpha,
        beta=beta,
        use_gelu=use_gelu,
        out=ref_out,
    )
    with flag_gems.use_gems():
        torch.ops.aten._addmm_activation.out(
            bias, mat1, mat2, alpha=alpha, beta=beta, use_gelu=use_gelu, out=out
        )

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)
