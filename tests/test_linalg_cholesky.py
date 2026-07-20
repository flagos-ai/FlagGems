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


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky(shape, dtype):
    # Create a positive-definite matrix: A = B @ B^T + I
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A)

    # For gems, use aten.linalg_cholesky with flag_gems
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_cholesky(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_upper(shape, dtype):
    # Test with upper=True
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A, upper=True)

    # For gems, use aten.linalg_cholesky with flag_gems
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_cholesky(A, upper=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_batch(shape, dtype):
    # Create positive-definite matrices for batched input: A = B @ B^T + I
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = (
        B @ B.transpose(-2, -1)
        + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1
    )

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A)

    # For gems, use aten.linalg_cholesky with flag_gems
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_cholesky(A)

    utils.gems_assert_close(res_out, ref_out, dtype)
