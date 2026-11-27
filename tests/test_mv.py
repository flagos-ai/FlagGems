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
from .gemm_utils import accuracy_dtype_cases


@pytest.mark.mv
@pytest.mark.parametrize(
    "dtype, N, K, column_major", accuracy_dtype_cases("mv", cfg.QUICK_MODE)
)
def test_mv(N, K, dtype, column_major):
    matrix = (
        torch.randn((K, N), dtype=dtype, device=flag_gems.device).t()
        if column_major
        else torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    )
    vector = torch.randn((K,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    res_out = flag_gems.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


# Odd sizes exercise partial row/column tiles; both layouts cover explicit
# non-contiguous matrix strides without making the matrix contiguous first.
@pytest.mark.mv_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("N,K", [(7, 11), (33, 65)])
@pytest.mark.parametrize("column_major", [False, True])
def test_mv_backward(N, K, dtype, column_major):
    matrix = (
        torch.randn((K, N), dtype=dtype, device=flag_gems.device).t()
        if column_major
        else torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    )
    vector_storage = torch.randn((K * 2,), dtype=dtype, device=flag_gems.device)
    vector = vector_storage[::2]
    grad = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_matrix = utils.to_reference(matrix, True).requires_grad_()
    ref_vector = utils.to_reference(vector, True).requires_grad_()
    ref_grad = utils.to_reference(grad, True)
    ref_out = torch.mv(ref_matrix, ref_vector)
    ref_matrix_grad, ref_vector_grad = torch.autograd.grad(
        ref_out, (ref_matrix, ref_vector), ref_grad
    )

    matrix.requires_grad_()
    vector.requires_grad_()
    out = flag_gems.mv(matrix, vector)
    matrix_grad, vector_grad = torch.autograd.grad(out, (matrix, vector), grad)

    utils.gems_assert_close(matrix_grad, ref_matrix_grad, dtype)
    utils.gems_assert_close(vector_grad, ref_vector_grad, dtype, reduce_dim=N)


@pytest.mark.mv_backward
@pytest.mark.parametrize("input_name", ["matrix", "vector"])
def test_mv_backward_selective_gradient(input_name):
    dtype = torch.float32
    matrix = torch.randn((17, 31), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((31,), dtype=dtype, device=flag_gems.device)
    selected = matrix if input_name == "matrix" else vector
    selected.requires_grad_()
    out = flag_gems.mv(matrix, vector)
    (gradient,) = torch.autograd.grad(out, (selected,), torch.randn_like(out))
    assert gradient.shape == selected.shape


@pytest.mark.mv_backward
def test_mv_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    matrix = torch.randn(
        (64, 257), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    vector = torch.randn(
        (257,), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((64,), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.mv(matrix, vector)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (matrix, vector), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "mv_backward_kernel" in item[0]]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
