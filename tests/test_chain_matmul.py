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

# Each case lists the chain dimensions (d0, d1, ..., dn), so the chain holds
# n >= 2 matrices of shapes (d0, d1), (d1, d2), ... and the result is (d0, dn).
# The optimal parenthesization differs between these cases, which is what
# exercises the matrix-chain DP rather than just the gemms. The one-matrix
# chain is covered separately because multi_dot needs at least two tensors.
CHAIN_CASES = [
    (16, 16, 16),
    (64, 32, 128),
    (64, 32, 128, 16),
    (64, 128, 32, 16),
    (128, 16, 256, 32, 64),
    (64, 32, 128, 64, 32, 16),
]


def _make_chain(dimensions, dtype):
    return [
        torch.randn(
            dimensions[index],
            dimensions[index + 1],
            dtype=dtype,
            device=flag_gems.device,
        )
        for index in range(len(dimensions) - 1)
    ]


def _reference_chain_matmul(tensors):
    # TF32 would make the reference itself lossy, so pin it off and compare
    # against the full-precision result.
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.linalg.multi_dot(tensors)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dimensions", CHAIN_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul(dimensions, dtype):
    tensors = _make_chain(dimensions, dtype)
    ref_tensors = [utils.to_reference(tensor) for tensor in tensors]

    ref_out = _reference_chain_matmul(ref_tensors)
    res_out = flag_gems.chain_matmul(tensors)

    # The error accumulates over the reduction dimensions of the whole chain,
    # so scale the tolerance with the largest one.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(dimensions))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_single_matrix(dtype):
    # A one-matrix chain is returned unchanged.
    inp = torch.randn(32, 16, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul([inp])

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_non_contiguous(dtype):
    # Transposed views: the kernels are stride-generic and must not assume a
    # row-major layout.
    tensors = [
        torch.randn(32, 64, dtype=dtype, device=flag_gems.device).t(),
        torch.randn(16, 32, dtype=dtype, device=flag_gems.device).t(),
    ]
    assert not any(tensor.is_contiguous() for tensor in tensors)
    ref_tensors = [utils.to_reference(tensor) for tensor in tensors]

    ref_out = _reference_chain_matmul(ref_tensors)
    res_out = flag_gems.chain_matmul(tensors)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_empty_list():
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.chain_matmul([])
