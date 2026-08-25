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

DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES

CHAIN_CASES = [
    ((64, 32, 128), False, False),
    ((64, 32, 128, 16), False, False),
    ((64, 128, 32, 16), False, False),
    ((1, 64, 32, 128, 16), True, False),
    ((64, 32, 128, 64, 32, 1), False, True),
    ((1, 64, 1), True, True),
]


def _make_chain(dimensions, first_vector, last_vector, dtype):
    tensors = [
        torch.randn(
            dimensions[index],
            dimensions[index + 1],
            dtype=dtype,
            device=flag_gems.device,
        )
        for index in range(len(dimensions) - 1)
    ]
    if first_vector:
        tensors[0] = tensors[0].squeeze(0)
    if last_vector:
        tensors[-1] = tensors[-1].squeeze(1)
    return tensors


def _reference_multi_dot(tensors):
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.linalg.multi_dot(tensors)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@pytest.mark.linalg_multi_dot
@pytest.mark.parametrize("dimensions,first_vector,last_vector", CHAIN_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_linalg_multi_dot(dimensions, first_vector, last_vector, dtype):
    tensors = _make_chain(dimensions, first_vector, last_vector, dtype)
    reference_tensors = [utils.to_reference(tensor) for tensor in tensors]

    reference = _reference_multi_dot(reference_tensors)
    with flag_gems.use_gems():
        result = torch.linalg.multi_dot(tensors)

    utils.gems_assert_close(result, reference, dtype, reduce_dim=max(dimensions))


@pytest.mark.linalg_multi_dot_out
@pytest.mark.parametrize("dimensions,first_vector,last_vector", CHAIN_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_linalg_multi_dot_out(dimensions, first_vector, last_vector, dtype):
    tensors = _make_chain(dimensions, first_vector, last_vector, dtype)
    reference_tensors = [utils.to_reference(tensor) for tensor in tensors]
    reference = _reference_multi_dot(reference_tensors)

    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.linalg.multi_dot(tensors, out=out)

    assert result is out
    assert tuple(out.shape) == tuple(reference.shape)
    utils.gems_assert_close(out, reference, dtype, reduce_dim=max(dimensions))


@pytest.mark.linalg_multi_dot_out
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_linalg_multi_dot_out_noncontiguous(dtype):
    dimensions = (64, 32, 128)
    tensors = _make_chain(dimensions, False, False, dtype)
    reference_tensors = [utils.to_reference(tensor) for tensor in tensors]
    reference = _reference_multi_dot(reference_tensors)
    out = torch.empty(
        dimensions[-1], dimensions[0], dtype=dtype, device=flag_gems.device
    ).t()

    with flag_gems.use_gems():
        result = torch.linalg.multi_dot(tensors, out=out)

    assert result is out
    assert not out.is_contiguous()
    utils.gems_assert_close(out, reference, dtype, reduce_dim=max(dimensions))


@pytest.mark.linalg_multi_dot
def test_error_linalg_multi_dot_requires_two_tensors():
    tensor = torch.randn(4, 4, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="at least 2"):
        torch.linalg.multi_dot([tensor])
