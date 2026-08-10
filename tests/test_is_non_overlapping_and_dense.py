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

import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPES = [
    (64,),
    (64, 128),
    (8, 16, 32),
    (4, 8, 16, 32),
]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense(shape, dtype):
    """Contiguous tensors of every rank are non-overlapping and dense."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.is_non_overlapping_and_dense(inp)

    assert res_out == ref_out, f"Expected {ref_out}, got {res_out}"
    assert res_out is True


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_permuted(shape, dtype):
    """Permuting dimensions reorders strides but keeps the storage span dense."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    for perm in itertools.permutations(range(len(shape))):
        permuted = inp.permute(perm)
        ref_permuted = utils.to_reference(inp).permute(perm)

        ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_permuted)

        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_non_overlapping_and_dense(permuted)

        assert res_out == ref_out, f"perm={perm}: expected {ref_out}, got {res_out}"
        assert res_out is True


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("shape", [(64,), (64, 128), (8, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_strided(shape, dtype):
    """A step-2 slice leaves gaps in storage, so it is not dense."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    strided = inp[..., ::2]
    ref_strided = utils.to_reference(inp)[..., ::2]

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_strided)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.is_non_overlapping_and_dense(strided)

    assert res_out == ref_out, f"Expected {ref_out}, got {res_out}"
    assert res_out is False


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_narrowed(dtype):
    """Narrowing an inner dimension leaves a gap per outer step."""
    inp = torch.randn(8, 8, dtype=dtype, device=flag_gems.device)
    narrowed = inp[:, :4]
    ref_narrowed = utils.to_reference(inp)[:, :4]

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_narrowed)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.is_non_overlapping_and_dense(narrowed)

    assert res_out == ref_out
    assert res_out is False


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_expanded(dtype):
    """An expanded tensor reads the same elements repeatedly, so it overlaps."""
    inp = torch.randn(1, 16, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(8, 16)
    ref_expanded = utils.to_reference(inp).expand(8, 16)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_expanded)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.is_non_overlapping_and_dense(expanded)

    assert res_out == ref_out
    assert res_out is False


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_unit_dims(dtype):
    """Size-1 dimensions carry unconstrained strides and must be ignored."""
    inp = torch.randn(4, 8, dtype=dtype, device=flag_gems.device)

    for candidate in (inp.unsqueeze(0), inp.unsqueeze(-1), inp.unsqueeze(1)):
        ref_out = torch.ops.aten.is_non_overlapping_and_dense(
            utils.to_reference(candidate)
        )

        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_non_overlapping_and_dense(candidate)

        assert res_out == ref_out
        assert res_out is True


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_non_overlapping_and_dense_small(dtype):
    """Scalar and empty tensors span their storage trivially."""
    for candidate in (
        torch.randn((), dtype=dtype, device=flag_gems.device),
        torch.randn(0, dtype=dtype, device=flag_gems.device),
        torch.randn(1, dtype=dtype, device=flag_gems.device),
    ):
        ref_out = torch.ops.aten.is_non_overlapping_and_dense(
            utils.to_reference(candidate)
        )

        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_non_overlapping_and_dense(candidate)

        assert res_out == ref_out
        assert res_out is True
