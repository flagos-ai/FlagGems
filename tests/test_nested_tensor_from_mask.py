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

# (N, L, D) shapes for the padded dense tensor `t` and the (N, L) mask.
NESTED_FROM_MASK_SHAPES = [
    (1, 4, 3),
    (2, 8, 4),
    (8, 16, 8),
    (16, 64, 32),
    (64, 128, 16),
    (4, 256, 64),
]


def _left_aligned_mask(N, L, device):
    """Generate a random left-aligned boolean mask of shape (N, L)."""
    lengths = torch.randint(0, L + 1, (N,), device=device)
    idx = torch.arange(L, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def _check_nested_metadata(res, ref):
    """Assert the result is a legacy NestedTensor with the same structural metadata.

    The reference aten implementation returns a legacy (``torch.strided`` layout)
    NestedTensor, so FlagGems must match it rather than returning a jagged
    NestedTensor -- otherwise downstream ops relying on strided layout would
    silently take a different code path.
    """
    assert res.layout == ref.layout
    assert res.is_nested == ref.is_nested
    assert torch.equal(res._nested_tensor_size(), ref._nested_tensor_size())
    assert torch.equal(res._nested_tensor_strides(), ref._nested_tensor_strides())
    assert torch.equal(
        res._nested_tensor_storage_offsets(), ref._nested_tensor_storage_offsets()
    )


def _assert_nested_equal(res, ref, dtype):
    _check_nested_metadata(res, ref)
    res_comps = torch.unbind(res)
    ref_comps = torch.unbind(ref)
    assert len(res_comps) == len(ref_comps)
    for res_t, ref_t in zip(res_comps, ref_comps):
        assert res_t.shape == ref_t.shape
        utils.gems_assert_close(res_t, ref_t, dtype)


@pytest.mark.nested_tensor_from_mask
@pytest.mark.parametrize("shape", NESTED_FROM_MASK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask(shape, dtype):
    N, L, D = shape
    t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = _left_aligned_mask(N, L, flag_gems.device)

    ref_t = utils.to_reference(t)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=True)
    res_out = flag_gems._nested_tensor_from_mask(t, mask, mask_check=True)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_tensor_from_mask
@pytest.mark.parametrize("shape", NESTED_FROM_MASK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_no_mask_check(shape, dtype):
    N, L, D = shape
    t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Arbitrary (possibly non-left-aligned) mask.
    mask = torch.randint(0, 2, (N, L), dtype=torch.bool, device=flag_gems.device)

    ref_t = utils.to_reference(t)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=False)
    res_out = flag_gems._nested_tensor_from_mask(t, mask, mask_check=False)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_tensor_from_mask
def test_nested_tensor_from_mask_mask_check_raises():
    N, L, D = 4, 8, 3
    t = torch.randn(N, L, D, device=flag_gems.device)
    # A mask with a gap (False followed by True) is not left-aligned.
    mask = torch.tensor(
        [
            [True, False, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
        device=flag_gems.device,
    )
    with pytest.raises(RuntimeError):
        flag_gems._nested_tensor_from_mask(t, mask, mask_check=True)
