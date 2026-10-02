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

import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_padded(batch_size, max_length, inner_dims, dtype, device, seed=42):
    """Create a padded tensor and the matching nested-size example tensor."""
    rng = random.Random(seed)
    lengths = [rng.randint(1, max_length) for _ in range(batch_size)]
    padded = torch.randn(
        (batch_size, max_length) + tuple(inner_dims), dtype=dtype, device=device
    )
    sizes = torch.tensor([[ln] + list(inner_dims) for ln in lengths], dtype=torch.int64)
    return padded, sizes, lengths


def _check_nested_metadata(res_out, ref_out):
    """Assert the result is a legacy NestedTensor with the same structural metadata.

    The reference aten implementation returns a legacy (``torch.strided`` layout)
    NestedTensor, so FlagGems must match it rather than returning a jagged
    NestedTensor -- otherwise downstream ops relying on strided layout would
    silently take a different code path.
    """
    assert res_out.layout == ref_out.layout
    assert res_out.is_nested == ref_out.is_nested
    assert torch.equal(res_out._nested_tensor_size(), ref_out._nested_tensor_size())
    assert torch.equal(
        res_out._nested_tensor_strides(), ref_out._nested_tensor_strides()
    )
    assert torch.equal(
        res_out._nested_tensor_storage_offsets(),
        ref_out._nested_tensor_storage_offsets(),
    )


def _assert_nested_close(res_out, ref_out, dtype):
    _check_nested_metadata(res_out, ref_out)
    res_unbind = torch.unbind(res_out)
    ref_unbind = torch.unbind(ref_out)
    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("shape", [(4, 8, [16]), (3, 6, [2, 4]), (16, 32, [64])])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded(shape, dtype):
    batch_size, max_length, inner_dims = shape
    padded, sizes, _ = _make_padded(
        batch_size, max_length, inner_dims, dtype, flag_gems.device
    )

    ref_padded = utils.to_reference(padded)
    ref_out = torch.ops.aten._nested_from_padded(ref_padded, sizes)

    res_out = flag_gems._nested_from_padded(padded, sizes)

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded
# fuse_transform_0213 CUDA kernel only supports fp32/fp16; bf16 falls back to a
# slow generic path, so restrict to the two fast dtypes.
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nested_from_padded_transform_0213(dtype):
    # (B, L, S, D) -> fuse_transform_0213 -> (B, S, L * D)
    batch_size, L, S, D = 2, 4, 3, 5
    padded = torch.randn(batch_size, L, S, D, dtype=dtype, device=flag_gems.device)
    lengths = [2, 3]
    sizes = torch.tensor([[ln, L * D] for ln in lengths], dtype=torch.int64)

    ref_padded = utils.to_reference(padded)
    ref_out = torch.ops.aten._nested_from_padded(
        ref_padded, sizes, fuse_transform_0213=True
    )

    res_out = flag_gems._nested_from_padded(padded, sizes, fuse_transform_0213=True)

    _assert_nested_close(res_out, ref_out, dtype)
