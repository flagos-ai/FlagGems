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

_SKIP_CAMBRICON = pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)


def _make_metadata(
    num_bags,
    embedding_dim,
    dtype,
    mode,
    weighted,
    scale_grad_by_freq,
    padding_idx,
):
    torch.manual_seed(42)
    num_weights = 50
    num_samples = num_bags * 3
    weight = torch.randn(
        num_weights, embedding_dim, dtype=dtype, device=flag_gems.device
    )
    # A short repeating range deterministically exercises duplicate-index
    # accumulation, frequency scaling, and the padding row when requested.
    indices = torch.arange(num_samples, device=flag_gems.device) % min(num_weights, 7)
    offsets = torch.arange(0, num_samples, 3, dtype=torch.long, device=flag_gems.device)
    per_sample_weights = (
        torch.rand(num_samples, dtype=dtype, device=flag_gems.device)
        if weighted
        else None
    )
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        False,
        per_sample_weights,
        False,
        padding_idx,
    )
    if offset2bag.numel() == 0:
        # Optimized sum forward may omit this optional output. The direct
        # backward API still requires the original sample-to-bag mapping.
        offset2bag = (
            torch.arange(num_bags, dtype=offset2bag.dtype, device="cpu")
            .repeat_interleave(3)
            .to(indices.device)
        )
    return (
        torch.randn_like(output),
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        per_sample_weights,
    )


@_SKIP_CAMBRICON
@pytest.mark.embedding_bag_dense_backward
@pytest.mark.underscore_embedding_bag_dense_backward
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "mode,weighted,scale_grad_by_freq,padding_idx",
    [
        (0, False, False, -1),
        (0, True, False, -1),
        (1, False, False, -1),
        (0, False, True, -1),
        (0, False, False, 0),
        (2, False, False, -1),
    ],
)
def test_embedding_bag_dense_backward(
    num_bags,
    embedding_dim,
    dtype,
    mode,
    weighted,
    scale_grad_by_freq,
    padding_idx,
):
    if (mode, weighted, scale_grad_by_freq, padding_idx) == (0, False, True, -1) and (
        flag_gems.vendor_name in {"ascend", "kunlunxin", "enflame"}
        or (flag_gems.vendor_name == "nvidia" and utils.TO_CPU)
    ):
        pytest.skip(
            "native reference frequency scaling differs from inverse minibatch "
            "frequency for the original repeated indices"
        )
    args = _make_metadata(
        num_bags,
        embedding_dim,
        dtype,
        mode,
        weighted,
        scale_grad_by_freq,
        padding_idx,
    )
    grad, indices, offset2bag, bag_size, maximum_indices, num_weights, weights = args
    ref_out = torch.ops.aten._embedding_bag_dense_backward(
        utils.to_reference(grad, True),
        utils.to_reference(indices, False),
        utils.to_reference(offset2bag, False),
        utils.to_reference(bag_size, False),
        utils.to_reference(maximum_indices, False),
        num_weights,
        scale_grad_by_freq,
        mode,
        None if weights is None else utils.to_reference(weights, True),
        padding_idx,
    )
    inputs = [grad, indices, offset2bag, bag_size, maximum_indices]
    if weights is not None:
        inputs.append(weights)
    before = [value.detach().cpu().clone() for value in inputs]
    before_layout = [
        (
            value.device,
            value.dtype,
            tuple(value.shape),
            value.stride(),
            value.storage_offset(),
            value.requires_grad,
            value.untyped_storage().data_ptr(),
        )
        for value in inputs
    ]
    result = flag_gems._embedding_bag_dense_backward(
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        weights,
        padding_idx,
    )
    assert isinstance(result, torch.Tensor)
    assert result.device == grad.device
    assert result is not grad
    assert result.untyped_storage().data_ptr() not in {
        value.untyped_storage().data_ptr() for value in inputs
    }
    assert result.shape == (num_weights, embedding_dim)
    assert result.dtype == dtype
    for value, original, layout in zip(inputs, before, before_layout):
        assert (
            value.device,
            value.dtype,
            tuple(value.shape),
            value.stride(),
            value.storage_offset(),
            value.requires_grad,
            value.untyped_storage().data_ptr(),
        ) == layout
        assert torch.equal(value.detach().cpu(), original)
    # Check the target ABI before the portable numerical comparison.
    utils.gems_assert_close(result.cpu(), ref_out.cpu(), dtype)
