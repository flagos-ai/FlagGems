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

FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES = [
    (4, 4, 4),
    (8, 8, 8),
    (3, 3, 3),
    (5, 5, 5),
    (7, 7, 7),
    (8, 8, 8),
]
ADAPTIVE_AVG_POOL3D_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 7, 7, 7),
    (1, 2, 10, 10, 10),
    (1, 1, 3, 3, 3),
    (1, 1, 2, 2, 2),
]


def _pool3d_raw_cpu(value):
    return torch.empty(0, dtype=value.dtype, device="cpu").set_(
        value.untyped_storage().cpu(),
        value.storage_offset(),
        value.shape,
        value.stride(),
    )


def _pool3d_input_snapshot(value):
    return (
        value.device,
        value.dtype,
        value.shape,
        value.stride(),
        value.storage_offset(),
        value.untyped_storage()._cdata,
        value.requires_grad,
        value.untyped_storage().data_ptr(),
        value.untyped_storage().nbytes(),
        value.is_conj(),
        value.is_neg(),
    ), _pool3d_raw_cpu(value).contiguous().reshape(-1).view(torch.uint8).clone()


@pytest.mark.underscore_adaptive_avg_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_AVG_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_adaptive_avg_pool3d_backward(shape, output_size, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (*shape[:-3], *output_size), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    expected = torch.ops.aten._adaptive_avg_pool3d_backward.default(
        ref_grad_output, ref_inp
    )
    snapshots = [_pool3d_input_snapshot(value) for value in (grad_output, inp)]
    try:
        actual = flag_gems._adaptive_avg_pool3d_backward(grad_output, inp)
    finally:
        for value, (metadata, contents) in zip((grad_output, inp), snapshots):
            after_metadata, after_contents = _pool3d_input_snapshot(value)
            assert after_metadata == metadata, "candidate changed input metadata"
            assert torch.equal(
                after_contents, contents
            ), "candidate changed input values"

    assert isinstance(actual, torch.Tensor)
    assert actual.device == inp.device
    assert actual.shape == inp.shape
    assert actual.dtype == inp.dtype
    actual_storage = actual.untyped_storage()
    assert all(
        actual_storage._cdata != metadata[5] for metadata, _ in snapshots
    ), "candidate output must not alias either input"
    if actual.numel():
        actual_start = actual_storage.data_ptr()
        actual_end = actual_start + actual_storage.nbytes()
        for metadata, _ in snapshots:
            if actual.device == metadata[0] and metadata[8]:
                input_start = metadata[7]
                input_end = input_start + metadata[8]
                assert (
                    actual_end <= input_start or input_end <= actual_start
                ), "candidate output storage must not overlap either input"
    utils.gems_assert_close(
        actual.detach().cpu(),
        expected.detach().cpu(),
        dtype,
        reduce_dim=output_size[0] * output_size[1] * output_size[2],
    )
