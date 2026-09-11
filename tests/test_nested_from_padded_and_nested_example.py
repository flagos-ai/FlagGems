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

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="CUDA-only op; no CPU reference implementation"
)


def _make_example(B, D_pad, D, lengths, dtype, device):
    """Build a raw legacy nested tensor whose component sizes encode ``lengths``."""
    if D == 1:
        data = torch.randn(B, D_pad, dtype=dtype, device=device)
        sizes = lengths.reshape(-1, 1)
    else:
        data = torch.randn(B, D_pad, D, dtype=dtype, device=device)
        sizes = torch.stack([lengths, torch.full((B,), D, dtype=torch.int64)], dim=1)
    return torch.ops.aten._nested_from_padded(data, sizes)


def _check_nested_metadata(res_out, ref_out):
    """Assert the result is a legacy NestedTensor with the same structural metadata."""
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


def _check_components(res_out, ref_out, dtype):
    """Compare the result against the reference component-wise."""
    _check_nested_metadata(res_out, ref_out)
    res_comps = torch.unbind(res_out)
    ref_comps = torch.unbind(ref_out)
    assert len(res_comps) == len(ref_comps)
    for res_c, ref_c in zip(res_comps, ref_comps):
        assert res_c.shape == ref_c.shape
        utils.gems_assert_close(res_c, ref_c, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_2d(dtype):
    B, D_pad = 4, 16
    device = flag_gems.device
    lengths = torch.tensor([3, 9, 5, 13], dtype=torch.int64)

    padded = torch.randn(B, D_pad, dtype=dtype, device=device)
    nt_example = _make_example(B, D_pad, 1, lengths, dtype, device)

    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        utils.to_reference(padded), nt_example
    )
    res_out = flag_gems._nested_from_padded_and_nested_example(padded, nt_example)

    _check_components(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_3d(dtype):
    B, D_pad, D = 4, 16, 8
    device = flag_gems.device
    lengths = torch.tensor([3, 9, 5, 13], dtype=torch.int64)

    padded = torch.randn(B, D_pad, D, dtype=dtype, device=device)
    nt_example = _make_example(B, D_pad, D, lengths, dtype, device)

    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        utils.to_reference(padded), nt_example
    )
    res_out = flag_gems._nested_from_padded_and_nested_example(padded, nt_example)

    _check_components(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_3d_trailing_ragged(dtype):
    # Trailing dimension is also ragged: sizes [[3, 4], [5, 2]] inside a
    # [2, 8, 4] padded tensor exercises the general (non-contiguous-prefix) path.
    B, D_pad0, D_pad1 = 2, 8, 4
    device = flag_gems.device
    sizes = torch.tensor([[3, 4], [5, 2]], dtype=torch.int64)

    padded = torch.randn(B, D_pad0, D_pad1, dtype=dtype, device=device)
    nt_example = torch.ops.aten._nested_from_padded(torch.randn_like(padded), sizes)

    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        utils.to_reference(padded), nt_example
    )
    res_out = flag_gems._nested_from_padded_and_nested_example(padded, nt_example)

    _check_components(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example_4d_trailing_ragged(dtype):
    # 4D component with a ragged trailing dimension exercises the general path
    # with RANK == 3.
    B, D_pad0, D_pad1, D_pad2 = 2, 8, 4, 3
    device = flag_gems.device
    sizes = torch.tensor([[3, 4, 3], [5, 2, 1]], dtype=torch.int64)

    padded = torch.randn(B, D_pad0, D_pad1, D_pad2, dtype=dtype, device=device)
    nt_example = torch.ops.aten._nested_from_padded(torch.randn_like(padded), sizes)

    ref_out = torch.ops.aten._nested_from_padded_and_nested_example(
        utils.to_reference(padded), nt_example
    )
    res_out = flag_gems._nested_from_padded_and_nested_example(padded, nt_example)

    _check_components(res_out, ref_out, dtype)
