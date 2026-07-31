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

pytestmark = pytest.mark.nested_view_from_buffer

# Component layouts: (buffer_size, sizes, strides, offsets). Each component is a
# 1-D contiguous slice of the flat buffer, covering single- and multi-component
# nested tensors of varying sizes.
LAYOUTS = [
    (8, [[3], [5]], [[1], [1]], [0, 3]),
    (6000, [[1000], [2000], [3000]], [[1], [1], [1]], [0, 1000, 3000]),
    (100, [[100]], [[1]], [0]),
    (150, [[10], [20], [30], [40], [50]], [[1]] * 5, [0, 10, 30, 60, 100]),
]


def _make_metadata(sizes, strides, offsets):
    # NOTE: nested_size / nested_strides / offsets must live on CPU. Passing CUDA
    # metadata tensors to aten._nested_view_from_buffer segfaults (a PyTorch
    # NestedTensor limitation), so metadata is always constructed on CPU while the
    # buffer itself stays on the target device.
    nested_size = torch.tensor(sizes, dtype=torch.int64, device="cpu")
    nested_strides = torch.tensor(strides, dtype=torch.int64, device="cpu")
    offsets_t = torch.tensor(offsets, dtype=torch.int64, device="cpu")
    return nested_size, nested_strides, offsets_t


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_nested_view_from_buffer(layout, dtype):
    buffer_size, sizes, strides, offsets = layout
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)
    nested_size, nested_strides, offsets_t = _make_metadata(sizes, strides, offsets)

    ref_out = torch.ops.aten._nested_view_from_buffer.default(
        buffer, nested_size, nested_strides, offsets_t
    )
    with flag_gems.use_gems():
        res_out = flag_gems._nested_view_from_buffer(
            buffer, nested_size, nested_strides, offsets_t
        )

    assert res_out.is_nested
    assert ref_out.is_nested

    res_unbind = torch.unbind(res_out)
    ref_unbind = torch.unbind(ref_out)
    assert len(res_unbind) == len(ref_unbind) == len(sizes)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        utils.gems_assert_equal(res_t, ref_t)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_buffer_matches_slices(dtype):
    # Each component must equal the corresponding contiguous slice of the buffer.
    buffer_size, sizes, _, offsets = (60, [[10], [20], [30]], None, [0, 10, 30])
    buffer = torch.randn(buffer_size, dtype=dtype, device=flag_gems.device)
    nested_size, nested_strides, offsets_t = _make_metadata(
        sizes, [[1], [1], [1]], offsets
    )

    with flag_gems.use_gems():
        out = flag_gems._nested_view_from_buffer(
            buffer, nested_size, nested_strides, offsets_t
        )

    comps = torch.unbind(out)
    lengths = [s[0] for s in sizes]
    for i, comp in enumerate(comps):
        start = offsets[i]
        expected = buffer[start : start + lengths[i]]
        utils.gems_assert_equal(comp, expected)


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_buffer_is_zero_copy(dtype):
    # A view must share storage with the buffer: mutating the buffer is observable
    # through the nested tensor, and no copy is made.
    buffer = torch.randn(100, dtype=dtype, device=flag_gems.device)
    nested_size, nested_strides, offsets_t = _make_metadata(
        [[40], [60]], [[1], [1]], [0, 40]
    )

    with flag_gems.use_gems():
        out = flag_gems._nested_view_from_buffer(
            buffer, nested_size, nested_strides, offsets_t
        )

    # Storage identity between the buffer and the nested tensor's values.
    assert (
        out.values().untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    )

    # Mutation of the buffer is reflected in the view (proves no copy).
    buffer[0] = buffer[0] + 1.0
    expected_first = buffer[0].item()
    assert torch.unbind(out)[0][0].item() == expected_first
