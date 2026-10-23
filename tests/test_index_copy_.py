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

import math

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference
from .conftest import QUICK_MODE

DIM_LIST = [1] if QUICK_MODE else [0, 1]
INDEX_COPY_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 40999, 3)]


@pytest.mark.index_copy
@pytest.mark.parametrize("shape", INDEX_COPY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_copy(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_shape = list(inp.shape)
    index_len = src_shape[dim]
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_out = torch.index_copy(ref_inp, dim, ref_index, ref_src)
    res_out = flag_gems.index_copy(inp, dim, index, src)
    gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_copy_
@pytest.mark.parametrize("shape", INDEX_COPY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_copy_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_shape = list(inp.shape)
    index_len = src_shape[dim]
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_inp.index_copy_(dim, ref_index, ref_src)
    flag_gems.index_copy_(inp, dim, index, src)
    gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_copy
@pytest.mark.index_copy_
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend stride addressing"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "layout,dim",
    [
        ("gdn", 0),
        ("gdn_single", 0),
        ("transpose", 1),
        ("slice", -1),
        ("source", 0),
        ("expanded_source", 0),
        ("index", 1),
        ("empty_index", 0),
        ("empty_outer", 1),
    ],
)
def test_ascend_index_copy_strides(dtype, inplace, layout, dim):
    raw_shape = (4, 3, 4096) if layout.startswith("gdn") else (6, 7, 10)
    cpu_raw = (torch.arange(math.prod(raw_shape)) % 97).reshape(raw_shape).to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    initial = raw.clone()
    if layout.startswith("gdn"):
        inp = raw.transpose(1, 2)
        reference = cpu_raw.transpose(1, 2)
    elif layout in ("transpose", "empty_index"):
        inp = raw.transpose(0, 2)
        reference = cpu_raw.transpose(0, 2)
    elif layout == "slice":
        inp = raw[1:5:2, 1:6:2, 1:9:2]
        reference = cpu_raw[1:5:2, 1:6:2, 1:9:2]
    elif layout == "empty_outer":
        inp = raw[:0]
        reference = cpu_raw[:0]
    else:
        inp, reference = raw, cpu_raw

    axis = dim % inp.ndim
    positions = [] if layout == "empty_index" else [1, inp.shape[axis] - 1]
    if layout == "gdn_single":
        positions = [1]
    cpu_index = torch.tensor(positions, dtype=torch.int64)
    if layout == "index":
        index_storage = torch.zeros(4, dtype=torch.int64, device=flag_gems.device)
        index = index_storage[::2]
        index.copy_(cpu_index)
    else:
        index = cpu_index.to(flag_gems.device)
    source_shape = list(inp.shape)
    source_shape[axis] = len(positions)
    cpu_source = (
        (-(torch.arange(math.prod(source_shape)) % 89 + 1))
        .reshape(source_shape)
        .to(dtype)
    )
    if layout in ("slice", "source"):
        source_storage = torch.empty(
            (*source_shape[:-1], 2 * source_shape[-1] + 1),
            dtype=dtype,
            device=flag_gems.device,
        )
        source = source_storage[..., 1::2]
        source.copy_(cpu_source)
    elif layout == "expanded_source":
        cpu_source = cpu_source[:1].expand(source_shape)
        source = cpu_source[:1].to(flag_gems.device).expand(source_shape)
    else:
        source = cpu_source.to(flag_gems.device)
    # Independent CPU reference, including all untouched backing-storage lanes.
    reference.index_copy_(dim, cpu_index, cpu_source)
    if inplace:
        actual = flag_gems.index_copy_(inp, dim, index, source)
        assert actual is inp
        gems_assert_equal(raw.cpu(), cpu_raw)
    else:
        actual = flag_gems.index_copy(inp, dim, index, source)
        assert not torch._C._is_alias_of(actual, inp)
        gems_assert_equal(raw.cpu(), initial.cpu())
    gems_assert_equal(actual.cpu(), reference)
