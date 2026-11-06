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

from .accuracy_utils import gems_assert_close, to_reference
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_index_copy_duplicate_indices(dtype):
    # Issue #8300: CPU ATen applies the copies in order, so a repeated index
    # means the LAST source position wins — the kernel must match that instead
    # of racing the duplicate writers
    inp = torch.randn(2, 2, 8, 4, dtype=dtype, device=flag_gems.device)
    src = torch.randn(2, 2, 2, 4, dtype=dtype, device=flag_gems.device)
    index = torch.tensor([1, 1], device=flag_gems.device, dtype=torch.int64)
    # ATen CUDA index_copy is itself nondeterministic on a repeated index, so
    # the reference is computed on the CPU, where the copies apply in order
    ref_out = torch.index_copy(inp.cpu(), 2, index.cpu(), src.cpu())
    res_out = flag_gems.index_copy(inp, 2, index, src)
    gems_assert_close(res_out.cpu(), ref_out, dtype=dtype, reduce_dim=2)


@pytest.mark.index_copy_
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_index_copy__duplicate_indices(dtype):
    # Issue #8300, in-place variant with a TRIPLE duplicate: the marked row is
    # the last writer, so it must be the one that survives
    inp = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    src = torch.zeros(3, 6, dtype=dtype, device=flag_gems.device)
    src[2] = 5.0
    index = torch.tensor([2, 2, 2], device=flag_gems.device, dtype=torch.int64)

    ref_inp = inp.clone().cpu()
    ref_out = ref_inp.index_copy_(0, index.cpu(), src.cpu())

    res_out = flag_gems.index_copy_(inp.clone(), 0, index, src)
    gems_assert_close(res_out.cpu(), ref_out, dtype=dtype, reduce_dim=0)
    gems_assert_close(res_out[2].cpu(), torch.full_like(res_out[2].cpu(), 5.0), dtype=dtype)
