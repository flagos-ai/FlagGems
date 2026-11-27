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
from .test_scatter import INPUT_SHAPES, SOURCE_SHAPES


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scatter_reduce_add(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
    res_out = flag_gems.scatter(inp, dim, index, src, reduce="add")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scatter_reduce_multiply(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
    res_out = flag_gems.scatter(inp, dim, index, src, reduce="multiply")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scatter_reduce_add_(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]

    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="add")
    res_out = flag_gems.scatter_(inp.clone(), dim, index, src, reduce="add")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scatter_reduce_multiply_(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="multiply")
    res_out = flag_gems.scatter_(inp.clone(), dim, index, src, reduce="multiply")

    utils.gems_assert_close(res_out, ref_out, dtype)
