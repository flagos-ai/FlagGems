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

import importlib
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


@pytest.mark.index_copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend row copies")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["gdn", "prefix_transpose"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("registered", [False, True])
def test_ascend_index_copy_wide_row_graph(dtype, layout, index_dtype, registered):
    raw_shape = (6, 3, 4097) if layout == "gdn" else (6, 2, 257)
    cpu_raw = (torch.arange(math.prod(raw_shape)) % 97).reshape(raw_shape).to(dtype)
    raw = cpu_raw.to(flag_gems.device)
    inp = raw[1:5].transpose(1, 2) if layout == "gdn" else raw[1:5].transpose(0, 1)
    dim = 0 if layout == "gdn" else 1
    source_shape = list(inp.shape)
    source_shape[dim] = 2
    source_cpu = torch.arange(math.prod(source_shape)).reshape(source_shape).to(dtype)
    source_cpu = source_cpu % 97
    # Construct special values in the source dtype, then compare backing bits.
    int_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    if dtype == torch.float32:
        bits = [0, -2147483648, 2139095040, -8388608, 2143289410]
    elif dtype == torch.bfloat16:
        bits = [0, -32768, 32640, -128, 32705]
    else:
        bits = [0, -32768, 31744, -1024, 32257]
    source_cpu.flatten()[:5] = torch.tensor(bits, dtype=int_dtype).view(dtype)
    storage = torch.empty(
        (*source_shape[:-1], 2 * source_shape[-1] + 1), dtype=dtype, device=inp.device
    )
    source = storage[..., 1::2]
    source.copy_(source_cpu)
    index_storage = torch.tensor([1, 99, 3, 99], dtype=index_dtype, device=inp.device)
    index = index_storage[::2]
    previous_registrar = flag_gems.current_work_registrar
    lib = None
    try:
        if registered:
            lib = torch.library.Library("aten", "IMPL")
            flag_gems.only_enable(lib=lib, include=["index_copy_"])

        def invoke():
            return (
                inp.index_copy_(dim, index, source)
                if registered
                else flag_gems.index_copy_(inp, dim, index, source)
            )

        invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        for positions in ([0, 2], [1, 3]):
            raw.copy_(cpu_raw)
            index.copy_(torch.tensor(positions, dtype=index_dtype))
            reference_raw = cpu_raw.clone()
            reference = (
                reference_raw[1:5].transpose(1, 2)
                if layout == "gdn"
                else reference_raw[1:5].transpose(0, 1)
            )
            reference.index_copy_(dim, torch.tensor(positions), source_cpu)
            graph.replay()
            torch.npu.synchronize()
            assert actual is inp
            gems_assert_equal(raw.cpu().view(int_dtype), reference_raw.view(int_dtype))
    finally:
        if lib is not None:
            lib._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.index_copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend row traversal")
def test_ascend_index_copy_row_bounded_grid():
    module = importlib.import_module(flag_gems.index_copy_.__module__)
    inp = torch.zeros((4, 3, 513), device=flag_gems.device).transpose(1, 2)
    source_cpu = torch.arange(2 * 513 * 3, dtype=torch.float32).reshape(2, 513, 3)
    source = source_cpu.to(inp.device)
    indices = torch.tensor([1, 3], device=inp.device)
    module._index_copy_strided_row_kernel.jit_function[(2,)](
        inp,
        source,
        indices,
        6,
        513,
        4,
        ROW_SHAPE=(2, 3),
        OUT_STRIDES=(1539, 513),
        SRC_STRIDES=(1539, 1),
        DIM=0,
        SRC_INNER_STRIDE=3,
        INDEX_STRIDE=1,
        BLOCK=256,
    )
    expected = torch.zeros((4, 513, 3))
    expected.index_copy_(0, torch.tensor([1, 3]), source_cpu)
    gems_assert_equal(inp.cpu(), expected)
