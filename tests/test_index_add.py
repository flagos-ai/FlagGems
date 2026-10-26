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
import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
else:
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


CONTIGUOUS_SUFFIX_CASES = [
    ((1024, 4), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
    ((2, 8, 2048, 32), 2),
    ((1024, 64), 0),
]


def _make_repeated_index(index_len):
    index_range = max(index_len // 2, 1)
    return torch.arange(index_len, device=flag_gems.device) % index_range


@pytest.mark.index_add
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_inplace_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_invalid_index(inplace):
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, shape[dim]], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        if inplace:
            flag_gems.index_add_(inp, dim, index, src)
        else:
            flag_gems.index_add(inp, dim, index, src)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend capture")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("shape", [(16,), (16, 8)])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_ascend_index_add_graph_replay(dtype, shape, inplace, registered):
    base = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
    inp = base.to(flag_gems.device)
    base_device = inp.clone()
    index = torch.tensor([1, 4], device=inp.device, dtype=torch.int64)
    src_base = torch.ones((2, *shape[1:]), dtype=dtype)
    src = src_base.to(inp.device)
    lib = None
    previous_registrar = flag_gems.current_work_registrar
    try:
        if registered:
            lib = torch.library.Library("aten", "IMPL")
            flag_gems.only_enable(lib=lib, include=["index_add", "index_add_"])

        def invoke():
            inp.copy_(base_device)
            if registered:
                return (
                    inp.index_add_(0, index, src)
                    if inplace
                    else torch.index_add(inp, 0, index, src)
                )
            return (
                flag_gems.index_add_(inp, 0, index, src)
                if inplace
                else flag_gems.index_add(inp, 0, index, src)
            )

        # Compile before capture, then exercise new device values on every replay.
        invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        assert torch._C._is_alias_of(actual, inp) == inplace
        for positions in ([3, 15], [2, 2], [0, 0]):
            cpu_index = torch.tensor(positions, dtype=torch.int64)
            index.copy_(cpu_index)
            src.copy_(src_base * 2)
            graph.replay()
            torch.npu.synchronize()
            expected = base.index_add(0, cpu_index, src_base * 2)
            utils.gems_assert_equal(actual.cpu(), expected)
            if not inplace:
                utils.gems_assert_equal(inp.cpu(), base)
    finally:
        if lib is not None:
            lib._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend validation")
@pytest.mark.parametrize("shape", [(8,), (8, 3)])
@pytest.mark.parametrize("bad_index", [-1, 8])
@pytest.mark.parametrize("inplace", [False, True])
def test_ascend_index_add_invalid_preserves_input(shape, bad_index, inplace):
    inp = torch.ones(shape, device=flag_gems.device)
    initial = inp.clone()
    index = torch.tensor([1, bad_index], device=inp.device)
    src = torch.ones((2, *shape[1:]), device=inp.device)
    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        if inplace:
            flag_gems.index_add_(inp, 0, index, src)
        else:
            flag_gems.index_add(inp, 0, index, src)
    utils.gems_assert_equal(inp.cpu(), initial.cpu())


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend empty updates")
@pytest.mark.parametrize("inplace", [False, True])
def test_ascend_index_add_empty_graph(inplace):
    inp = torch.ones((8, 3), device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=inp.device)
    src = torch.empty((0, 3), device=inp.device)

    def invoke():
        return (
            flag_gems.index_add_(inp, 0, index, src)
            if inplace
            else flag_gems.index_add(inp, 0, index, src)
        )

    invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = invoke()
    graph.replay()
    torch.npu.synchronize()
    assert torch._C._is_alias_of(actual, inp) == inplace
    utils.gems_assert_equal(actual.cpu(), inp.cpu())


@pytest.mark.index_add_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend padding updates")
@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("registered", [False, True])
def test_ascend_index_add_padding_graph(tokens, registered):
    base = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    cache = base.to(flag_gems.device)
    base_device = cache.clone()
    slots = torch.arange(tokens, dtype=torch.int64, device=cache.device)
    values_cpu = torch.arange(tokens * 8, dtype=torch.float32).reshape(tokens, 8) / 8
    values = values_cpu.to(cache.device)
    lib = None
    previous_registrar = flag_gems.current_work_registrar
    try:
        if registered:
            lib = torch.library.Library("aten", "IMPL")
            flag_gems.only_enable(lib=lib, include=["index_add_"])

        def invoke():
            cache.copy_(base_device)
            valid = (slots >= 0) & (slots < cache.shape[0])
            indices = slots.clamp(0, cache.shape[0] - 1)
            updates = torch.zeros_like(cache)
            masked = torch.where(valid[:, None], values, 0)
            touched = torch.zeros(
                cache.shape[0], device=cache.device, dtype=torch.int32
            )
            if registered:
                updates.index_add_(0, indices, masked)
                touched.index_add_(0, indices, valid.to(torch.int32))
            else:
                flag_gems.index_add_(updates, 0, indices, masked)
                flag_gems.index_add_(touched, 0, indices, valid.to(torch.int32))
            cache.copy_(torch.where(touched[:, None] > 0, updates, cache))
            return touched

        invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            touched = invoke()
        for positions in ([3, 31], [-1, 6], [-1, -1], [4, 4], [32, -1]):
            cpu_slots = torch.tensor(positions[:tokens], dtype=torch.int64)
            slots.copy_(cpu_slots)
            graph.replay()
            torch.npu.synchronize()
            valid = (cpu_slots >= 0) & (cpu_slots < base.shape[0])
            indices = cpu_slots.clamp(0, base.shape[0] - 1)
            updates = torch.zeros_like(base).index_add_(
                0, indices, torch.where(valid[:, None], values_cpu, 0)
            )
            counts = torch.zeros(base.shape[0], dtype=torch.int32).index_add_(
                0, indices, valid.to(torch.int32)
            )
            expected = torch.where(counts[:, None] > 0, updates, base)
            utils.gems_assert_equal(cache.cpu(), expected)
            utils.gems_assert_equal(touched.cpu(), counts)
    finally:
        if lib is not None:
            lib._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.index_add_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend short indices")
@pytest.mark.parametrize(
    "shape,dim,tokens", [((3, 5), 1, 2), ((3, 5, 7), 1, 1), ((2, 5, 513), 1, 2)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32]
)
@pytest.mark.parametrize("duplicate", [False, True])
def test_ascend_index_add_short_suffix(shape, dim, tokens, dtype, duplicate):
    base = (torch.arange(math.prod(shape)) % 8).reshape(shape)
    src_shape = list(shape)
    src_shape[dim] = tokens
    source = (torch.arange(math.prod(src_shape)) % 7 - 3).reshape(src_shape)
    if dtype != torch.int32:
        base, source = base / 8, source / 8
    base, source = base.to(dtype), source.to(dtype)
    indices = torch.tensor(([2, 2] if duplicate else [1, 3])[:tokens])
    expected = base.index_add(dim, indices, source, alpha=2)
    inp = base.to(flag_gems.device)
    actual = flag_gems.index_add_(
        inp, dim, indices.to(inp.device), source.to(inp.device), alpha=2
    )
    assert actual is inp
    utils.gems_assert_equal(actual.cpu(), expected)
