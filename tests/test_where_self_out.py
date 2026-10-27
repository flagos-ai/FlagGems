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


@pytest.mark.where_self
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_where_self(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp1 > 0, ref_inp1, ref_inp2)
    res_out = flag_gems.where_self(inp1 > 0, inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self
@pytest.mark.where_scalar_self
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_where_self_scalar(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, inp1, ref_inp2)
    res_out = flag_gems.where_scalar_self(inp2 > 0, inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self
@pytest.mark.where_scalar_other
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_where_self_scalar_other(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, ref_inp2, inp1)
    res_out = flag_gems.where_scalar_other(inp2 > 0, inp2, inp1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_where_self_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(out)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_cond = utils.to_reference(cond)

    ref_out = torch.where(ref_cond, ref_inp1, ref_inp2, out=ref_out)
    res_out = flag_gems.where_self_out(cond, inp1, inp2, out=out)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_where_self_out_cross_device(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

    import itertools

    shapes = (shape, None)
    for a_shape, b_shape, c_shape in itertools.product(shapes, shapes, shapes):
        if not (a_shape and b_shape and c_shape):
            # Combination with CPU scalars only: the dispatcher selects torch's
            # native CPU kernel, so the GEMS where kernel (which asserts on a
            # device tensor) is never in play here. Skip that combination.
            continue
        a = inp1 if a_shape else torch.tensor(0)
        b = inp2 if b_shape else torch.tensor(1)
        c = cond if c_shape else torch.tensor(True)

        ref_a = utils.to_reference(a)
        ref_b = utils.to_reference(b)
        ref_c = utils.to_reference(c)

        ref_out = torch.where(ref_c, ref_a, ref_b)
        res_out = flag_gems.where_self(c, a, b)

        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self
@pytest.mark.where_self_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend uniform condition"
)
@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.bool]
)
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_ascend_where_uniform_condition_graph(
    width, expanded, dtype, registered, use_out
):
    left_cpu = (torch.arange(2 * width).reshape(2, width) % 7).to(dtype)
    right_cpu = ((torch.arange(2 * width).reshape(2, width) + 1) % 5).to(dtype)
    left = left_cpu.to(flag_gems.device)[1]
    right = right_cpu.to(flag_gems.device)[1]
    condition_storage = torch.tensor([False, True], device=left.device)
    condition = condition_storage[1]
    if expanded:
        condition = condition.expand(width)
    backing = torch.full((width, 2), 1, dtype=dtype, device=left.device)
    output = backing[:, 1]
    lib = None
    previous_registrar = flag_gems.current_work_registrar
    try:
        if registered:
            lib = torch.library.Library("aten", "IMPL")
            flag_gems.only_enable(lib=lib, include=["where_self", "where_self_out"])

        def invoke():
            if registered:
                return (
                    torch.where(condition, left, right, out=output)
                    if use_out
                    else torch.where(condition, left, right)
                )
            if use_out:
                return flag_gems.where_self_out(condition, left, right, out=output)
            return flag_gems.where_self(condition, left, right)

        actual = invoke()
        torch.npu.synchronize()
        utils.gems_assert_equal(actual.cpu(), left_cpu[1])
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        for value in (False, True):
            condition_storage[1].fill_(value)
            graph.replay()
            torch.npu.synchronize()
            expected = left_cpu[1] if value else right_cpu[1]
            utils.gems_assert_equal(actual.cpu(), expected)
            if use_out:
                assert actual is output
                utils.gems_assert_equal(
                    backing[:, 0].cpu(), torch.ones(width, dtype=dtype)
                )
    finally:
        if lib is not None:
            lib._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.where_self
@pytest.mark.where_self_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend uniform condition"
)
@pytest.mark.parametrize("value", [False, True])
def test_ascend_where_uniform_condition_strided_broadcast(value):
    left_cpu = torch.tensor([[1], [2]], dtype=torch.int32)
    right_cpu = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float32)
    condition = torch.tensor(value, device=flag_gems.device).expand(2, 3)
    left, right = left_cpu.to(flag_gems.device), right_cpu.to(flag_gems.device)
    output = torch.empty((3, 2), device=left.device).t()
    actual = flag_gems.where_self_out(condition, left, right, out=output)
    expected = torch.where(torch.tensor(value), left_cpu, right_cpu)
    assert actual is output
    assert actual.dtype == expected.dtype
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.where_scalar_self
@pytest.mark.where_scalar_other
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend uniform condition"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_ascend_where_uniform_condition_scalar(dtype, scalar_first):
    values_cpu = torch.arange(21).reshape(3, 7).t().to(dtype)
    values = values_cpu.to(flag_gems.device)
    for value in (False, True):
        condition = torch.tensor([[value]], device=values.device)
        if scalar_first:
            actual = flag_gems.where_scalar_self(condition, 1.25, values)
            expected = torch.where(torch.tensor([[value]]), 1.25, values_cpu)
        else:
            actual = flag_gems.where_scalar_other(condition, values, 1.25)
            expected = torch.where(torch.tensor([[value]]), values_cpu, 1.25)
        assert actual.dtype == expected.dtype
        utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.where_self
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend uniform condition"
)
@pytest.mark.parametrize("shape", [(), (0, 3), (65537,)])
def test_ascend_where_uniform_condition_boundaries(shape):
    left_cpu = torch.ones(shape, dtype=torch.bfloat16)
    right_cpu = torch.zeros(shape, dtype=torch.bfloat16)
    left, right = left_cpu.to(flag_gems.device), right_cpu.to(flag_gems.device)
    condition = torch.tensor(True, device=left.device)
    actual = flag_gems.where_self(condition, left, right)
    utils.gems_assert_equal(actual.cpu(), left_cpu)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = flag_gems.where_self(condition, left, right)
    condition.fill_(False)
    graph.replay()
    torch.npu.synchronize()
    utils.gems_assert_equal(actual.cpu(), right_cpu)


@pytest.mark.where_self
@pytest.mark.where_self_out
@pytest.mark.where_scalar_other
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend row masks")
@pytest.mark.parametrize(
    "shape", [(2, 17, 128), (2, 17, 257), (1, 3, 1024), (2, 0, 128)]
)
@pytest.mark.parametrize("mode", ["python", "tensor", "out"])
@pytest.mark.parametrize("registered", [False, True])
def test_ascend_where_row_scalar_graph(shape, mode, registered):
    numel = shape[0] * shape[1] * shape[2]
    values_cpu = ((torch.arange(numel) % 256) - 128).to(torch.int8).reshape(shape)
    storage = torch.empty(numel + 1, dtype=torch.int8, device=flag_gems.device)
    storage[0] = 37
    values = storage[1:].view(shape)
    values.copy_(values_cpu)
    rows = shape[0] * shape[1]
    condition_storage = torch.zeros(rows + 1, dtype=torch.bool, device=values.device)
    condition = condition_storage[1:].view(*shape[:-1], 1)
    scalar = torch.tensor(0, dtype=torch.int8, device=values.device)
    backing = torch.full((numel + 1,), 33, dtype=torch.int8, device=values.device)
    output = backing[1:].view(shape) if mode == "out" else None
    other = 0 if mode == "python" else scalar
    previous_registrar = flag_gems.current_work_registrar
    lib = None
    try:
        if registered:
            lib = torch.library.Library("aten", "IMPL")
            flag_gems.only_enable(lib=lib, include=["where_self", "where_self_out"])

        def invoke():
            if registered:
                if output is not None:
                    return torch.where(condition, values, other, out=output)
                return torch.where(condition, values, other)
            return flag_gems.where_self_out(condition, values, other, out=output)

        first, second = invoke(), invoke()
        if numel and output is None:
            assert first.data_ptr() != second.data_ptr()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        for value, fill in [(False, -128), (True, 127)]:
            condition.fill_(value)
            scalar.fill_(fill)
            graph.replay()
            torch.npu.synchronize()
            expected = (
                values_cpu
                if value
                else torch.full_like(values_cpu, 0 if mode == "python" else fill)
            )
            utils.gems_assert_equal(actual.cpu(), expected)
            if output is not None:
                assert actual is output
                utils.gems_assert_equal(
                    backing[:1].cpu(), torch.tensor([33], dtype=torch.int8)
                )
        utils.gems_assert_equal(storage[:1].cpu(), torch.tensor([37], dtype=torch.int8))
        utils.gems_assert_equal(values.cpu(), values_cpu)
    finally:
        if lib is not None:
            lib._destroy()
        flag_gems.current_work_registrar = previous_registrar
