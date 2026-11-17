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
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + [torch.int64])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_where_uniform_condition_strided_storage(dtype, expanded, value):
    left_cpu = torch.arange(21).reshape(7, 3).t()
    right_cpu = -torch.arange(7).reshape(1, 7) - 1
    if dtype == torch.int64:
        left_cpu = left_cpu + 2**54
        right_cpu = right_cpu - 2**54
    left_cpu, right_cpu = left_cpu.to(dtype), right_cpu.to(dtype)
    left, right = left_cpu.to(flag_gems.device), right_cpu.to(flag_gems.device)
    storage = torch.tensor([not value, value], device=flag_gems.device)
    condition = storage[1].expand(3, 7) if expanded else storage[1]
    backing = torch.full((3, 14), 123, dtype=dtype, device=flag_gems.device)
    out = backing[:, 1::2]
    expected = torch.where(torch.tensor(value), left_cpu, right_cpu)
    actual = flag_gems.where_self_out(condition, left, right, out=out)
    assert actual is out
    expected_backing = torch.full((3, 14), 123, dtype=dtype)
    expected_backing[:, 1::2].copy_(expected)
    utils.gems_assert_equal(backing.cpu(), expected_backing)


@pytest.mark.where_self
@pytest.mark.parametrize("shape", [(), (0, 7), (1,), (257,)])
def test_where_uniform_condition_boundaries(shape):
    left = torch.ones(shape, device=flag_gems.device)
    right = torch.full(shape, -2.0, device=flag_gems.device)
    storage = torch.tensor([True, False], device=flag_gems.device)
    condition = storage[1].expand(shape)
    actual = flag_gems.where_self(condition, left, right)
    utils.gems_assert_equal(actual.cpu(), torch.full(shape, -2.0))


@pytest.mark.where_scalar_self
@pytest.mark.where_scalar_other
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scalar_first", [False, True])
def test_where_uniform_condition_scalar(dtype, scalar_first):
    cpu = torch.arange(21).reshape(3, 7).to(dtype)
    inp = cpu.to(flag_gems.device)
    storage = torch.tensor([False, True], device=inp.device)
    condition = storage[1].expand(3, 7)
    for value in (True, False):
        storage[1].fill_(value)
        if scalar_first:
            actual = flag_gems.where_scalar_self(condition, 1.25, inp)
            expected = torch.where(torch.tensor(value), 1.25, cpu)
        else:
            actual = flag_gems.where_scalar_other(condition, inp, 1.25)
            expected = torch.where(torch.tensor(value), cpu, 1.25)
        utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.where_self_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend output overlap")
def test_where_uniform_condition_overlapping_output_rejected():
    inp = torch.arange(21, device=flag_gems.device).reshape(3, 7).float()
    condition = torch.tensor(True, device=inp.device).expand(3, 7)
    backing = torch.full((1,), 123.0, device=inp.device)
    with pytest.raises(RuntimeError, match="internal overlapping"):
        flag_gems.where_self_out(condition, inp, inp, out=backing.expand(3, 7))
    utils.gems_assert_equal(backing.cpu(), torch.tensor([123.0]))


@pytest.mark.where_self
@pytest.mark.where_self_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend NPUGraph")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided_out", [False, True])
def test_where_uniform_condition_graph_reloads_inputs(dtype, strided_out):
    left_cpu = torch.arange(21).reshape(7, 3).t().to(dtype)
    right_cpu = -torch.arange(7).reshape(1, 7).to(dtype)
    left, right = left_cpu.to(flag_gems.device), right_cpu.to(flag_gems.device)
    storage = torch.tensor([False, True], device=left.device)
    condition = storage[1].reshape(1, 1).expand(3, 7)
    backing = torch.full((3, 14), -123.0, dtype=dtype, device=left.device)
    out = backing[:, 1::2]

    def invoke():
        if strided_out:
            return flag_gems.where_self_out(condition, left, right, out=out)
        return flag_gems.where_self(condition, left, right)

    invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = invoke()
    for step, value in enumerate((False, True, False), 1):
        current_left = left_cpu + step
        current_right = right_cpu - step
        storage[1].fill_(value)
        left.copy_(current_left)
        right.copy_(current_right)
        actual.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        expected = torch.where(torch.tensor(value), current_left, current_right)
        utils.gems_assert_equal(actual.cpu(), expected)
        if strided_out:
            utils.gems_assert_equal(
                backing[:, ::2].cpu(), torch.full((3, 7), -123.0, dtype=dtype)
            )


@pytest.mark.where_self
@pytest.mark.where_self_out
@pytest.mark.parametrize(
    "with_out",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name != "ascend",
                reason="Ascend out= broadcast error contract",
            ),
        ),
    ],
)
@pytest.mark.parametrize("uniform", [False, True])
def test_where_invalid_broadcast_raises_runtime_error(with_out, uniform):
    condition = torch.tensor([True, False], device=flag_gems.device)
    if uniform:
        condition = condition[:1].expand(2)
    left = torch.ones((2,), device=flag_gems.device)
    right = torch.ones((3,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        if with_out:
            out = torch.empty((3,), device=flag_gems.device)
            flag_gems.where_self_out(condition, left, right, out=out)
        else:
            flag_gems.where_self(condition, left, right)
