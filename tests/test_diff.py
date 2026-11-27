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

from .accuracy_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

DIFF_SHAPES = [(1024,), (100, 200), (10, 20, 30)]
# DIFF_SHAPES = [(1024,), (100, 200), (10, 20, 30), (16, 128, 64, 60)]  # original, commented out to reduce CI timeout
DIFF_3D_SHAPES = [(8, 16, 32)]
# DIFF_3D_SHAPES = [(8, 16, 32), (32, 64, 100)]  # original, commented out to reduce CI timeout

if QUICK_MODE:
    DIFF_SHAPES = DIFF_SHAPES[:1]
    DIFF_3D_SHAPES = DIFF_3D_SHAPES[:1]


def _make_input(shape, dtype, device):
    if dtype in INT_DTYPES:
        if flag_gems.vendor_name == "cambricon":
            # Cambricon torch.randint currently does not support int8/int16 generation.
            return torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(device)
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize(
    "n", [1, 2]
)  # original: [1, 2, 5], n=5 commented out to reduce CI timeout
def test_diff(shape, dtype, n):
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, n=n)
    res_out = flag_gems.diff(inp, n=n)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_3D_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_diff_3d(shape, dim, dtype):
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, dim=dim)
    res_out = flag_gems.diff(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_diff_prepend_append(shape, dtype):
    inp = _make_input(shape, dtype, flag_gems.device)
    prepend = _make_input(
        shape[:-1] + (3,) if len(shape) > 1 else (3,),
        dtype,
        flag_gems.device,
    )
    append = _make_input(
        shape[:-1] + (2,) if len(shape) > 1 else (2,),
        dtype,
        flag_gems.device,
    )
    ref_inp = to_reference(inp)
    ref_prepend = to_reference(prepend)
    ref_append = to_reference(append)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend, append=ref_append)
    res_out = flag_gems.diff(inp, prepend=prepend, append=append)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend diff slow-path regression"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "length,n", [(4099, 1), (4099, 2), (20, 3), (22, 5), (1027, 3)]
)
def test_diff_slow_path_bounds(dtype, length, n):
    # Include the UB limit, higher orders, and output-tile boundaries.
    cpu = (
        ((torch.arange(2 * length * 3) % 17).float() / 8)
        .to(dtype)
        .reshape(2, length, 3)
    )
    expected = torch.diff(cpu, n=n, dim=1)
    actual = flag_gems.diff(cpu.to(flag_gems.device), n=n, dim=1)
    gems_assert_equal(actual.cpu(), expected)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend diff input-tail regression"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,n",
    [
        ((16,), 1),
        ((16,), 2),
        ((18,), 1),
        ((19,), 2),
        ((17, 2), 1),
        ((17, 3), 2),
        ((3, 17), 2),
        ((256, 256), 1),
    ],
)
def test_diff_fast_path_tail(shape, n, dtype):
    size = 1
    for length in shape:
        size *= length
    cpu = ((torch.arange(size) % 17).float() / 8).to(dtype).reshape(shape)
    expected = torch.diff(cpu, n=n)
    actual = flag_gems.diff(cpu.to(flag_gems.device), n=n)
    gems_assert_equal(actual.cpu(), expected)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend diff strided input"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n,dim", [(1, 1), (3, 0)])
def test_diff_strided_input(dtype, n, dim):
    cpu = ((torch.arange(32).float() % 17) / 8).to(dtype).reshape(4, 8).t()
    inp = cpu.to(flag_gems.device)
    assert not inp.is_contiguous()
    expected = torch.diff(cpu, n=n, dim=dim)
    actual = flag_gems.diff(inp, n=n, dim=dim)
    gems_assert_equal(actual.cpu(), expected)


# Exact rounding witnesses cover all n2 implementations deterministically.
@pytest.mark.diff
@pytest.mark.parametrize(
    "dtype,small", [(torch.float16, 1 / 4096), (torch.bfloat16, 1 / 512)]
)
@pytest.mark.parametrize("path", ["flat", "row", "inner"])
def test_diff_second_order_rounding(dtype, small, path):
    values = torch.tensor([1.0, small, 0.0], dtype=dtype).repeat(342)
    kwargs = dict(n=2)
    if path == "flat":
        cpu = values[:1024]
    elif path == "row":
        cpu = values[:51].reshape(3, 17)
    else:
        cpu = values[:102].reshape(2, 17, 3)
        kwargs["dim"] = 1
    expected = torch.diff(cpu, **kwargs)
    device_kwargs = {
        k: v.to(flag_gems.device) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }
    actual = flag_gems.diff(cpu.to(flag_gems.device), **device_kwargs)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend diff boundary regression"
)
@pytest.mark.parametrize(
    "shape,n", [((0, 7), 1), ((3, 0), 1), ((3, 2), 4), ((3, 2), 2)]
)
def test_diff_empty_boundaries(shape, n):
    cpu = torch.zeros(shape)
    actual = flag_gems.diff(cpu.to(flag_gems.device), n=n)
    gems_assert_equal(actual.cpu(), torch.diff(cpu, n=n))


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend zero-order copy semantics"
)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "empty"])
def test_diff_zero_order_copy(layout):
    cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    if layout == "transpose":
        cpu = cpu.t()
    elif layout == "slice":
        cpu = cpu[1:, 1::2]
    elif layout == "empty":
        cpu = cpu[:0]
    inp = cpu.to(flag_gems.device)
    actual = flag_gems.diff(inp, n=0, prepend=inp, append=inp)
    gems_assert_equal(actual.cpu(), torch.diff(cpu, n=0, prepend=cpu, append=cpu))
    assert actual.stride() == torch.diff(inp, n=0).stride()
    assert not torch._C._is_alias_of(actual, inp)
    actual.fill_(123)
    gems_assert_equal(inp.cpu(), cpu)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend large flat second-order kernel"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_diff_large_second_order(dtype):
    cpu = ((torch.arange(65536) % 17).float() / 8).to(dtype).reshape(64, 1024)
    actual = flag_gems.diff(cpu.to(flag_gems.device), n=2)
    gems_assert_equal(actual.cpu(), torch.diff(cpu, n=2))


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Requires Ascend NPUGraph capture and replay",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,dim", [(0, 1), (2, -1), (3, 1), (5, 1)])
def test_diff_graph_replay(dtype, n, dim):
    cpu = ((torch.arange(2 * 17 * 3) % 17).float() / 8).to(dtype).reshape(2, 17, 3)
    inp = cpu.to(flag_gems.device)
    flag_gems.diff(inp, n=n, dim=dim)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = flag_gems.diff(inp, n=n, dim=dim)
    for updated in (cpu.flip(1), -cpu):
        inp.copy_(updated)
        actual.fill_(-123)
        graph.replay()
        torch.npu.synchronize()
        gems_assert_equal(actual.cpu(), torch.diff(updated, n=n, dim=dim))
        gems_assert_equal(inp.cpu(), updated)


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend logical launch-grid limit"
)
def test_diff_launch_boundaries():
    cpu = torch.arange(17, dtype=torch.float32).pow(3).reshape(1, 17, 1)
    cpu = cpu.expand(1, 17, 65536).contiguous()
    actual = flag_gems.diff(cpu.to(flag_gems.device), n=3, dim=1)
    gems_assert_equal(actual.cpu(), torch.diff(cpu, n=3, dim=1))
