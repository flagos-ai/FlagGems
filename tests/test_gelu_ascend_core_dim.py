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

"""Ascend GELU accuracy, layout safety, dispatch and bounded launch regression.

Run from the repository root with the Ascend environment loaded::

    python -m pytest tests/test_gelu_ascend_core_dim.py --ref=cpu -q

References always use CPU float64 on the quantized input and gradient.
Tolerances match the existing GELU suite. Large cases really execute more than
65535 logical tiles; they use float16 and compare EVERY output in chunks, with
roughly 1.5 GiB of live NPU tensors per backward case. All other accuracy
families cover float16, bfloat16 and float32. No known failure is xfailed.
"""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.testing import RESOLUTION

pytestmark = [
    pytest.mark.gelu,
    pytest.mark.gelu_backward,
    pytest.mark.gelu_,
    pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU"),
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
APPROXIMATIONS = ["none", "tanh"]
OPERATIONS = ["forward", "backward", "inplace"]
TILE_SIZE = 512
CORE_DIM_LIMIT = 65535
ATOL = 1e-4


@pytest.fixture(scope="module", autouse=True)
def gelu_ops():
    # Import the actual dispatch module; importing its fully qualified alias
    # can create a second module object which would miss public API launches.
    ops = importlib.import_module(flag_gems.gelu.__module__)
    assert "_ascend.ops.gelu" in ops.__name__
    for name in ("gelu", "gelu_", "gelu_backward"):
        assert getattr(flag_gems, name) is getattr(ops, name)
    return ops


@pytest.fixture
def launches(gelu_ops, monkeypatch):
    """Observe generated wrappers and delegate every launch to the real NPU."""
    calls = []
    patched = set()

    class CheckedKernel:
        def __init__(self, entry, name):
            self.entry = entry
            self.name = name

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                resolved = grid(kwargs) if callable(grid) else grid
                core_dim = math.prod(resolved)
                # Fail before submitting an invalid grid to the device.
                assert 0 < core_dim <= min(gelu_ops.CORE_NUM, CORE_DIM_LIMIT)
                tile = math.prod(
                    value
                    for key, value in kwargs.items()
                    if key.startswith("tile_size")
                )
                assert tile <= TILE_SIZE
                calls.append(
                    {
                        "kernel": self.name,
                        "grid": tuple(resolved),
                        "tile": tile,
                        "tiles_per_cta": kwargs.get("tiles_per_cta", 1),
                        "one_tile_per_cta": kwargs.get("one_tile_per_cta", True),
                    }
                )
                return self.entry[grid](*args, **kwargs)

            return run

    def observe(pointwise):
        instantiate = pointwise.instantiate

        def checked_instantiate(ndim):
            wrapper = instantiate(ndim)
            info = pointwise.get_kernel_info(ndim)
            namespace = wrapper.__globals__
            key = (id(namespace), info.kernel_name)
            if key not in patched:
                entry = namespace[info.kernel_name]
                monkeypatch.setitem(
                    namespace, info.kernel_name, CheckedKernel(entry, info.kernel_name)
                )
                patched.add(key)
            return wrapper

        monkeypatch.setattr(pointwise, "instantiate", checked_instantiate)

    for name in (
        "gelu_none",
        "gelu_tanh",
        "gelu_backward_none",
        "gelu_backward_tanh",
    ):
        observe(getattr(gelu_ops, name))
    return calls


def _inputs(shape, dtype):
    generator = torch.Generator(device="cpu").manual_seed(20261021)
    x = (torch.randn(shape, generator=generator) * 3).to(dtype)
    dy = torch.randn(shape, generator=generator).to(dtype)
    return x, dy


def _reference(x, dy, operation, approximate):
    if operation == "backward":
        return torch.ops.aten.gelu_backward(
            dy.double(), x.double(), approximate=approximate
        )
    return F.gelu(x.double(), approximate=approximate)


def _run(x, dy, operation, approximate, api="direct"):
    if api == "torch":
        with flag_gems.use_gems(include=["gelu", "gelu_", "gelu_backward"]):
            if operation == "backward":
                return torch.ops.aten.gelu_backward(dy, x, approximate=approximate)
            if operation == "inplace":
                return torch.ops.aten.gelu_.default(x, approximate=approximate)
            return F.gelu(x, approximate=approximate)
    if operation == "backward":
        return flag_gems.gelu_backward(dy, x, approximate=approximate)
    if operation == "inplace":
        return flag_gems.gelu_(x, approximate=approximate)
    return flag_gems.gelu(x, approximate=approximate)


def _assert_accuracy(actual, expected, dtype):
    assert actual.device.type == "npu"
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    result = actual.detach().cpu()
    torch.testing.assert_close(
        result, expected.to(dtype), rtol=RESOLUTION[dtype], atol=ATOL
    )
    if not result.numel():
        return 0.0
    # Report the unrounded float64-reference error, including output rounding.
    return (result.double() - expected).abs().max().item()


def _assert_alias(actual, original):
    assert actual.data_ptr() == original.data_ptr()
    assert actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert actual.storage_offset() == original.storage_offset()
    assert actual.stride() == original.stride()


@pytest.mark.parametrize(
    "kernel",
    ["gelu_none", "gelu_tanh", "gelu_backward_none", "gelu_backward_tanh"],
)
def test_gelu_ascend_config(gelu_ops, kernel):
    config = getattr(gelu_ops, kernel).config
    assert config.max_tile_size == TILE_SIZE
    assert config.max_grid_size == (gelu_ops.CORE_NUM, 1, 1)
    assert 0 < math.prod(config.max_grid_size) <= CORE_DIM_LIMIT
    assert config.prefer_block_pointer is False
    assert config.prefer_1d_tile is True


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "shape_case",
    [
        (),
        (0,),
        (2, 0, 3),
        (1,),
        (17,),
        (511,),
        (512,),
        (513,),
        (4095,),
        (4096,),
        (4097,),
        (7, 17, 65),
        "core_minus_one",
        "core_exact",
        "core_plus_one",
        "multiple_rounds",
    ],
)
def test_gelu_tile_boundaries(
    gelu_ops, launches, dtype, approximate, operation, shape_case, record_property
):
    core_elements = gelu_ops.CORE_NUM * TILE_SIZE
    dynamic_shapes = {
        "core_minus_one": (core_elements - 1,),
        "core_exact": (core_elements,),
        "core_plus_one": (core_elements + 1,),
        "multiple_rounds": (2 * core_elements + 37,),
    }
    shape = dynamic_shapes.get(shape_case, shape_case)
    x_cpu, dy_cpu = _inputs(shape, dtype)
    expected = _reference(x_cpu, dy_cpu, operation, approximate)
    x, dy = x_cpu.to(flag_gems.device), dy_cpu.to(flag_gems.device)
    actual = _run(x, dy, operation, approximate)
    error = _assert_accuracy(actual, expected, dtype)
    record_property("max_abs_error_fp64", error)
    if operation == "inplace":
        _assert_alias(actual, x)
    else:
        torch.testing.assert_close(x.cpu(), x_cpu, rtol=0, atol=0)
    torch.testing.assert_close(dy.cpu(), dy_cpu, rtol=0, atol=0)
    assert len(launches) == (1 if x.numel() else 0)
    if shape_case in ("core_plus_one", "multiple_rounds"):
        assert launches[0]["tiles_per_cta"] > 1
        assert launches[0]["one_tile_per_cta"] is False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_gelu_finite_edge_accuracy(
    launches, dtype, approximate, operation, record_property
):
    x_cpu = torch.tensor(
        [
            -100,
            -20,
            -10,
            -6,
            -3,
            -1,
            -0.75,
            -0.1,
            -1e-7,
            -0.0,
            0.0,
            1e-7,
            0.1,
            0.75,
            1,
            3,
            6,
            10,
            20,
            100,
        ],
        dtype=dtype,
    )
    dy_cpu = torch.linspace(-2, 2, x_cpu.numel()).to(dtype)
    dy_cpu[3] = 0
    expected = _reference(x_cpu, dy_cpu, operation, approximate)
    actual = _run(
        x_cpu.to(flag_gems.device),
        dy_cpu.to(flag_gems.device),
        operation,
        approximate,
    )
    record_property("max_abs_error_fp64", _assert_accuracy(actual, expected, dtype))
    assert len(launches) == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("api", ["direct", "torch"])
@pytest.mark.parametrize(
    "layout", ["transpose", "strided_offset", "strided_loop", "permuted_offset"]
)
def test_gelu_layout_and_storage(
    gelu_ops, launches, dtype, approximate, operation, layout, api, record_property
):
    shapes = {
        "transpose": (37, 53),
        "strided_offset": (15, 35, 61),
        "strided_loop": (15, 35, 455),
        "permuted_offset": (5, 17, 11, 73),
    }
    shape = shapes[layout]
    base_cpu, _ = _inputs(shape, dtype)
    base = base_cpu.to(flag_gems.device)
    mask = torch.zeros(shape, dtype=torch.bool)
    if layout == "transpose":
        x, x_cpu = base.t(), base_cpu.t()
        mask[:] = True
    elif layout in ("strided_offset", "strided_loop"):
        index = (slice(1, 14), slice(1, 35, 2), slice(2, shape[2] - 2, 3))
        x, x_cpu = base[index], base_cpu[index]
        mask[index] = True
        assert x.storage_offset() > 0
    else:
        index = (slice(1, 5), slice(1, 17, 2), slice(2, 11, 3), slice(1, 73, 2))
        x = base[index].permute(2, 0, 3, 1)
        x_cpu = base_cpu[index].permute(2, 0, 3, 1)
        mask[index] = True
        assert x.storage_offset() > 0
    assert not x.is_contiguous()
    # Exercise a strided gradient as well as a strided activation. Retain the
    # whole gradient storage to check that backward does not modify its guards.
    _, dy_base_cpu = _inputs((*x.shape, 2), dtype)
    dy_base = dy_base_cpu.to(flag_gems.device)
    dy, dy_cpu = dy_base[..., 1], dy_base_cpu[..., 1]
    expected = _reference(x_cpu, dy_cpu, operation, approximate)
    actual = _run(x, dy, operation, approximate, api=api)
    record_property("max_abs_error_fp64", _assert_accuracy(actual, expected, dtype))
    if operation == "inplace":
        _assert_alias(actual, x)
        torch.testing.assert_close(base.cpu()[~mask], base_cpu[~mask], rtol=0, atol=0)
    else:
        torch.testing.assert_close(base.cpu(), base_cpu, rtol=0, atol=0)
    torch.testing.assert_close(dy_base.cpu(), dy_base_cpu, rtol=0, atol=0)
    assert len(launches) == 1
    if layout == "strided_loop":
        assert x.numel() > gelu_ops.CORE_NUM * TILE_SIZE
        assert launches[0]["tiles_per_cta"] > 1
        assert launches[0]["one_tile_per_cta"] is False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_gelu_torch_dispatch(launches, dtype, approximate, operation, record_property):
    x_cpu, dy_cpu = _inputs((17, 257), dtype)
    expected = _reference(x_cpu, dy_cpu, operation, approximate)
    x, dy = x_cpu.to(flag_gems.device), dy_cpu.to(flag_gems.device)
    actual = _run(x, dy, operation, approximate, api="torch")
    record_property("max_abs_error_fp64", _assert_accuracy(actual, expected, dtype))
    if operation == "inplace":
        _assert_alias(actual, x)
    assert len(launches) == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
def test_gelu_autograd_accuracy(launches, dtype, approximate, record_property):
    x_cpu, dy_cpu = _inputs((4097,), dtype)
    reference = x_cpu.double().requires_grad_()
    expected_y = F.gelu(reference, approximate=approximate)
    expected_dx = torch.autograd.grad(expected_y, reference, dy_cpu.double())[0]
    x = x_cpu.to(flag_gems.device).requires_grad_()
    with flag_gems.use_gems(include=["gelu", "gelu_backward"]):
        actual_y = F.gelu(x, approximate=approximate)
        actual_dx = torch.autograd.grad(actual_y, x, dy_cpu.to(x.device))[0]
    _assert_accuracy(actual_y, expected_y.detach(), dtype)
    record_property(
        "max_abs_error_fp64", _assert_accuracy(actual_dx, expected_dx, dtype)
    )
    assert len(launches) == 2
    assert any("backward" in call["kernel"] for call in launches)


@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_gelu_more_than_65535_tiles(
    gelu_ops, launches, approximate, operation, record_property
):
    dtype = torch.float16
    # Retain the original regression's input size after reducing the UB tile.
    # It now requires 524289 logical tiles with a 512-element compute tile.
    period = 4096
    size = (CORE_DIM_LIMIT + 1) * period + 17
    x_cpu = torch.linspace(-8, 8, period).to(dtype)
    dy_cpu = torch.linspace(-2, 2, period).roll(937).to(dtype)
    expected = _reference(x_cpu, dy_cpu, operation, approximate)

    def repeated_npu(pattern):
        tensor = torch.empty(size, dtype=dtype, device=flag_gems.device)
        full = (size // period) * period
        tensor[:full].view(-1, period).copy_(pattern.to(tensor.device))
        tensor[full:].copy_(pattern[: size - full].to(tensor.device))
        return tensor

    x = repeated_npu(x_cpu)
    dy = repeated_npu(dy_cpu) if operation == "backward" else None
    actual = _run(x, dy, operation, approximate)
    assert len(launches) == 1
    call = launches[0]
    assert math.ceil(size / call["tile"]) > CORE_DIM_LIMIT
    assert call["grid"] == (gelu_ops.CORE_NUM, 1, 1)
    assert call["tiles_per_cta"] > 1
    assert call["one_tile_per_cta"] is False
    if operation == "inplace":
        _assert_alias(actual, x)
    chunk_size = period * 256
    expected_chunk = expected.repeat(256)
    error = 0.0
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        error = max(
            error,
            _assert_accuracy(actual[start:stop], expected_chunk[: stop - start], dtype),
        )
    record_property("max_abs_error_fp64", error)
    record_property("checked_elements", size)
    record_property("actual_core_dim", math.prod(call["grid"]))
    record_property("logical_tiles", math.ceil(size / call["tile"]))
