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
import torch.nn.functional as F

import flag_gems
from flag_gems.testing import RESOLUTION

from . import accuracy_utils as utils
from . import conftest as cfg


def _reference_input(x):
    """Input to build the reference from: CPU, float32.

    In the default (device) reference mode the reference is the backend's own
    implementation, and on Ascend ``gelu(approximate="none")`` is a coarse
    approximation -- measured against fp64 it is off by up to 1.6e-2 for
    bfloat16, 2.0e-3 for float16 and 4.7e-4 for float32, all of which exceed
    this file's 1e-4 atol, so the kernel would be reported as failing while
    being the more accurate of the two. CPU float32 gelu is exact to ~5e-7,
    which keeps the check meaningful in both reference modes.
    """
    return x.detach().cpu().to(torch.float32)


def _place_reference(ref, like):
    """Put a CPU reference where ``gems_assert_close`` expects it."""
    return ref if cfg.TO_CPU else ref.to(like.device)


@pytest.mark.gelu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if flag_gems.vendor_name == "ascend":
        ref_inp = res_inp.to(device="cpu", dtype=torch.float64)
    else:
        ref_inp = _reference_input(res_inp)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    res_out = flag_gems.gelu(res_inp, approximate=approximate)

    if ref_out.device.type == "cpu":
        res_out = res_out.cpu()

    atol = 1e-4
    if flag_gems.vendor_name == "aipu" and dtype == torch.float16:
        atol = 1e-3
    utils.gems_assert_close(
        res_out, _place_reference(ref_out, res_inp), dtype, atol=atol
    )


@pytest.mark.gelu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_inp)

    if flag_gems.vendor_name == "ascend":
        ref_inp = res_inp.to(device="cpu", dtype=torch.float64)
        ref_out = res_out.to(device="cpu", dtype=torch.float64)
    else:
        ref_inp = _reference_input(res_inp)
        ref_out = _reference_input(res_out)

    ref_in_grad = torch.ops.aten.gelu_backward(
        ref_out, ref_inp, approximate=approximate
    )
    res_in_grad = flag_gems.gelu_backward(res_out, res_inp, approximate=approximate)

    if ref_in_grad.device.type == "cpu":
        res_in_grad = res_in_grad.cpu()

    utils.gems_assert_close(res_in_grad, _place_reference(ref_in_grad, res_inp), dtype)


@pytest.mark.gelu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "ascend":
        ref_inp = res_inp.to(device="cpu", dtype=torch.float64)
    else:
        ref_inp = _reference_input(res_inp)

    ref_out = torch.ops.aten.gelu_.default(ref_inp, approximate=approximate)
    res_out = flag_gems.gelu_(res_inp, approximate=approximate)

    if ref_out.device.type == "cpu":
        res_out = res_out.cpu()

    utils.gems_assert_close(res_out, _place_reference(ref_out, res_inp), dtype)


# Ascend coreDim and large-tensor regressions.
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
APPROXIMATIONS = ["none", "tanh"]
OPERATIONS = ["forward", "backward", "inplace"]
TILE_SIZE = 512
CORE_DIM_LIMIT = 65535
ATOL = 1e-4


@pytest.fixture(scope="module")
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


def _run(x, dy, operation, approximate):
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


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
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


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
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


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
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


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "layout", ["transpose", "strided_offset", "strided_loop", "permuted_offset"]
)
def test_gelu_layout_and_storage(
    gelu_ops, launches, dtype, approximate, operation, layout, record_property
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
    actual = _run(x, dy, operation, approximate)
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


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_gelu_public_api_accuracy(
    launches, dtype, approximate, operation, record_property
):
    x_cpu, dy_cpu = _inputs((17, 257), dtype)
    expected = _reference(x_cpu, dy_cpu, operation, approximate)
    x, dy = x_cpu.to(flag_gems.device), dy_cpu.to(flag_gems.device)
    actual = _run(x, dy, operation, approximate)
    record_property("max_abs_error_fp64", _assert_accuracy(actual, expected, dtype))
    if operation == "inplace":
        _assert_alias(actual, x)
    assert len(launches) == 1


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("approximate", APPROXIMATIONS)
def test_gelu_forward_backward_against_autograd(
    launches, dtype, approximate, record_property
):
    x_cpu, dy_cpu = _inputs((4097,), dtype)
    reference = x_cpu.double().requires_grad_()
    expected_y = F.gelu(reference, approximate=approximate)
    expected_dx = torch.autograd.grad(expected_y, reference, dy_cpu.double())[0]
    x = x_cpu.to(flag_gems.device)
    actual_y = flag_gems.gelu(x, approximate=approximate)
    actual_dx = flag_gems.gelu_backward(dy_cpu.to(x.device), x, approximate=approximate)
    _assert_accuracy(actual_y, expected_y.detach(), dtype)
    record_property(
        "max_abs_error_fp64", _assert_accuracy(actual_dx, expected_dx, dtype)
    )
    assert len(launches) == 2
    assert any("backward" in call["kernel"] for call in launches)


@pytest.mark.usefixtures("gelu_ops")
@pytest.mark.gelu
@pytest.mark.gelu_backward
@pytest.mark.gelu_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend GELU")
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
