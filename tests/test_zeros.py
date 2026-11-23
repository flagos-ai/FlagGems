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

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.zeros
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test_zeros(shape, dtype):
    expected_dev = "cpu" if cfg.TO_CPU else device
    # without dtype
    res_out = flag_gems.zeros(shape, device=flag_gems.device)

    utils.gems_assert_equal(res_out, torch.zeros(shape, device=expected_dev))

    # with dtype
    res_out = flag_gems.zeros(shape, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device=expected_dev)
    )


# Ascend coreDim and large-tensor regressions.
BLOCK_SIZE = 20480
BLOCK_SIZE_SUB = 1024
CORE_DIM_LIMIT = 65535
GUARD = 37
DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.fixture
def zeros_ops():
    ops = importlib.import_module(flag_gems.zero_.__module__)
    assert ops.__name__ == flag_gems.zeros.__module__
    assert "_ascend" in ops.__name__
    return ops


@pytest.fixture
def checked_launches(zeros_ops, monkeypatch):
    kernel = zeros_ops.zeros_kernel
    launches = []

    class CheckedKernel:
        def __getitem__(self, grid):
            def checked_grid(meta):
                resolved = grid(meta) if callable(grid) else grid
                # Reject an invalid launch before it can poison the NPU context.
                assert 0 < math.prod(resolved) <= CORE_DIM_LIMIT
                launches.append((resolved, meta["BLOCK_SIZE"], meta["BLOCK_SIZE_SUB"]))
                return resolved

            return kernel[checked_grid]

    checked = CheckedKernel()
    monkeypatch.setattr(zeros_ops, "zeros_kernel", checked)
    return launches


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _check_inplace(base_cpu, view):
    expected_base = base_cpu.clone()
    expected_view = view(expected_base)
    expected_view.zero_()
    native_base = base_cpu.to(flag_gems.device)
    view(native_base).zero_()
    base = base_cpu.to(flag_gems.device)
    inp = view(base)
    pointer, offset, strides = inp.data_ptr(), inp.storage_offset(), inp.stride()
    result = flag_gems.zero_(inp)
    torch.npu.synchronize()
    assert result is inp
    assert result.data_ptr() == pointer
    assert result.storage_offset() == offset
    assert result.stride() == strides
    assert result.device == base.device
    _assert_exact(result, expected_view)
    # Check the entire backing storage, including guards and gaps in views.
    _assert_exact(base, expected_base)
    _assert_exact(base, native_base.cpu())
    if result.is_floating_point():
        assert not torch.signbit(result.cpu()).any().item()


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (2, 0, 3),
        (1,),
        (1023,),
        (1024,),
        (1025,),
        (20479,),
        (20480,),
        (20481,),
        (40977,),
        (3, 5, 1371),
    ],
)
def test_zero_inplace_boundaries_exact(checked_launches, dtype, shape):
    numel = math.prod(shape)
    base_cpu = torch.full(
        (numel + 2 * GUARD,), 1 if dtype == torch.bool else 7, dtype=dtype
    )
    view = lambda x: x[GUARD : GUARD + numel].view(shape)
    _check_inplace(base_cpu, view)
    assert checked_launches
    assert all(
        grid == (max(math.ceil(numel / BLOCK_SIZE), 1),)
        for grid, _, _ in checked_launches
    )
    assert all(
        (block, sub) == (BLOCK_SIZE, BLOCK_SIZE_SUB)
        for _, block, sub in checked_launches
    )


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_zero_overwrites_special_values_exact(checked_launches, dtype):
    info = torch.finfo(dtype)
    values = [
        float("nan"),
        float("inf"),
        -float("inf"),
        0.0,
        -0.0,
        info.max,
        -info.max,
        info.tiny,
        -info.tiny,
        info.eps,
        -info.eps,
    ]
    payload = torch.tensor(values, dtype=dtype).repeat(4000)[:40977]
    base_cpu = torch.full((payload.numel() + 2 * GUARD,), 7, dtype=dtype)
    base_cpu[GUARD:-GUARD] = payload
    _check_inplace(base_cpu, lambda x: x[GUARD:-GUARD])
    assert checked_launches


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize(
    "layout", ["transpose", "permute", "step", "column_slice", "diagonal"]
)
def test_zero_noncontiguous_storage_exact(checked_launches, dtype, layout):
    base_cpu = torch.full(
        (5 * 7 * 11 + 2 * GUARD,), 1 if dtype == torch.bool else 7, dtype=dtype
    )

    def view(x):
        tensor = x[GUARD:-GUARD].view(5, 7, 11)
        if layout == "transpose":
            return tensor.transpose(0, 2)
        if layout == "permute":
            return tensor.permute(1, 2, 0)
        if layout == "step":
            return tensor[1::2, :, 1::3]
        if layout == "column_slice":
            return tensor[:, :, 2:7]
        return tensor.diagonal(dim1=1, dim2=2)

    assert not view(base_cpu).is_contiguous()
    _check_inplace(base_cpu, view)
    assert checked_launches


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_repeated_calls_exact(checked_launches, dtype):
    inp = torch.full(
        (20481,),
        1 if dtype == torch.bool else 7,
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = torch.zeros(inp.shape, dtype=dtype)
    for _ in range(3):
        assert flag_gems.zero_(inp) is inp
        torch.npu.synchronize()
        _assert_exact(inp, expected)
    assert len(checked_launches) == 3


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", [None] + DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (20481,)])
def test_zeros_factory_poisoned_allocation_exact(
    zeros_ops, checked_launches, monkeypatch, dtype, shape
):
    expected = torch.zeros(shape, dtype=dtype)
    native = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    original_empty = torch.empty
    allocations = []

    def poisoned_empty(*args, **kwargs):
        out = original_empty(*args, **kwargs)
        out.fill_(1 if out.dtype == torch.bool else 7)
        allocations.append(out)
        return out

    # Nonzero allocation contents ensure an unwritten output cannot pass.
    with monkeypatch.context() as patch:
        patch.setattr(zeros_ops.torch, "empty", poisoned_empty)
        result = flag_gems.zeros(list(shape), dtype=dtype, device=flag_gems.device)
    torch.npu.synchronize()
    assert allocations
    assert any(result.data_ptr() == allocation.data_ptr() for allocation in allocations)
    assert result.device == native.device
    _assert_exact(result, expected)
    _assert_exact(result, native.cpu())
    assert checked_launches


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("op", ["zero_", "zeros"])
@pytest.mark.parametrize("logical_blocks", [65534, 65535, 65536, 131072])
def test_zero_launch_core_dim_limit(zeros_ops, monkeypatch, op, logical_blocks):
    # Host-only launch planning check: no multi-GiB allocation or invalid NPU
    # launch is required to expose the exact 65536/131072 boundary failure.
    numel = logical_blocks * BLOCK_SIZE

    class MetadataOnlyTensor:
        device = torch.device(flag_gems.device)

        def numel(self):
            return numel

        def is_contiguous(self):
            return True

    tensor = MetadataOnlyTensor()
    launches = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def capture(out, size, **meta):
                assert out is tensor and size == numel
                launches.append(grid(meta) if callable(grid) else grid)

            return capture

    monkeypatch.setattr(zeros_ops, "zeros_kernel", CaptureKernel())
    if op == "zero_":
        result = zeros_ops.zero_(tensor)
    else:
        monkeypatch.setattr(zeros_ops.torch, "empty", lambda *args, **kwargs: tensor)
        result = zeros_ops.zeros((numel,), dtype=torch.bool, device=tensor.device)
    assert result is tensor
    assert launches
    for grid in launches:
        assert (
            0 < math.prod(grid) <= CORE_DIM_LIMIT
        ), f"{op}: numel={numel}, grid={grid}, coreDim must be <= {CORE_DIM_LIMIT}"


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize(
    "length,dtype",
    [
        (65536 * 1024 + 1, torch.float32),
        (131072 * 1024 + 17, torch.float16),
        (65535 * BLOCK_SIZE, torch.bool),
        (65535 * BLOCK_SIZE + 1, torch.bool),
        (131072 * BLOCK_SIZE + 17, torch.bool),
    ],
)
def test_zero_real_large_input_exact(checked_launches, length, dtype):
    base = torch.full(
        (length + 2 * GUARD,),
        1 if dtype == torch.bool else 7,
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = base[GUARD:-GUARD]
    result = flag_gems.zero_(inp)
    torch.npu.synchronize()
    assert result is inp
    # Compare every output element, with bounded host-memory use.
    for start in range(0, length, 2**24):
        actual = result[start : start + 2**24].cpu()
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    guards = torch.cat((base[:GUARD].cpu(), base[-GUARD:].cpu()))
    _assert_exact(guards, torch.full_like(guards, 1 if dtype == torch.bool else 7))
    assert checked_launches


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "step", "diagonal"])
def test_zero_capped_grid_covers_all_tiles_exact(
    zeros_ops, checked_launches, monkeypatch, dtype, layout
):
    # Force several rounds on small tensors; checking the whole storage catches
    # both missing tiles and writes into gaps/guards after wrapping the grid.
    monkeypatch.setattr(zeros_ops, "_MAX_CORE_DIM", 2)
    length = 12345 * 7 * 11 if layout == "diagonal" else 9 * BLOCK_SIZE + 257
    base_cpu = torch.full(
        (length + 2 * GUARD,), 1 if dtype == torch.bool else 7, dtype=dtype
    )

    def view(x):
        payload = x[GUARD:-GUARD]
        if layout == "step":
            return payload[1::2]
        if layout == "diagonal":
            return payload.view(12345, 7, 11).diagonal(dim1=1, dim2=2)
        return payload

    _check_inplace(base_cpu, view)
    assert checked_launches
    assert all(grid == (2,) for grid, _, _ in checked_launches)


@pytest.mark.zeros
@pytest.mark.zero_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend zero kernel")
def test_zeros_factory_capped_grid_exact(zeros_ops, checked_launches, monkeypatch):
    monkeypatch.setattr(zeros_ops, "_MAX_CORE_DIM", 2)
    shape = (9 * 32768 + 17,)
    original_allocate = torch.empty

    def poisoned_allocate(*args, **kwargs):
        out = original_allocate(*args, **kwargs)
        out.fill_(7)
        return out

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", poisoned_allocate)
        result = flag_gems.zeros(shape, device=flag_gems.device, dtype=torch.float32)
    torch.npu.synchronize()
    _assert_exact(result, torch.zeros(shape, dtype=torch.float32))
    assert checked_launches
    assert all(grid == (2,) for grid, _, _ in checked_launches)
