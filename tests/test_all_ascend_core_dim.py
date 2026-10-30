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

"""Exact Ascend all regressions for bounded grids, tiles, and strided inputs.

Run from the repository root with the Ascend environment loaded::

    ASCEND_LAUNCH_BLOCKING=1 python -m pytest \
        tests/test_all_ascend_core_dim.py --ref=cpu -q

Accuracy requires exact Boolean equality (rtol=atol=0) against CPU PyTorch.
The real coreDim cases allocate up to approximately 1 GiB per input copy.
Int64 axis-reduction accuracy uses explicit small tiles because the existing
default autotune set contains a configuration exceeding Ascend's UB capacity.
The public-entry failure is retained in ascend_all_int64_ub_repro.py and can be
run explicitly; global int64 reductions use the unmodified production tiles.
"""

import importlib
import math

import pytest
import torch
import triton

import flag_gems

pytestmark = [
    pytest.mark.all,
    pytest.mark.all_dim,
    pytest.mark.all_dims,
    pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend reduction limits"),
]

DTYPES = [
    torch.bool,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
]
FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def all_ops():
    # Use the registered backend module, including its actual import alias.
    return importlib.import_module(flag_gems.all.__module__)


@pytest.fixture
def checked_launches(all_ops, monkeypatch):
    launches = []

    class CheckedKernel:
        def __init__(self, name, kernel):
            self.name = name
            self.kernel = kernel

        def __getitem__(self, grid):
            def checked_grid(meta):
                resolved = grid(meta) if callable(grid) else grid
                assert 0 < math.prod(resolved) <= all_ops._ASCEND_MAX_CORE_DIM
                tiles = {
                    name: meta[name]
                    for name in (
                        "BLOCK_SIZE",
                        "BLOCK_MID",
                        "BLOCK_M",
                        "BLOCK_N",
                    )
                    if name in meta
                }
                launches.append((self.name, resolved, tiles))
                return resolved

            return self.kernel[checked_grid]

    for name in ("all_kernel_1", "all_kernel_2", "all_kernel_dim"):
        monkeypatch.setattr(all_ops, name, CheckedKernel(name, getattr(all_ops, name)))
    return launches


def _assert_exact(result, expected, native):
    torch.npu.synchronize()
    assert result.dtype == expected.dtype == torch.bool
    assert result.shape == expected.shape
    assert result.device == native.device
    # all is a logical reduction: any differing bit is an accuracy failure.
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(result, native, rtol=0, atol=0)


def _check_all(
    inp_cpu,
    inp,
    *,
    dim=None,
    keepdim=False,
    global_reduce=True,
    native_nonzero=False,
):
    kwargs = {} if global_reduce else {"dim": dim, "keepdim": keepdim}
    expected = torch.all(inp_cpu, **kwargs)
    # torch_npu 2.7.1.post8 returns True for all([NaN, 0]). For NaN cases,
    # use its boolean reduction as the second oracle, keeping the unmodified
    # CPU input as the primary oracle. Both comparisons remain exact.
    native = torch.all(inp != 0 if native_nonzero else inp, **kwargs)
    with flag_gems.use_gems(include=["all", "all_dim", "all_dims"]):
        result = torch.all(inp, **kwargs)
    _assert_exact(result, expected, native)


@pytest.mark.parametrize(
    "helper,value,expected",
    [
        ("_reduction_block_size", 1, 1),
        ("_reduction_block_size", 17, 8),
        ("_reduction_block_size", 4096**2, 4096),
        ("_reduction_block_size", 4096**2 + 1, 4096),
        ("_reduction_block_size", 2**28 + 97, 4096),
        ("_reduction_mid_block_size", 1, 1),
        ("_reduction_mid_block_size", 4095, 4096),
        ("_reduction_mid_block_size", 4096, 4096),
        ("_reduction_mid_block_size", 4097, 4096),
        ("_reduction_mid_block_size", 65537, 4096),
        ("_reduction_grid_size", 1, 1),
        ("_reduction_grid_size", 65534, 65534),
        ("_reduction_grid_size", 65535, 65535),
        ("_reduction_grid_size", 65536, 65535),
        ("_reduction_grid_size", 131072, 65535),
    ],
)
def test_all_reduction_limits(all_ops, helper, value, expected):
    assert getattr(all_ops, helper)(value) == expected


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (17,)])
@pytest.mark.parametrize("has_zero", [False, True])
def test_all_scalar_and_tail_accuracy(checked_launches, dtype, shape, has_zero):
    inp_cpu = torch.ones(shape, dtype=dtype)
    if has_zero:
        inp_cpu.reshape(-1)[-1] = 0
    _check_all(inp_cpu, inp_cpu.to(flag_gems.device))
    assert {name for name, _, _ in checked_launches} == {
        "all_kernel_1",
        "all_kernel_2",
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grid_limit", [1, 3])
@pytest.mark.parametrize("zero_index", [None, 0, 127, 128, 384, 16384, 65536])
def test_all_grid_stride_accuracy(
    all_ops, monkeypatch, checked_launches, dtype, grid_limit, zero_index
):
    # 513 logical tiles force both stage-one strides and stage-two iterations.
    monkeypatch.setattr(all_ops, "_ASCEND_MAX_CORE_DIM", grid_limit)
    monkeypatch.setattr(all_ops, "_ASCEND_ALL_MAX_BLOCK_SIZE", 128)
    inp_cpu = torch.ones(65537, dtype=dtype)
    if dtype != torch.bool:
        inp_cpu[1::2] = -1
    if zero_index is not None:
        inp_cpu[zero_index] = 0
    _check_all(inp_cpu, inp_cpu.to(flag_gems.device))
    assert checked_launches
    for name, grid, tiles in checked_launches:
        if name == "all_kernel_1":
            assert math.prod(grid) == grid_limit
            assert tiles["BLOCK_SIZE"] == 128
        else:
            assert name == "all_kernel_2"
            assert math.prod(grid) == 1
            assert tiles["BLOCK_MID"] == 128


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("workers", [1, 3])
def test_all_stage_one_every_tile_accuracy(all_ops, dtype, workers):
    block = 128
    length = 7 * block + 17
    inp_cpu = torch.ones(length, dtype=dtype)
    inp_cpu[[0, block - 1, block, 3 * block + 7, length - 1]] = 0
    expected = torch.stack([torch.all(x) for x in inp_cpu.split(block)])
    inp = inp_cpu.to(flag_gems.device)
    # Opposite initial bits make unwritten intermediate tiles fail reliably.
    mid = (~expected).to(flag_gems.device)
    all_ops.all_kernel_1[(workers,)](inp, mid, length, mid.numel(), block)
    native = torch.stack([torch.all(x) for x in inp.split(block)])
    _assert_exact(mid, expected, native)


@pytest.mark.parametrize("mid_size", [1, 127, 128, 129, 257, 513])
@pytest.mark.parametrize("zero_location", ["none", "first", "last"])
def test_all_stage_two_multitile_accuracy(all_ops, mid_size, zero_location):
    mid_cpu = torch.ones(mid_size, dtype=torch.bool)
    if zero_location != "none":
        mid_cpu[0 if zero_location == "first" else -1] = False
    expected = torch.all(mid_cpu)
    mid = mid_cpu.to(flag_gems.device)
    out = (~expected).to(flag_gems.device)
    all_ops.all_kernel_2[(1,)](mid, out, mid_size, 128)
    _assert_exact(out, expected, torch.all(mid))


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("length", [65535 * 4096, 65535 * 4096 + 1, 2**28 + 97])
def test_all_real_core_dim_and_ub_limits(checked_launches, dtype, length):
    # Actual production caps, not monkeypatched: up to 1 GiB per input copy.
    inp_cpu = torch.ones(length, dtype=dtype)
    inp = inp_cpu.to(flag_gems.device)
    _check_all(inp_cpu, inp)
    for index in (0, length - 1):
        inp_cpu[index] = 0
        inp[index] = 0
        _check_all(inp_cpu, inp)
        inp_cpu[index] = 1
        inp[index] = 1
    assert checked_launches
    for name, grid, tiles in checked_launches:
        if name == "all_kernel_1":
            assert math.prod(grid) == 65535
            assert tiles["BLOCK_SIZE"] == 4096
        else:
            assert name == "all_kernel_2"
            assert math.prod(grid) == 1
            assert tiles["BLOCK_MID"] == 4096


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "slice_offset", "expand"])
@pytest.mark.parametrize("has_zero", [False, True])
def test_all_noncontiguous_accuracy(checked_launches, dtype, layout, has_zero):
    if layout == "transpose":
        base_cpu = torch.ones((17, 9), dtype=dtype)
        view = lambda x: x.t()
    elif layout == "slice_offset":
        # Zeros in storage gaps must not be included in the reduction.
        base_cpu = torch.zeros((19, 13), dtype=dtype)
        view = lambda x: x[1::2, 2::3]
        view(base_cpu).fill_(1)
    else:
        base_cpu = torch.ones((1, 7), dtype=dtype)
        view = lambda x: x.expand(19, 7)
    inp_cpu = view(base_cpu)
    if has_zero:
        inp_cpu[-1, -1] = 0
    # Transfer the base before applying the view to preserve NPU strides.
    inp = view(base_cpu.to(flag_gems.device))
    assert not inp_cpu.is_contiguous()
    assert not inp.is_contiguous()
    if layout == "slice_offset":
        assert inp.storage_offset() > 0
    _check_all(inp_cpu, inp)
    assert checked_launches


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_all_empty_identity_without_launch(checked_launches, dtype, shape):
    inp_cpu = torch.empty(shape, dtype=dtype)
    inp = inp_cpu.to(flag_gems.device)
    _check_all(inp_cpu, inp)
    _check_all(inp_cpu, inp, dim=None, keepdim=True, global_reduce=False)
    assert not checked_launches


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "special",
    [
        pytest.param(None, id="finite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(-float("inf"), id="negative-inf"),
    ],
)
@pytest.mark.parametrize("zero", [None, 0.0, -0.0])
def test_all_floating_truth_accuracy(checked_launches, dtype, special, zero):
    info = torch.finfo(dtype)
    values = [info.tiny, -info.tiny, info.eps, -info.eps, 1.0, -1.0]
    if special is not None:
        values.append(special)
    if zero is not None:
        values.append(zero)
    inp_cpu = torch.tensor(values, dtype=dtype)
    _check_all(
        inp_cpu,
        inp_cpu.to(flag_gems.device),
        native_nonzero=special is not None and math.isnan(special),
    )
    assert checked_launches


# The int64 public-entry UB failure has a standalone reproduction. The changed
# loop is checked for int64 with small tiles in the kernel test below.
@pytest.mark.parametrize("dtype", [dtype for dtype in DTYPES if dtype != torch.int64])
@pytest.mark.parametrize("dim", [0, -1, (0, 2), (2, 0)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("has_zeros", [False, True])
def test_all_dim_and_dims_exact_accuracy(
    checked_launches, dtype, dim, keepdim, has_zeros
):
    inp_cpu = torch.ones((5, 97, 259), dtype=dtype)
    if dtype != torch.bool:
        inp_cpu[:, :, 1::2] = -1
    if has_zeros:
        inp_cpu[0, 0, 0] = 0
        inp_cpu[4, 48, 128] = 0
        inp_cpu[-1, -1, -1] = 0
    _check_all(
        inp_cpu,
        inp_cpu.to(flag_gems.device),
        dim=dim,
        keepdim=keepdim,
        global_reduce=False,
    )
    assert checked_launches
    assert all(name == "all_kernel_dim" for name, _, _ in checked_launches)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("workers", [1, 3])
def test_all_dim_nested_loops_accuracy(all_ops, dtype, workers):
    # Isolate the changed elementwise & with multiple row and column tiles.
    inp_cpu = torch.ones((31, 259), dtype=dtype)
    inp_cpu[0, 0] = 0
    inp_cpu[5, 127] = 0
    inp_cpu[17, 128] = 0
    inp_cpu[-1, -1] = 0
    if dtype in FLOAT_DTYPES:
        inp_cpu[1, 0] = float("nan")
        inp_cpu[2, 128] = float("inf")
        inp_cpu[5, 1] = float("nan")
        inp_cpu[-1, 0] = -float("inf")
    expected = torch.all(inp_cpu, dim=1)
    inp = inp_cpu.to(flag_gems.device)
    out = (~expected).to(flag_gems.device)
    kernel = all_ops.all_kernel_dim
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    kernel[(workers,)](inp, out, 31, 259, BLOCK_M=4, BLOCK_N=128)
    _assert_exact(out, expected, torch.all(inp != 0, dim=1))
