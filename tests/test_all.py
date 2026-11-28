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
import triton

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, 1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KIND_KEEPDIM_DIMS_SHAPE = list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            utils.REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )


@pytest.mark.all
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_all(shape, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp)
    res_out = flag_gems.all(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_all_dim(shape, dtype, keepdim, dim, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.all_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dims
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.all_dims(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


# Ascend coreDim and large-tensor regressions.
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
    utils.gems_assert_equal(result, expected)
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
    if global_reduce:
        result = flag_gems.all(inp)
    elif dim is None or isinstance(dim, int):
        result = flag_gems.all_dim(inp, dim=dim, keepdim=keepdim)
    else:
        result = flag_gems.all_dims(inp, dim=dim, keepdim=keepdim)
    _assert_exact(result, expected, native)


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_all_empty_identity_without_launch(checked_launches, dtype, shape):
    inp_cpu = torch.empty(shape, dtype=dtype)
    inp = inp_cpu.to(flag_gems.device)
    _check_all(inp_cpu, inp)
    _check_all(inp_cpu, inp, dim=None, keepdim=True, global_reduce=False)
    assert not checked_launches


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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
@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all
@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction limits")
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


@pytest.mark.all_dim
@pytest.mark.all_dims
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend reduction masks")
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("dim", [1, (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("has_zeros", [False, True])
def test_all_dim_tail_masks_review(
    all_ops, monkeypatch, dtype, dim, keepdim, has_zeros
):
    # Compressing either input yields M=31, N=259. Both masks are active
    # in the last tile for BLOCK_M=4 and BLOCK_N=128.
    shape = (31, 259) if isinstance(dim, int) else (7, 31, 37)
    inp_cpu = torch.ones(shape, dtype=dtype)
    if dtype != torch.bool:
        inp_cpu[..., 1::2] = -2
    if has_zeros:
        inp_cpu[(0,) * len(shape)] = 0
        inp_cpu[tuple(s - 1 for s in shape)] = 0
    kernel = all_ops.all_kernel_dim
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    launches = []

    class TailKernel:
        def __getitem__(self, grid):
            def launch(inp, out, m, n):
                assert (m, n) == (31, 259)
                assert m % 4 and n % 128
                assert grid({"BLOCK_M": 4}) == (8,)
                launches.append(True)
                # Three programs also force multiple row grid strides.
                return kernel[(3,)](inp, out, m, n, BLOCK_M=4, BLOCK_N=128)

            return launch

    monkeypatch.setattr(all_ops, "all_kernel_dim", TailKernel())
    _check_all(
        inp_cpu,
        inp_cpu.to(flag_gems.device),
        dim=dim,
        keepdim=keepdim,
        global_reduce=False,
    )
    assert launches == [True]


@pytest.mark.all_dim
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon vendor override")
class TestHygonAllDim:
    @staticmethod
    def _module():
        import importlib

        return importlib.import_module(flag_gems.all_dim.__module__)

    @staticmethod
    def _check(inp, dim, keepdim, op=None):
        cpu = inp.cpu()
        before = cpu.clone()
        expected = torch.all(cpu, dim=dim, keepdim=keepdim)
        result = (op or flag_gems.all_dim)(inp, dim=dim, keepdim=keepdim)
        assert result.device == inp.device
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert not result.requires_grad
        assert not torch._C._is_alias_of(result, inp)
        utils.gems_assert_equal(result, expected)
        torch.testing.assert_close(inp.cpu(), before, rtol=0, atol=0, equal_nan=True)
        return result

    @pytest.mark.parametrize(
        "shape",
        [
            (128,),
            (5, 129),
            (17, 255),
            (256, 256),
            (8, 33, 513),
            (1024, 1024),
            (65536, 128),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("positive_dim", [False, True])
    def test_fast_accuracy(self, shape, keepdim, positive_dim):
        generator = torch.Generator().manual_seed(20261104)
        cpu = torch.randint(1, 9, shape, generator=generator).float()
        cpu.reshape(-1, shape[-1])[::4, -1] = 0
        inp = cpu.to(flag_gems.device)
        dim = inp.ndim - 1 if positive_dim else -1
        assert self._module()._can_use_all_dim_fast_path(inp, dim, keepdim)
        self._check(inp, dim, keepdim)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_truth_and_padding(self, keepdim):
        cpu = torch.ones((11, 129), dtype=torch.float32)
        # All-true rows, signed zeros, NaNs, infinities and the smallest signed
        # FP32 subnormals; the last group has three real rows and one padding row.
        cpu[0].fill_(float("nan"))
        cpu[1].fill_(float("inf"))
        cpu[2].fill_(-float("inf"))
        cpu[3].fill_(0.0)
        cpu[4, -1] = -0.0
        tiny = torch.tensor([1, -2147483647], dtype=torch.int32).view(torch.float32)
        cpu[5].fill_(tiny[0])
        cpu[6].fill_(tiny[1])
        cpu[7, -1] = float("nan")
        cpu[8, 0] = 0
        cpu[9, -1] = 0
        self._check(cpu.to(flag_gems.device), -1, keepdim)

    @pytest.mark.parametrize(
        "shape,dtype,dim,layout",
        [
            ((7, 127), torch.float32, -1, "contiguous"),
            ((7, 1025), torch.float32, -1, "contiguous"),
            ((7, 128), torch.float32, 0, "contiguous"),
            ((7, 128), torch.float16, -1, "contiguous"),
            ((7, 128), torch.bfloat16, -1, "contiguous"),
            ((7, 128), torch.bool, -1, "contiguous"),
            ((7, 256), torch.float32, -1, "stride2"),
            ((128, 7), torch.float32, -1, "transpose"),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_fallback_accuracy(self, shape, dtype, dim, layout, keepdim):
        cpu = torch.ones(shape, dtype=dtype)
        cpu.reshape(-1)[-1] = 0
        inp = cpu.to(flag_gems.device)
        if layout == "stride2":
            inp = inp[:, ::2]
        elif layout == "transpose":
            inp = inp.T
        assert not self._module()._can_use_all_dim_fast_path(inp, dim, keepdim)
        self._check(inp, dim, keepdim)

    @pytest.mark.parametrize("dim", [None, -1, 1])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_offset_alias_and_dispatch(self, dim, keepdim, monkeypatch):
        # The data pointer already includes the nonzero storage offset.
        backing = torch.ones(7 + 5 * 129, device=flag_gems.device)
        inp = backing[7:].view(5, 129)
        inp[2, -1] = 0
        self._check(inp, dim, keepdim)
        with monkeypatch.context() as registration:
            registration.setattr(flag_gems, "current_work_registrar", None)
            library = torch.library.Library("aten", "IMPL")
            try:
                flag_gems.only_enable(lib=library, include=["all_dim"])
                actual = torch.all(inp, dim=dim, keepdim=keepdim)
            finally:
                library._destroy()
        utils.gems_assert_equal(
            actual.cpu(), torch.all(inp.cpu(), dim=dim, keepdim=keepdim)
        )
        # Positive/negative dimension aliases must agree, without output aliasing.
        if dim is not None:
            self._check(inp, -1, keepdim)
            self._check(inp, 1, keepdim)

    def test_formal_export_and_full_reduction(self):
        from flag_gems.ops.all import all as generic_all
        from flag_gems.ops.all import all_dims as generic_all_dims

        assert flag_gems.all_dim is self._module().all_dim
        assert flag_gems.all is generic_all
        assert flag_gems.all_dims is generic_all_dims
        inp = torch.ones((5, 129), device=flag_gems.device)
        torch.testing.assert_close(
            flag_gems.all(inp).cpu(), torch.all(inp.cpu()), rtol=0, atol=0
        )
        for dim in ([1], (-1,), [], [0, 1]):
            self._check(inp, dim, True, op=flag_gems.all_dims)

    @pytest.mark.parametrize("dim", [-3, 2])
    def test_invalid_dim_falls_back(self, dim):
        inp = torch.ones((5, 128), device=flag_gems.device)
        assert not self._module()._can_use_all_dim_fast_path(inp, dim, False)
        with pytest.raises((AssertionError, IndexError)):
            flag_gems.all_dim(inp, dim=dim)

    def test_unresolved_negative_view(self):
        inp = torch.ones((5, 129), device=flag_gems.device)._neg_view()
        assert not self._module()._can_use_all_dim_fast_path(inp, -1, False)
        self._check(inp, -1, False)

    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            ((3, 0), -1, False, torch.float32),
            ((0, 128), -1, True, torch.float32),
            ((), 0, False, torch.float32),
            ((5, 128), -1, False, torch.uint8),
            ((5, 128), None, True, torch.float32),
        ],
    )
    def test_excluded_semantics_delegate(self, monkeypatch, shape, dim, keepdim, dtype):
        # Even known baseline defects must reach the captured generic function.
        module = self._module()
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
        sentinel = object()
        calls = []

        def fallback(value, dim=None, keepdim=False):
            calls.append((value, dim, keepdim))
            return sentinel

        monkeypatch.setattr(module, "generic_all_dim", fallback)
        assert not module._can_use_all_dim_fast_path(inp, dim, keepdim)
        assert flag_gems.all_dim(inp, dim=dim, keepdim=keepdim) is sentinel
        assert len(calls) == 1
        assert calls[0][0] is inp
        assert calls[0][1:] == (dim, keepdim)

    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            pytest.param(
                (3, 0),
                -1,
                torch.float32,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic divides by N=0; upstream #8412 is separate",
                ),
            ),
            pytest.param(
                (),
                0,
                torch.float32,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic rejects scalar dim=0; separate semantics",
                ),
            ),
            pytest.param(
                (5, 128),
                -1,
                torch.uint8,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic returns bool instead of uint8; separate semantics",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_known_baseline_semantics(self, shape, dim, dtype, keepdim):
        # Keep CPU correctness assertions visible; --runxfail exposes defects.
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
        self._check(inp, dim, keepdim)

    def test_empty_full_reduction(self):
        inp = torch.empty((3, 0), device=flag_gems.device)
        self._check(inp, None, False)
        self._check(inp, None, True)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_empty_spectator(self, keepdim):
        inp = torch.empty((0, 128), device=flag_gems.device)
        assert not self._module()._can_use_all_dim_fast_path(inp, -1, keepdim)
        self._check(inp, -1, keepdim)

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((5, 127), False),
            ((5, 128), True),
            ((5, 1024), True),
            ((5, 1025), False),
            ((65536, 128), True),
            ((65537, 128), False),
            ((1, 1, 1, 1, 1, 1, 1, 128), True),
            ((1, 1, 1, 1, 1, 1, 1, 1, 128), False),
        ],
    )
    def test_guard_boundaries(self, shape, expected):
        # Metadata-only tensors avoid allocating the excluded upper-bound case.
        inp = torch.empty(shape, dtype=torch.float32, device="meta")
        assert self._module()._can_use_all_dim_fast_path(inp, -1, False) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="Frozen generic full all ignores strides; upstream #7158 is separate",
    )
    def test_known_full_strided_semantics(self):
        backing = torch.zeros((2, 60), device=flag_gems.device)
        backing[:, -1] = 1
        inp = backing[:, -1]
        expected = torch.all(inp.cpu())
        result = flag_gems.all(inp)
        utils.gems_assert_equal(result, expected)

    @pytest.mark.xfail(
        strict=True,
        reason="Frozen generic full all returns bool for uint8; separate semantics",
    )
    def test_known_full_uint8_semantics(self):
        inp = torch.ones((5, 128), dtype=torch.uint8, device=flag_gems.device)
        expected = torch.all(inp.cpu())
        result = flag_gems.all(inp)
        assert result.dtype == expected.dtype
        utils.gems_assert_equal(result, expected)

    def test_fast_launch_is_selected(self, monkeypatch):
        module = self._module()
        real_kernel = module._all_f32_rows
        grids = []

        class RecordingKernel:
            def __getitem__(self, grid):
                grids.append(grid)
                return real_kernel[grid]

        monkeypatch.setattr(module, "_all_f32_rows", RecordingKernel())
        inp = torch.ones((5, 129), device=flag_gems.device)
        self._check(inp, -1, True)
        assert grids == [(2,)]
