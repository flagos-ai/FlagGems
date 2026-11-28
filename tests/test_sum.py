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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


@pytest.mark.sum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    res_out = flag_gems.sum(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.sum
@pytest.mark.sum_dim
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend sum reduction")
@pytest.mark.parametrize("dim", [None, ()])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_dim_full_reduction_fp16_partial_overflow(dim, keepdim):
    # Each first-stage partial exceeds FP16's range, but the final sum is zero.
    inp_cpu = torch.full((1024, 1024), 128.0, dtype=torch.float16)
    inp_cpu[512:] = -128.0
    expected = torch.sum(inp_cpu.float(), dim=dim, keepdim=keepdim).to(inp_cpu.dtype)
    inp = inp_cpu.to(flag_gems.device)

    result = flag_gems.sum_dim(inp, dim=dim, keepdim=keepdim)

    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_large_ascend_reduction_tile(use_out):
    if flag_gems.device != "npu":
        pytest.skip("Ascend local-buffer reduction limit")

    # The old sqrt-based heuristic selected 32768, overflowing Ascend's UB.
    # At BLOCK_SIZE=4096 this also exceeds coreDim and needs multiple mid tiles.
    inp = torch.randn(2**28 + 97, dtype=torch.float32, device=flag_gems.device)
    if use_out:
        result = torch.empty((), dtype=inp.dtype, device=inp.device)
        flag_gems.sum_out(inp, out=result)
    else:
        result = flag_gems.sum(inp)
    expected = torch.sum(inp)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-2)


@pytest.mark.sum
def test_sum_ascend_grid_stride(monkeypatch):
    if flag_gems.device != "npu":
        pytest.skip("Ascend launch-grid limit")

    sum_ops = importlib.import_module(flag_gems.sum.__module__)
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 1)
    monkeypatch.setattr(sum_ops, "_ASCEND_SUM_MAX_BLOCK_SIZE", 128)
    inp = torch.randn(65537, dtype=torch.float32, device=flag_gems.device)
    expected = torch.sum(inp)
    result = flag_gems.sum(inp)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-3)


INCLUDE_0_SHAPES = [(1, 0, 128, 512), (4096, 1, 256, 0), (200, 10, 0, 3)]


@pytest.mark.sum_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + INCLUDE_0_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    # Direct GEMS call: the dispatcher lifts an int dim to IntList, so pass a
    # list here to match sum_dim's own contract.
    dims = [dim] if isinstance(dim, int) else list(dim)
    res_out = flag_gems.sum_dim(inp, dim=dims, keepdim=keepdim)

    dim_list = [d % inp.ndim for d in dims]
    _dim = 1
    for d in dim_list:
        _dim *= shape[d]
    if dim_list == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.sum_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_sum_dim_bool(shape, dim, keepdim):
    # Regression test for #6704: sum.dim_IntList on a bool input must count
    # True values, not saturate to an "any" 0/1. Force a mix of True/False
    # along the reduced dim so a saturated accumulator disagrees with a
    # true count instead of accidentally matching it.
    inp = torch.zeros(shape, dtype=torch.bool, device=flag_gems.device)
    idx = [slice(None)] * inp.ndim
    idx[dim] = slice(0, shape[dim] // 2 + 1)
    inp[tuple(idx)] = True
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.sum_dim_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim_out(shape, dim, keepdim, dtype):
    # Regression test: sum_dim_out must resize external out tensor and skip squeeze.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_result = torch.sum(ref_inp, dim=dim, keepdim=keepdim)

    # Pre-allocate out tensor with wrong shape to test resize logic
    out = torch.empty((1,), dtype=dtype, device=flag_gems.device)
    # Direct GEMS call: pass dim as a list (the dispatcher would lift int -> IntList).
    dims = [dim] if isinstance(dim, int) else list(dim)
    res_result = flag_gems.sum_dim_out(inp, dim=dims, keepdim=keepdim, out=out)

    dim_list = [d % inp.ndim for d in dims]
    _dim = 1
    for d in dim_list:
        _dim *= shape[d]
    if dim_list == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_result, ref_result, dtype, reduce_dim=_dim)
    utils.gems_assert_close(out, ref_result, dtype, reduce_dim=_dim)


@pytest.mark.sum_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)

    out = torch.empty([], dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.sum_out(inp, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=inp.numel())


# Ascend coreDim and large-tensor regressions.
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: (1e-5, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (8e-3, 8e-3),
}


@pytest.fixture
def sum_ops():
    # Import the actual registered module so monkeypatches affect dispatch too.
    return importlib.import_module(flag_gems.sum.__module__)


@pytest.fixture
def checked_launches(sum_ops, monkeypatch):
    launches = []

    class CheckedKernel:
        def __init__(self, name, kernel):
            self.name = name
            self.kernel = kernel

        def __getitem__(self, grid):
            def checked_grid(meta):
                resolved = grid(meta) if callable(grid) else grid
                assert 0 < math.prod(resolved) <= sum_ops._ASCEND_MAX_CORE_DIM
                launches.append((self.name, resolved))
                return resolved

            return self.kernel[checked_grid]

    for name in (
        "sum_kernel_1",
        "sum_kernel_2",
        "sum_dim_kernel_non_inner",
        "sum_dim_kernel_inner",
        "sum_dim_kernel",
    ):
        monkeypatch.setattr(sum_ops, name, CheckedKernel(name, getattr(sum_ops, name)))
    return launches


def _check_sum(inp_cpu, dim, keepdim, use_out, output_dtype=None):
    inp = inp_cpu.to(flag_gems.device)
    dtype = output_dtype or inp_cpu.dtype
    # FP64 CPU accumulation provides an independent precision reference.
    expected = torch.sum(inp_cpu.double(), dim=dim, keepdim=keepdim).to(dtype)
    native = torch.sum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    if use_out:
        out = torch.empty(0, dtype=dtype, device=inp.device)
        result = flag_gems.sum_dim_out(
            inp, dim=dim, keepdim=keepdim, dtype=dtype, out=out
        )
        assert result is out
    else:
        result = flag_gems.sum_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    # Force asynchronous launch errors to surface at the test that caused them.
    torch.npu.synchronize()
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert result.device == inp.device
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(result, native, rtol=rtol, atol=atol)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize(
    "num_tiles,expected",
    [(1, 1), (65534, 65534), (65535, 65535), (65536, 65535), (131072, 65535)],
)
def test_sum_launch_grid_limit(sum_ops, num_tiles, expected):
    assert sum_ops._reduction_grid_size(num_tiles) == expected


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize("dim", [(0,), (-1,), (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_dim_empty_output_does_not_launch(
    sum_ops, monkeypatch, checked_launches, dim, keepdim, use_out
):
    def unexpected_zero_launch(out):
        pytest.fail("An empty output must not launch the zero_ kernel")

    monkeypatch.setattr(sum_ops, "zero_", unexpected_zero_launch)
    _check_sum(torch.empty((2, 0, 3)), dim, keepdim, use_out=use_out)
    assert not checked_launches


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize(
    "shape,dim",
    [
        pytest.param((65534, 3), (-1,), id="inner-below-limit"),
        pytest.param((65535, 3), (-1,), id="inner-at-limit"),
        pytest.param((65536, 3), (-1,), id="inner-above-limit"),
        pytest.param((131073, 3), (-1,), id="inner-multiple-strides-tail"),
        # TILE_K=4: both original grid axes fit, but their product does not.
        pytest.param((32768, 3, 5), (1,), id="non-inner-product-65536"),
        pytest.param((65536, 3, 5), (1,), id="non-inner-product-131072"),
        pytest.param((65537, 3, 5), (1,), id="non-inner-multiple-strides-tail"),
        # M exceeds the limit even with the largest autotuned BLOCK_M=8.
        pytest.param((524291, 2, 3), (1, 2), id="multi-dim-all-configs-tail"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_dim_real_core_dim_limit(
    checked_launches, shape, dim, dtype, keepdim, use_out
):
    generator = torch.Generator().manual_seed(20261020)
    inp_cpu = torch.randn(shape, generator=generator).to(dtype)
    _check_sum(inp_cpu, dim, keepdim, use_out)
    assert checked_launches


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize(
    "shape,dim",
    [
        pytest.param((13, 7), (-1,), id="inner-one-tile"),
        pytest.param((7, 32771), (-1,), id="inner-reduction-loop"),
        pytest.param((13, 7, 5), (1,), id="non-inner-one-tile"),
        pytest.param((13, 1027, 5), (1,), id="non-inner-reduction-loop"),
        pytest.param((1, 3, 257), (1,), id="non-inner-k-only"),
        pytest.param((1, 259, 257), (1,), id="non-inner-k-only-reduction-loop"),
        pytest.param((2, 37, 3), (0, 2), id="multi-dim-nonconsecutive"),
        pytest.param((3, 37, 683), (0, 2), id="multi-dim-reduction-loop"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_dim_forced_grid_stride(
    sum_ops, monkeypatch, checked_launches, shape, dim, dtype, keepdim, use_out
):
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 3)
    generator = torch.Generator().manual_seed(20261020)
    inp_cpu = torch.randn(shape, generator=generator).to(dtype)
    _check_sum(inp_cpu, dim, keepdim, use_out)
    assert checked_launches
    assert all(grid == (3,) for _, grid in checked_launches)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize("dim", [(0,), (-1,), (0, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_dim_grid_stride_noncontiguous_dtype(
    sum_ops, monkeypatch, checked_launches, dim, dtype, use_out
):
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 3)
    generator = torch.Generator().manual_seed(20261020)
    inp_cpu = torch.randn((37, 5, 7), generator=generator).to(dtype).transpose(0, 1)
    assert not inp_cpu.is_contiguous()
    _check_sum(inp_cpu, dim, True, use_out, output_dtype=torch.float32)
    assert checked_launches


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_dim_broadcast_gradient_core_dim(checked_launches, dtype):
    # Explicitly run the reduction used by ExpandBackward through the public
    # API. Its original grid was (65536, 2, 1), i.e. 131072.
    generator = torch.Generator().manual_seed(20261020)
    grad_cpu = torch.randn((65536, 3, 5), generator=generator).to(dtype)
    bias = torch.zeros((65536, 1, 5), dtype=torch.float64, requires_grad=True)
    output = bias.expand(grad_cpu.shape)
    grad = grad_cpu.to(flag_gems.device)
    (expected,) = torch.autograd.grad(output, bias, grad_outputs=grad_cpu.double())
    result = flag_gems.sum_dim(grad, dim=[1], keepdim=True)
    torch.npu.synchronize()
    expected = expected.to(dtype)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    assert any(name == "sum_dim_kernel_non_inner" for name, _ in checked_launches)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend coreDim limit")
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_full_reduction_forced_grid_stride(
    sum_ops, monkeypatch, checked_launches, use_out
):
    # Exercise both existing full-reduction stages alongside the dim kernels.
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 3)
    monkeypatch.setattr(sum_ops, "_ASCEND_SUM_MAX_BLOCK_SIZE", 128)
    generator = torch.Generator().manual_seed(20261020)
    inp_cpu = torch.randn(65537, generator=generator)
    inp = inp_cpu.to(flag_gems.device)
    expected = inp_cpu.double().sum().float()
    if use_out:
        out = torch.empty((), device=inp.device)
        result = flag_gems.sum_out(inp, out=out)
        assert result is out
    else:
        result = flag_gems.sum(inp)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-4)
    assert {name for name, _ in checked_launches} == {"sum_kernel_1", "sum_kernel_2"}


@pytest.fixture
def no_sum_reduction_launch(sum_ops, monkeypatch):
    class UnexpectedKernel:
        def __getitem__(self, grid):
            pytest.fail("Empty input must not launch a reduction kernel")

    for name in (
        "sum_kernel_1",
        "sum_kernel_2",
        "sum_dim_kernel_non_inner",
        "sum_dim_kernel_inner",
        "sum_dim_kernel",
    ):
        monkeypatch.setattr(sum_ops, name, UnexpectedKernel())
    zero = sum_ops.zero_

    def checked_zero(out):
        assert out.numel() > 0, "Empty output must not launch zero_"
        return zero(out)

    monkeypatch.setattr(sum_ops, "zero_", checked_zero)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend sum reduction")
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((2, 0, 3), 1),
        ((2, 0, 3), (1,)),
        ((2, 0, 3), (-2,)),
        ((2, 0, 3), (0, 1)),
        ((2, 0, 3), 0),
        ((2, 0, 3), (0, 2)),
        ((0, 2, 3), (1,)),
        ((2, 0, 3), None),
        ((2, 0, 3), ()),
        ((2, 0, 3), []),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [
        (torch.float32, None),
        (torch.bool, None),
        (torch.int32, None),
        (torch.float16, torch.float32),
        (torch.float32, torch.bool),
    ],
)
def test_sum_dim_empty_review(
    sum_ops, no_sum_reduction_launch, shape, dim, keepdim, use_out, dtype, output_dtype
):
    assert "_ascend" in sum_ops.__name__
    inp_cpu = torch.empty(shape, dtype=dtype)
    expected = torch.sum(inp_cpu, dim=dim, keepdim=keepdim, dtype=output_dtype)
    inp = inp_cpu.to(flag_gems.device)
    if use_out:
        out = torch.empty(0, dtype=expected.dtype, device=inp.device)
        result = flag_gems.sum_dim_out(
            inp, dim=dim, keepdim=keepdim, dtype=output_dtype, out=out
        )
        assert result is out
    else:
        result = flag_gems.sum_dim(inp, dim=dim, keepdim=keepdim, dtype=output_dtype)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend sum reduction")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.bool, torch.int32]
)
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_empty_review(no_sum_reduction_launch, dtype, output_dtype, use_out):
    inp_cpu = torch.empty((2, 0, 3), dtype=dtype)
    expected = torch.sum(inp_cpu, dtype=output_dtype)
    inp = inp_cpu.to(flag_gems.device)
    if use_out:
        out = torch.empty(0, dtype=expected.dtype, device=inp.device)
        result = flag_gems.sum_out(inp, dtype=output_dtype, out=out)
        assert result is out
    else:
        result = flag_gems.sum(inp, dtype=output_dtype)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.sum_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend sum reduction")
@pytest.mark.parametrize("layout", ["transpose", "slice", "permute"])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_sum_out_noncontiguous_review(
    sum_ops, monkeypatch, checked_launches, layout, output_dtype, explicit_dtype
):
    assert "_ascend" in sum_ops.__name__
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 3)
    generator = torch.Generator().manual_seed(20261117)
    base_cpu = torch.randn((17, 19, 23), generator=generator).half()
    base = base_cpu.to(flag_gems.device)

    def view(x):
        if layout == "transpose":
            return x.transpose(0, 1)
        if layout == "slice":
            return x[:, ::2]
        return x.permute(2, 0, 1)

    inp_cpu, inp = view(base_cpu), view(base)
    assert not inp.is_contiguous()
    expected = inp_cpu.double().sum().to(output_dtype)
    out = torch.empty(0, dtype=output_dtype, device=inp.device)
    result = flag_gems.sum_out(
        inp, dtype=output_dtype if explicit_dtype else None, out=out
    )
    assert result is out
    torch.npu.synchronize()
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=TOLERANCES[output_dtype][0],
        atol=TOLERANCES[output_dtype][1],
    )
    assert {name for name, _ in checked_launches} == {"sum_kernel_1", "sum_kernel_2"}


@pytest.mark.sum
@pytest.mark.sum_out
@pytest.mark.sum_dim
@pytest.mark.sum_dim_out
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend sum reduction")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["overflow", "rounding"])
@pytest.mark.parametrize("entry", ["full", None, ()])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_sum_partial_precision_review(
    sum_ops, monkeypatch, dtype, case, entry, keepdim, use_out
):
    monkeypatch.setattr(sum_ops, "_ASCEND_MAX_CORE_DIM", 3)
    if case == "overflow":
        inp_cpu = torch.full((1024, 1024), 128.0, dtype=dtype)
        inp_cpu[512:] = -128.0
    else:
        # First-stage tiles sum to +4097 and -4096. Rounding the positive
        # partial to FP16/BF16 loses the residual before cancellation.
        inp_cpu = torch.zeros((1024, 1024), dtype=dtype)
        inp_cpu[:512, 0] = 4096
        inp_cpu[:512, 1] = 1
        inp_cpu[512:, 0] = -4096
    expected = inp_cpu.double().sum().to(dtype)
    if entry != "full" and keepdim:
        expected = expected.reshape(1, 1)
    inp = inp_cpu.to(flag_gems.device)
    first_stage = sum_ops.sum_kernel_1
    launches = []

    class CheckedPartials:
        def __getitem__(self, grid):
            def launch(inp, mid, size, block):
                assert mid.dtype == torch.float32
                assert mid.numel() == math.ceil(size / block)
                assert mid.numel() > math.prod(grid)
                launches.append(grid)
                return first_stage[grid](inp, mid, size, block)

            return launch

    monkeypatch.setattr(sum_ops, "sum_kernel_1", CheckedPartials())
    if use_out:
        out = torch.empty(0, dtype=dtype, device=inp.device)
        if entry == "full":
            result = flag_gems.sum_out(inp, out=out)
        else:
            result = flag_gems.sum_dim_out(inp, dim=entry, keepdim=keepdim, out=out)
        assert result is out
    elif entry == "full":
        result = flag_gems.sum(inp)
    else:
        result = flag_gems.sum_dim(inp, dim=entry, keepdim=keepdim)
    torch.npu.synchronize()
    assert launches
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
