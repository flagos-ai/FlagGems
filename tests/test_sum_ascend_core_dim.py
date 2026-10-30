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

"""Ascend sum launch limits: real oversized grids and forced grid-stride loops."""

import importlib
import math

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.sum,
    pytest.mark.sum_dim,
    pytest.mark.sum_dim_out,
    pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend coreDim limit"),
]

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
        with flag_gems.use_gems():
            result = torch.sum(inp, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        assert result is out
    else:
        with flag_gems.use_gems():
            result = torch.sum(inp, dim=dim, keepdim=keepdim, dtype=dtype)
    # Force asynchronous launch errors to surface at the test that caused them.
    torch.npu.synchronize()
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert result.device == inp.device
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(result, native, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "num_tiles,expected",
    [(1, 1), (65534, 65534), (65535, 65535), (65536, 65535), (131072, 65535)],
)
def test_sum_launch_grid_limit(sum_ops, num_tiles, expected):
    assert sum_ops._reduction_grid_size(num_tiles) == expected


@pytest.mark.parametrize("dim", [(0,), (-1,), (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_dim_empty_output_does_not_launch(sum_ops, monkeypatch, dim, keepdim):
    def unexpected_zero_launch(out):
        pytest.fail("An empty output must not launch the zero_ kernel")

    monkeypatch.setattr(sum_ops, "zero_", unexpected_zero_launch)
    _check_sum(torch.empty((2, 0, 3)), dim, keepdim, use_out=False)


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


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_dim_broadcast_backward_core_dim(checked_launches, dtype):
    # ExpandBackward reduces a broadcast gradient using the failing non-inner
    # path from training. Its original grid was (65536, 2, 1), i.e. 131072.
    generator = torch.Generator().manual_seed(20261020)
    grad_cpu = torch.randn((65536, 3, 5), generator=generator).to(dtype)
    bias = torch.zeros((65536, 1, 5), dtype=dtype, device=flag_gems.device)
    bias.requires_grad_()
    output = bias.expand(grad_cpu.shape)
    grad = grad_cpu.to(flag_gems.device)
    with flag_gems.use_gems(include=["sum", "sum_dim"]):
        (result,) = torch.autograd.grad(output, bias, grad_outputs=grad)
    torch.npu.synchronize()
    expected = grad_cpu.double().sum(dim=1, keepdim=True).to(dtype)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    assert any(name == "sum_dim_kernel_non_inner" for name, _ in checked_launches)


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
        with flag_gems.use_gems():
            result = torch.ops.aten.sum.out(inp, out=out)
        assert result is out
    else:
        with flag_gems.use_gems():
            result = torch.sum(inp)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-4)
    assert {name for name, _ in checked_launches} == {"sum_kernel_1", "sum_kernel_2"}
