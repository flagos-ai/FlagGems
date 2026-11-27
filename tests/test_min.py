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
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


def _skip_if_device_ref_int(dtype):
    # The XPU (xpu) device reference does not implement the min reduction for
    # integer dtypes (e.g. "kint16 is unsupported in xdnn_pytorch_wrapper"), so
    # the on-device torch reference raises for them. The integer semantics are
    # covered by the --ref=cpu run (see test-op.sh TEST_CASES_CPU); the device
    # run records them as SKIP (reference-side limitation, not a kernel
    # defect).
    if not cfg.TO_CPU and dtype in utils.ALL_INT_DTYPES:
        pytest.skip(
            f"XPU device reference does not implement min for {dtype}; "
            "run with --ref=cpu"
        )


def _skip_if_device_ref_all_inf():
    # The XPU (xpu) device reduction initializes with +FLT_MAX, so an all-inf
    # input reduces to +FLT_MAX instead of inf; skipped on the device run
    # (reference-side deviation, not a kernel defect) and covered by the
    # --ref=cpu run (test-op.sh TEST_CASES_CPU).
    if not cfg.TO_CPU:
        pytest.skip("XPU device reduction mishandles all +inf; run with --ref=cpu")


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min(shape, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    res_out = flag_gems.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min_uncontiguous(shape, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu")[::2, ::2].to(
            flag_gems.device
        )
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu")[
            ::2, ::2
        ].to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    res_out = flag_gems.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.min
def test_min_column_slice_uncontiguous():
    inp = torch.tensor([[10.0, 1.0], [2.0, 20.0]], device=flag_gems.device)[:, 0]
    assert not inp.is_contiguous()
    ref_out = torch.min(utils.to_reference(inp))
    res_out = flag_gems.min(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_all_inf(shape, dtype):
    _skip_if_device_ref_all_inf()
    # ensure that padding value used in min is inf, not max value
    inp = torch.full(
        shape, fill_value=float("inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    res_out = flag_gems.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


# Issue #2832: fails at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.min_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min_dim(shape, dim, keepdim, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = flag_gems.min_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.min_dim
@pytest.mark.parametrize("shape", [(1024, 16, 1024), (2, 8, 70000)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_min_dim_large_grid_stride(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=1, keepdim=False)
    res_out_value, res_out_index = flag_gems.min_dim(inp, dim=1, keepdim=False)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


# ---------------------------------------------------------------------------
# Backward
#
# aten::min(Tensor) and aten::min.dim are both differentiated by the autograd
# engine, so the tests below always go through the full autograd path: build a
# forward graph, then compare the input gradient of the FlagGems entry against
# the input gradient of the native torch reference for the same upstream
# gradient.
# ---------------------------------------------------------------------------

BACKWARD_GLOBAL_SHAPES = [
    (1,),
    (1024,),
    (7, 19),
    (4096, 256),
]

BACKWARD_DIM_CASES = [
    ((1, 2), 0, False),
    ((1, 2), 1, True),
    ((7, 5, 3), 0, False),
    ((7, 5, 3), 1, True),
    ((7, 5, 3), 2, False),
    ((7, 5, 3), -1, True),
    ((7, 5, 3), -2, False),
    ((4096, 256), 0, False),
    ((4096, 256), 1, True),
    ((4096, 256), 1, False),
    ((4096, 256), -1, False),
    ((1024,), 0, False),
]


def _tie_input(shape, dtype, levels=4):
    inp = torch.randint(0, levels, shape, dtype=torch.int64, device="cpu")
    return inp.to(dtype=dtype).to(flag_gems.device)


def _nan_input(shape, dtype, stride=7):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    flat = inp.reshape(-1)
    flat[::stride] = float("nan")
    return inp


def _grad_global(inp, dtype, grad=None):
    if grad is None:
        grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    res_inp = inp.detach().clone().requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()

    res_out = flag_gems.min(res_inp)
    ref_out = torch.min(ref_inp)
    utils.gems_assert_close(res_out, ref_out, dtype)

    (res_dx,) = torch.autograd.grad(res_out, res_inp, grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
    utils.gems_assert_close(res_dx, ref_dx, dtype)
    return res_out, ref_out, res_dx, ref_dx


def _grad_dim(inp, dim, keepdim, dtype, equal_nan=False):
    res_inp = inp.detach().clone().requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()

    ref_values, ref_indices = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    res_values, res_indices = flag_gems.min_dim(res_inp, dim=dim, keepdim=keepdim)
    utils.gems_assert_equal(res_indices, ref_indices)
    utils.gems_assert_close(res_values, ref_values, dtype, equal_nan=equal_nan)

    grad = torch.randn(ref_values.shape, dtype=dtype, device=flag_gems.device)
    (res_dx,) = torch.autograd.grad(res_values, res_inp, grad)
    (ref_dx,) = torch.autograd.grad(ref_values, ref_inp, utils.to_reference(grad))
    utils.gems_assert_close(res_dx, ref_dx, dtype, equal_nan=equal_nan)
    return res_values, res_dx, ref_dx


@pytest.mark.min_backward
@pytest.mark.parametrize("shape", BACKWARD_GLOBAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _grad_global(inp, dtype)


@pytest.mark.min_backward
@pytest.mark.parametrize("shape", [(2, 32), (4096, 256)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward_ties(shape, dtype):
    # The global adjoint must split grad_output evenly between every tie.
    inp = _tie_input(shape, dtype)
    grad = (
        torch.tensor(32.0, dtype=dtype, device=flag_gems.device)
        if dtype == torch.float16 and inp.numel() > 1024
        else None
    )
    res_out, ref_out, res_dx, ref_dx = _grad_global(inp, dtype, grad=grad)
    tie_count = int((inp == ref_out).sum())
    assert tie_count > 1, "the tie case must actually contain ties"
    assert torch.count_nonzero(res_dx) == tie_count
    tied_grad = res_dx[inp == ref_out]
    assert torch.equal(tied_grad, tied_grad[0].expand_as(tied_grad))


@pytest.mark.min_backward
@pytest.mark.parametrize("shape", [(2, 32), (4096, 256)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward_nan_forward_gap(shape, dtype):
    # Some Triton backends (e.g. CUDA) use `tl.min` that drops NaN, whereas
    # others (e.g. Ascend) propagate NaN to match ATen. When `flag_gems.min`
    # returns NaN, verify full parity with ATen; otherwise verify consistency
    # with the non-NaN forward value.
    inp = _nan_input(shape, dtype)
    assert bool(torch.isnan(torch.min(inp)))

    res_inp = inp.detach().clone().requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)
    grad = (
        torch.tensor(32.0, dtype=dtype, device=flag_gems.device)
        if dtype == torch.float16 and inp.numel() > 1024
        else torch.randn((), dtype=dtype, device=flag_gems.device)
    )
    res_out = flag_gems.min(res_inp)
    ref_out = torch.min(ref_inp)

    (res_dx,) = torch.autograd.grad(res_out, res_inp, grad)
    if bool(torch.isnan(res_out)):
        (ref_dx,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
        nan_count = int(torch.isnan(inp).sum())
        assert nan_count >= 1
        utils.gems_assert_close(res_dx, ref_dx, dtype)
        assert int(torch.count_nonzero(res_dx)) == nan_count
    else:
        tie_count = int((inp == res_out).sum())
        assert tie_count >= 1
        assert int(torch.count_nonzero(res_dx)) == tie_count
        torch.testing.assert_close(
            res_dx.sum().cpu(),
            grad.cpu(),
            atol=1e-4 * tie_count,
            rtol=flag_gems.testing.RESOLUTION[dtype],
        )


@pytest.mark.min_backward
@pytest.mark.parametrize("shape,dim,keepdim", BACKWARD_DIM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _grad_dim(inp, dim, keepdim, dtype)


@pytest.mark.min_backward
@pytest.mark.parametrize(
    "shape,dim,keepdim", [((65, 130), 1, False), ((8, 9, 3), 0, True)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward_dim_ties(shape, dim, keepdim, dtype):
    # min.dim keeps only the first argmin: the adjoint must not be split.
    inp = _tie_input(shape, dtype)
    res_values, res_dx, ref_dx = _grad_dim(inp, dim, keepdim, dtype)
    assert torch.equal(res_dx.cpu(), ref_dx.cpu())
    assert int((res_dx != 0).sum()) == res_values.numel()


@pytest.mark.min_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_backward_dim_non_contiguous(dtype):
    # The adjoint only needs the reduced layout, so a transposed input works.
    inp = torch.randn((17, 33), dtype=dtype, device=flag_gems.device).t()
    assert not inp.is_contiguous()
    _grad_dim(inp, 0, False, dtype)
    _grad_dim(inp, 1, False, dtype)


@pytest.mark.min_backward
def test_min_backward_grid_stride_and_large_grid_limit(monkeypatch):
    import importlib

    module = importlib.import_module(flag_gems.min.__module__)
    assert module._global_plan(65536 * 4096)[2] == 65535
    original_plan = module._global_plan

    def force_grid_stride(n_elements):
        count_grid, count_block, _, fill_block, wide = original_plan(n_elements)
        return count_grid, count_block, 1, fill_block, wide

    monkeypatch.setattr(module, "_global_plan", force_grid_stride)
    inp = torch.randn(8193, device=flag_gems.device, requires_grad=True)
    ref_inp = inp.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(flag_gems.min(inp), inp)
    (ref_grad,) = torch.autograd.grad(torch.min(ref_inp), ref_inp)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
