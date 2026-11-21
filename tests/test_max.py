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

import sys
import unittest.mock

import pytest
import torch

import flag_gems
from flag_gems.ops.max import evenly_distribute_backward, max_backward

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
    # The XPU (xpu) device reference does not implement the max reduction for
    # integer dtypes ("kint16 is unsupported in xdnn_pytorch_wrapper"), so the
    # on-device torch reference raises for them. The integer semantics are
    # covered by the --ref=cpu run (see test-op.sh TEST_CASES_CPU); the device
    # run records them as SKIP (reference-side limitation, not a kernel
    # defect).
    if not cfg.TO_CPU and dtype in utils.ALL_INT_DTYPES:
        pytest.skip(
            f"XPU device reference does not implement max for {dtype}; "
            "run with --ref=cpu"
        )


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max(shape, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    res_out = flag_gems.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_all_neg_inf(shape, dtype):
    # The XPU (xpu) device reduction initializes with -FLT_MAX, so an all--inf
    # input reduces to -FLT_MAX instead of -inf; skipped on the device run
    # (reference-side deviation, not a kernel defect) and covered by the
    # --ref=cpu run (test-op.sh TEST_CASES_CPU).
    if not cfg.TO_CPU:
        pytest.skip("XPU device reduction mishandles all -inf; run with --ref=cpu")
    inp = torch.full(
        shape, fill_value=float("-inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    res_out = flag_gems.max(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [[1]])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_max_int(shape, dtype):
    _skip_if_device_ref_int(dtype)
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    res_out = flag_gems.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_uncontiguous(shape, dtype):
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

    ref_out = torch.max(ref_inp)
    res_out = flag_gems.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


# Issue #2831: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.max_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim(shape, dim, keepdim, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = flag_gems.max_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.max_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "aipu", reason="Issue #3009: Big shape run slowly."
)
@pytest.mark.parametrize("shape", [(4, 1048577, 4)])
@pytest.mark.parametrize("keepdim, dim", [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim_big_shape(shape, dim, keepdim, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)

    res_out_value, res_out_index = flag_gems.max_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


# ---------------------------------------------------------------------------
# Backward: torch.max(x) (full reduction) and torch.max(x, dim, keepdim)
#
# Two ATen backward operators cover the differentiable torch.max overloads:
# `aten::evenly_distribute_backward` splits the upstream gradient evenly over
# every position that holds the maximum, and `aten::max_backward` scatters it
# onto the int64 argmax positions saved by the forward pass. Both are covered by
# the `max_backward` marker.
# ---------------------------------------------------------------------------

BACKWARD_DIM_CASES = [
    ((4096, 256), 1),
    ((4096, 256), -1),
    ((4096, 256), 0),
    ((200, 2560, 3), 1),
    ((200, 2560, 3), -1),
]


def _reference_leaf(inp):
    return utils.to_reference(inp, True).requires_grad_(True)


@pytest.mark.max_backward
@pytest.mark.evenly_distribute_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [(1,)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_full_reduction(shape, dtype):
    """`torch.max(x)` backward splits the gradient evenly over the maximum."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_leaf = _reference_leaf(inp)

    out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_grad = torch.autograd.grad(
        torch.max(ref_leaf), ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    res_out = flag_gems.max(inp)
    res_grad = evenly_distribute_backward(out_grad, inp, res_out)

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.max_backward
@pytest.mark.evenly_distribute_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_full_reduction_ties(dtype):
    """Ties in a full reduction share the gradient instead of picking one index."""
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    for position in ((0, 1), (3, 7), (63, 0)):
        inp[position] = torch.tensor(5.0, dtype=dtype)
    ref_leaf = _reference_leaf(inp)

    out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_grad = torch.autograd.grad(
        torch.max(ref_leaf), ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    res_out = flag_gems.max(inp)
    res_grad = evenly_distribute_backward(out_grad, inp, res_out)

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert torch.count_nonzero(res_grad) == 3
    utils.gems_assert_close(
        res_grad[0, 1], utils.to_reference(out_grad, True) / 3, dtype
    )


@pytest.mark.max_backward
@pytest.mark.parametrize("shape, dim", BACKWARD_DIM_CASES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_dim(shape, dim, keepdim, dtype):
    """`torch.max(x, dim, keepdim)` backward scatters onto the saved argmax."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_values, res_indices = flag_gems.max_dim(inp, dim=dim, keepdim=keepdim)

    out_grad = torch.randn(res_values.shape, dtype=dtype, device=flag_gems.device)
    ref_leaf = _reference_leaf(inp)
    ref_values, _ = torch.max(ref_leaf, dim=dim, keepdim=keepdim)
    ref_grad = torch.autograd.grad(
        ref_values, ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    res_grad = max_backward(out_grad, inp, res_values, res_indices, dim)

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.max_backward
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_dim_ties(keepdim, dtype):
    """Tied maxima along a dim follow the single argmax index the forward saved."""
    inp = torch.randn((37, 64), dtype=dtype, device=flag_gems.device)
    row_max = inp.max(dim=1).values
    inp[:, 5] = row_max
    inp[:, 9] = row_max
    res_values, res_indices = flag_gems.max_dim(inp, dim=1, keepdim=keepdim)

    out_grad = torch.randn(res_values.shape, dtype=dtype, device=flag_gems.device)
    ref_leaf = _reference_leaf(inp)
    ref_values, ref_indices = torch.max(ref_leaf, dim=1, keepdim=keepdim)
    ref_grad = torch.autograd.grad(
        ref_values, ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    res_grad = max_backward(out_grad, inp, res_values, res_indices, 1)

    utils.gems_assert_equal(res_indices, ref_indices)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    # The forward argmax keeps the lower tied index, so the duplicate gets none.
    assert not torch.any(res_grad.select(1, 9) != 0)


@pytest.mark.max_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_dim_nan(dtype):
    """A NaN row sends its whole gradient to the argmax position ATen saved.

    The backward only consumes the int64 indices (and the values gradient), so
    this pins the NaN path of the backward contract. The indices come from ATen
    here: the pre-existing ``max_dim`` forward reduction drops NaN while it
    selects its own index, which is a forward-side limitation that this
    backward change does not alter (see the PR notes).
    """
    inp = torch.randn((64, 128), dtype=dtype, device=flag_gems.device)
    inp[3, 17] = float("nan")
    inp[40, 0] = float("nan")
    values, indices = torch.max(inp, dim=1)

    out_grad = torch.randn(values.shape, dtype=dtype, device=flag_gems.device)
    ref_leaf = _reference_leaf(inp)
    ref_values, ref_indices = torch.max(ref_leaf, dim=1)
    ref_grad = torch.autograd.grad(
        ref_values, ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    utils.gems_assert_equal(indices, ref_indices)
    res_grad = max_backward(out_grad, inp, values, indices, 1)

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert res_grad[3, 17] == out_grad[3]
    assert res_grad[40, 0] == out_grad[40]


@pytest.mark.max_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_full_reduction_nan(dtype):
    """A NaN maximum matches every NaN, so the share is split over all of them."""
    inp = torch.randn((256, 128), dtype=dtype, device=flag_gems.device)
    inp[7, 9] = float("nan")
    inp[100, 3] = float("nan")
    ref_leaf = _reference_leaf(inp)

    out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_grad = torch.autograd.grad(
        torch.max(ref_leaf), ref_leaf, utils.to_reference(out_grad, True)
    )[0]

    result = torch.max(inp)
    assert torch.isnan(result)
    res_grad = evenly_distribute_backward(out_grad, inp, result)

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert torch.count_nonzero(res_grad) == 2
    utils.gems_assert_close(
        res_grad[7, 9], utils.to_reference(out_grad, True) / 2, dtype
    )


@pytest.mark.max_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_backward_autograd_dispatch(dtype):
    """The FlagGems autograd entries must run the FlagGems Triton backward.

    ``flag_gems.max`` / ``flag_gems.max_dim`` return ``torch.autograd.Function``
    outputs, but a value comparison alone cannot tell which formula produced the
    gradient. Spy on the two backward entry points the Functions call and check
    the real ``torch.autograd.grad`` chain invokes each exactly once.
    """
    module = sys.modules["flag_gems.ops.max"]
    calls = {"max_backward": 0, "evenly_distribute_backward": 0}
    real_max_backward = module.max_backward
    real_evenly = module.evenly_distribute_backward

    def spy_max_backward(*args, **kwargs):
        calls["max_backward"] += 1
        return real_max_backward(*args, **kwargs)

    def spy_evenly(*args, **kwargs):
        calls["evenly_distribute_backward"] += 1
        return real_evenly(*args, **kwargs)

    shape = (64, 256)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_leaf = _reference_leaf(inp)

    dim_grad_out = torch.randn(
        (shape[0],), dtype=ref_leaf.dtype, device=ref_leaf.device
    )
    ref_dim_values, _ = torch.max(ref_leaf, dim=1)
    ref_dim_grad = torch.autograd.grad(ref_dim_values, ref_leaf, dim_grad_out)[0]
    full_grad_out = torch.randn((), dtype=ref_leaf.dtype, device=ref_leaf.device)
    ref_full_grad = torch.autograd.grad(torch.max(ref_leaf), ref_leaf, full_grad_out)[0]

    leaf = inp.detach().clone().requires_grad_(True)
    with (
        unittest.mock.patch.object(module, "max_backward", spy_max_backward),
        unittest.mock.patch.object(module, "evenly_distribute_backward", spy_evenly),
    ):
        (res_dim_grad,) = torch.autograd.grad(
            flag_gems.max_dim(leaf, dim=1).values,
            (leaf,),
            dim_grad_out.to(device=leaf.device, dtype=leaf.dtype),
        )
        (res_full_grad,) = torch.autograd.grad(
            flag_gems.max(leaf),
            (leaf,),
            full_grad_out.to(device=leaf.device, dtype=leaf.dtype),
        )

    assert calls["max_backward"] == 1, calls
    assert calls["evenly_distribute_backward"] == 1, calls
    utils.gems_assert_close(res_dim_grad, ref_dim_grad, dtype)
    utils.gems_assert_close(res_full_grad, ref_full_grad, dtype)


@pytest.mark.max
@pytest.mark.parametrize("shape", [(3,), (7, 129), (16, 4096)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_nan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    flat = inp.view(-1)
    # One NaN in the first reduction tile and one in the last, so both the
    # per-tile reduction and the merge of the two stages have to keep it.
    flat[0] = float("nan")
    flat[-1] = float("nan")

    ref_out = torch.max(utils.to_reference(inp))
    res_out = flag_gems.max(inp)

    assert torch.isnan(ref_out)
    assert torch.isnan(res_out)


@pytest.mark.max_dim
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_dim_nan(dtype, keepdim):
    inp = torch.randn((6, 128), dtype=dtype, device=flag_gems.device)
    inp[0, 5] = float("nan")
    inp[1, 0] = float("nan")
    inp[1, 3] = float("nan")  # the first NaN must win over the later one
    inp[2, 127] = float("nan")

    ref_value, ref_index = torch.max(utils.to_reference(inp), dim=1, keepdim=keepdim)
    res_value, res_index = flag_gems.max_dim(inp, dim=1, keepdim=keepdim)

    assert torch.isnan(res_value[:3]).all()
    assert torch.isfinite(res_value[3:]).all()
    utils.gems_assert_equal(res_index, ref_index)
    utils.gems_assert_equal(res_value, ref_value, equal_nan=True)


@pytest.mark.max_dim
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_dim_nan_cross_tile(dtype):
    inp = torch.randn((4, 5000), dtype=dtype, device=flag_gems.device)
    inp[0, 10] = float("nan")
    inp[0, 4900] = float("nan")

    ref_value, ref_index = torch.max(utils.to_reference(inp), dim=1)
    res_value, res_index = flag_gems.max_dim(inp, dim=1)

    assert torch.isnan(res_value[0])
    assert torch.isfinite(res_value[1:]).all()
    utils.gems_assert_equal(res_index, ref_index)
    utils.gems_assert_equal(res_value, ref_value, equal_nan=True)
