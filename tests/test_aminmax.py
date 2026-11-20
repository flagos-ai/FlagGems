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
from .accuracy_utils import _SKIP_FP64

if cfg.QUICK_MODE:
    DIMS_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
    KEEPDIM_DIMS_SHAPE = [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
else:
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    KEEPDIM_DIMS_SHAPE = list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )


@pytest.mark.aminmax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_aminmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # torch.aminmax only supports single dim, use torch.amin/amax for multi-dim
    if isinstance(dim, list):
        ref_min = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
        ref_max = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    else:
        ref_min, ref_max = torch.aminmax(ref_inp, dim=dim, keepdim=keepdim)
    # Call the FlagGems entry points directly rather than dispatching through
    # use_gems(): the framework handles dispatch, and check-kernelgen-tests
    # forbids use_gems() in tests for KernelGen operators.
    if isinstance(dim, list):
        res_min = flag_gems.amin(inp, dim=dim, keepdim=keepdim)
        res_max = flag_gems.amax(inp, dim=dim, keepdim=keepdim)
    else:
        res_min, res_max = flag_gems.aminmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.aminmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_aminmax_no_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch.aminmax(ref_inp)
    res_min, res_max = flag_gems.aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out
#
# The operator ids stay `_aminmax` / `_aminmax_out`, but pytest refuses to build a
# marker from an attribute starting with an underscore, so the markers are spelled
# `underscore_aminmax` / `underscore_aminmax_out`. The earlier approach of
# registering the names on pytest's private MarkGenerator has been dropped.
# ---------------------------------------------------------------------------

# Shapes for the whole-tensor _aminmax reduction.
UNDERSCORE_AMINMAX_SHAPES = utils.REDUCTION_SHAPES + [(1, 8192), (32, 50257)]


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_zero(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_inf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_flat = inp.float().flatten()
    inp_flat[0] = float("inf")
    inp_flat[1] = float("-inf")
    inp = inp_flat.reshape(shape).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min, equal_nan=True)
    utils.gems_assert_equal(res_max, ref_max, equal_nan=True)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
def test__aminmax_int(shape):
    inp = torch.randint(-100, 100, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
def test__aminmax_scalar():
    inp = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax_out
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_max = torch.empty((), dtype=dtype, device=ref_inp.device)
    torch.ops.aten._aminmax.out(ref_inp, out0=ref_min, out1=ref_max)

    min_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    max_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    flag_gems._aminmax_out(inp, out0=min_out, out1=max_out)

    utils.gems_assert_equal(min_out, ref_min)
    utils.gems_assert_equal(max_out, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, pytest.param(torch.float64, marks=_SKIP_FP64), torch.float16],
)
@pytest.mark.parametrize("nan_pos", ["first", "middle", "last"])
def test_underscore_aminmax_nan(dtype, nan_pos):
    """A single NaN forces *both* outputs to NaN, as torch._aminmax does.

    tl.min / tl.max drop NaNs rather than propagating them, so this needs an
    explicit check in each reduction stage. The tensor is large enough that the
    NaN has to survive both stages, and the position is varied so a NaN in a
    non-first tile is covered too.
    """
    n = 100000
    inp = torch.randn(n, dtype=dtype, device=flag_gems.device)
    idx = {"first": 0, "middle": n // 2, "last": n - 1}[nan_pos]
    inp[idx] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    assert torch.isnan(ref_min) and torch.isnan(ref_max), "reference must be NaN"
    utils.gems_assert_equal(res_min, ref_min, equal_nan=True)
    utils.gems_assert_equal(res_max, ref_max, equal_nan=True)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize(
    "view", ["slice", "transpose", "narrow", "expand", "broadcast"]
)
def test_underscore_aminmax_non_contiguous(view):
    """Non-dense views must be read in the right order.

    The kernels index with a flat offset, so a sliced or transposed view would
    otherwise read the wrong elements, and an expanded view (stride 0) would read
    past its own storage.
    """
    base = torch.randn(8, 12, dtype=torch.float32, device=flag_gems.device)
    inp = {
        "slice": base[:, ::2],
        "transpose": base.t(),
        "narrow": base[2:6, 3:9],
        "expand": base[:1, :1].expand(6, 9),
        "broadcast": base[:1].expand(8, 12),
    }[view]
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax_out
def test_underscore_aminmax_out_resizes_and_aliases():
    """Mis-shaped outputs are resized to scalars and the caller's objects returned."""
    inp = torch.randn(64, dtype=torch.float32, device=flag_gems.device)
    ref_min, ref_max = torch._aminmax(utils.to_reference(inp))

    out0 = torch.empty(5, dtype=torch.float32, device=flag_gems.device)
    out1 = torch.empty(3, 7, dtype=torch.float32, device=flag_gems.device)
    res0, res1 = flag_gems._aminmax_out(inp, out0=out0, out1=out1)

    assert res0 is out0 and res1 is out1, "returned tensors must alias out0/out1"
    assert out0.shape == torch.Size([]) and out1.shape == torch.Size([])
    utils.gems_assert_equal(out0, ref_min)
    utils.gems_assert_equal(out1, ref_max)


@pytest.mark.underscore_aminmax_out
@pytest.mark.parametrize("bad", ["dtype", "device"])
def test_underscore_aminmax_out_rejects_mismatch(bad):
    """A dtype or device mismatch on the outputs raises, as in ATen."""
    inp = torch.randn(32, dtype=torch.float32, device=flag_gems.device)
    if bad == "dtype":
        out0 = torch.empty((), dtype=torch.float64, device=flag_gems.device)
        out1 = torch.empty((), dtype=torch.float32, device=flag_gems.device)
        match = "dtype"
    else:
        if utils.TO_CPU:
            pytest.skip("device mismatch needs a non-CPU device under test")
        out0 = torch.empty((), dtype=torch.float32, device="cpu")
        out1 = torch.empty((), dtype=torch.float32, device="cpu")
        match = "device"
    with pytest.raises(RuntimeError, match=match):
        flag_gems._aminmax_out(inp, out0=out0, out1=out1)


@pytest.mark.underscore_aminmax
def test_underscore_aminmax_empty_matches_aten():
    """An empty input is rejected, matching ATen's "no identity" error."""
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._aminmax(inp)


@pytest.mark.aminmax
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column", [0, 1])
def test_aminmax_coordinate_view(dtype, column):
    coords_cpu = torch.tensor([[0, 5], [2, 1], [3, 6]], dtype=dtype)
    coords = coords_cpu.to(flag_gems.device)
    inp = coords[:, column]
    assert inp.stride() == (2,)
    assert inp.storage_offset() == column

    expected = torch.aminmax(coords_cpu[:, column])
    actual = flag_gems.aminmax(inp)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


def _aminmax_strided_view(base, view):
    return {
        "column": lambda: base[:, 1],
        "slice": lambda: base[:, 1::3],
        "transpose": lambda: base.t(),
        "narrow": lambda: base[2:6, 3:9],
        "expand": lambda: base[:1, :1].expand(6, 9),
        "broadcast": lambda: base[:1].expand(8, 12),
    }[view]()


@pytest.mark.aminmax
@pytest.mark.parametrize(
    "view", ["column", "slice", "transpose", "narrow", "expand", "broadcast"]
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("with_out", [False, True])
def test_aminmax_no_dim_strided(view, dtype, keepdim, with_out):
    base_cpu = torch.arange(-50, 46, dtype=dtype).reshape(8, 12)
    base = base_cpu.to(flag_gems.device)
    inp = _aminmax_strided_view(base, view)
    ref_inp = _aminmax_strided_view(base_cpu, view)
    assert not inp.is_contiguous()
    expected = torch.aminmax(ref_inp, keepdim=keepdim)
    out = None
    if with_out:
        out = tuple(
            torch.empty(ref.shape, dtype=dtype, device=flag_gems.device)
            for ref in expected
        )

    actual = flag_gems.aminmax(inp, keepdim=keepdim, out=out)

    for i, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == dtype
        assert result.device == inp.device
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
        if with_out:
            assert result.data_ptr() == out[i].data_ptr()
    torch.testing.assert_close(base.cpu(), base_cpu, rtol=0, atol=0)


# PyTorch does not register an aminmax derivative, so the mathematical
# reference composes its tie-aware amin and amax derivatives.
AMINMAX_BACKWARD_CASES = [
    ((), None, False),
    ((31, 47), 1, False),
    ((17, 33, 5), 0, False),
    ((8, 9, 10), -2, True),
]


def _aminmax_reference(inp, dim=None, keepdim=False):
    return (
        torch.amin(inp, dim=dim, keepdim=keepdim),
        torch.amax(inp, dim=dim, keepdim=keepdim),
    )


@pytest.mark.aminmax_backward
@pytest.mark.parametrize("shape, dim, keepdim", AMINMAX_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_aminmax_backward(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    min_output, max_output = flag_gems.aminmax(inp, dim=dim, keepdim=keepdim)
    grad_min = torch.randn_like(min_output)
    grad_max = torch.randn_like(max_output)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_min, ref_max = _aminmax_reference(ref_inp, dim=dim, keepdim=keepdim)
    expected = torch.autograd.grad(
        (ref_min, ref_max),
        ref_inp,
        (utils.to_reference(grad_min, False), utils.to_reference(grad_max, False)),
    )[0]
    result = flag_gems.aminmax_backward(
        grad_min,
        grad_max,
        inp,
        min_output,
        max_output,
        dim=dim,
        keepdim=keepdim,
    )
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.aminmax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_aminmax_backward_ties(dtype):
    inp = torch.tensor(
        [[-2.0, -2.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    min_output, max_output = flag_gems.aminmax(inp, dim=1)
    grad_min = torch.tensor([2.0, -4.0], dtype=dtype, device=flag_gems.device)
    grad_max = torch.tensor([6.0, 8.0], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_min, ref_max = _aminmax_reference(ref_inp, dim=1)
    expected = torch.autograd.grad(
        (ref_min, ref_max),
        ref_inp,
        (utils.to_reference(grad_min, False), utils.to_reference(grad_max, False)),
    )[0]
    result = flag_gems.aminmax_backward(
        grad_min, grad_max, inp, min_output, max_output, dim=1
    )
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.aminmax_backward
@pytest.mark.parametrize("use_min", [False, True])
def test_aminmax_backward_selective_gradient(use_min):
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn((31, 47), dtype=dtype, device=flag_gems.device)
    min_output, max_output = flag_gems.aminmax(inp, dim=1)
    selected_grad = torch.randn_like(min_output)
    zero_grad = torch.zeros_like(selected_grad)
    grad_min, grad_max = (
        (selected_grad, zero_grad) if use_min else (zero_grad, selected_grad)
    )
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_output = torch.amin(ref_inp, dim=1) if use_min else torch.amax(ref_inp, dim=1)
    expected = torch.autograd.grad(
        ref_output, ref_inp, utils.to_reference(selected_grad, False)
    )[0]
    result = flag_gems.aminmax_backward(
        grad_min, grad_max, inp, min_output, max_output, dim=1
    )
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.aminmax_backward
def test_aminmax_backward_registered_dispatch():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn(
        (31, 47), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad_min = torch.randn((31,), dtype=dtype, device=flag_gems.device)
    grad_max = torch.randn((31,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_min, ref_max = _aminmax_reference(ref_inp, dim=1)
    expected = torch.autograd.grad(
        (ref_min, ref_max),
        ref_inp,
        (utils.to_reference(grad_min, False), utils.to_reference(grad_max, False)),
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["aminmax"])
        min_output, max_output = torch.aminmax(inp, dim=1)
        assert "AminmaxFunction" in type(min_output.grad_fn).__name__
        result = torch.autograd.grad(
            (min_output, max_output), inp, (grad_min, grad_max)
        )[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.aminmax_backward
def test_aminmax_backward_noncontiguous_and_nan():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn((13, 17, 19), dtype=dtype, device=flag_gems.device).transpose(
        0, 2
    )
    inp[0, 0, 0] = float("nan")
    min_output, max_output = flag_gems.aminmax(inp, dim=2)
    grad_min = torch.randn_like(min_output)
    grad_max = torch.randn_like(max_output)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    ref_min, ref_max = _aminmax_reference(ref_inp, dim=2)
    expected = torch.autograd.grad(
        (ref_min, ref_max),
        ref_inp,
        (utils.to_reference(grad_min, False), utils.to_reference(grad_max, False)),
    )[0]
    result = flag_gems.aminmax_backward(
        grad_min, grad_max, inp, min_output, max_output, dim=2
    )
    utils.gems_assert_close(result, expected, dtype, equal_nan=True)


@pytest.mark.aminmax_backward
def test_aminmax_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((64, 512), dtype=torch.float16, device=flag_gems.device)
    min_output, max_output = flag_gems.aminmax(inp, dim=1)
    grad_min = torch.randn_like(min_output)
    grad_max = torch.randn_like(max_output)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.aminmax_backward(
            grad_min, grad_max, inp, min_output, max_output, dim=1
        )
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "aminmax" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
