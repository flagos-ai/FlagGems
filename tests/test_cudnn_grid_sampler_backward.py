# Copyright 2026, The FlagOS Contributors.
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
"""Accuracy tests for ``cudnn_grid_sampler_backward``.

The reference is ``aten::cudnn_grid_sampler_backward`` itself, which is the op
this implementation replaces.  It is CUDA/cuDNN-only, so under ``--ref=cpu``
the reference is taken from ``aten::grid_sampler_2d_backward`` with the same
fixed options (bilinear, zero padding, ``align_corners=True``) instead.

cuDNN supports float16/float32/float64 for this operator; bfloat16 is rejected
by the dispatcher, so only the Triton kernel is exercised for it, against a
float64 reference.
"""

import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# cuDNN accepts float16/float32/float64 only; bfloat16 still goes through the
# Triton kernel but must be compared against a float32 reference.
GRID_SAMPLER_BACKWARD_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

# BILINEAR, ZEROS_PADDING, ALIGN_CORNERS: the fixed cuDNN configuration.
BILINEAR, ZEROS_PADDING, ALIGN_CORNERS = 0, 0, True

# (N, C, H_in, W_in, H_out, W_out) tuples for the 2D cuDNN grid sampler
# backward.  Includes degenerate 1x1 inputs, odd and non-square sizes, and
# downsampled/upsampled outputs.
SHAPES = [
    (1, 1, 4, 4, 2, 2),
    (2, 3, 8, 8, 8, 8),
    (2, 3, 9, 7, 5, 4),
    (4, 8, 16, 16, 17, 13),
    (1, 32, 32, 24, 30, 32),
    (2, 16, 56, 56, 28, 28),
    (1, 3, 1, 1, 4, 5),
    (3, 5, 2, 7, 1, 1),
]

# Grid coordinates that clearly fall outside of the input image.
OUT_OF_BOUND_SHAPES = [
    (2, 3, 10, 10, 6, 6),
    (1, 4, 3, 3, 5, 5),
]


def _supports_cudnn_backward(dtype):
    """cuDNN is the only backend that provides the reference op.

    cuDNN rejects bfloat16 for this operator, so those cases fall back to the
    equivalent public ATen op.
    """
    return (
        dtype != torch.bfloat16
        and torch.cuda.is_available()
        and flag_gems.device != "cpu"
        and torch.version.cuda is not None
        and flag_gems.vendor_name in ("nvidia", "")
    )


def _reference_grads(inp, grid, grad_output, dtype):
    """Reference backward pass computed with the replaced op in float64.

    ``aten::cudnn_grid_sampler_backward`` is the op this module replaces, so it
    is the reference whenever it is available.  Under ``--ref=cpu`` (and on
    non-NVIDIA backends) the equivalent public ATen op is used instead: the
    cuDNN configuration is ``grid_sampler_2d_backward`` with bilinear
    interpolation, zero padding and ``align_corners=True``.
    """
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    if _supports_cudnn_backward(dtype) and not utils.TO_CPU:
        ref_grad_input, ref_grad_grid = torch.ops.aten.cudnn_grid_sampler_backward(
            ref_inp, ref_grid, ref_grad_output
        )
    else:
        ref_grad_input, ref_grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            ref_grad_output,
            ref_inp,
            ref_grid,
            BILINEAR,
            ZEROS_PADDING,
            ALIGN_CORNERS,
            [True, True],
        )
    return ref_grad_input.to(dtype), ref_grad_grid.to(dtype)


def _cpu_reference_grads(inp, grid, grad_output, dtype):
    """Reference computed on the CPU in float64.

    The CUDA ``grid_sampler_2d_backward`` kernel maps non-finite grid
    coordinates to zero while the CPU kernel (and cuDNN) propagate the NaN into
    ``grad_grid``.  Non-finite cases are therefore checked against the CPU
    reference on every backend, which also keeps them running when cuDNN has no
    bfloat16 support.
    """
    ref_grad_output = utils.to_reference(grad_output, True).cpu()
    ref_inp = utils.to_reference(inp, True).cpu()
    ref_grid = utils.to_reference(grid, True).cpu()
    ref_grad_input, ref_grad_grid = torch.ops.aten.grid_sampler_2d_backward(
        ref_grad_output,
        ref_inp,
        ref_grid,
        BILINEAR,
        ZEROS_PADDING,
        ALIGN_CORNERS,
        [True, True],
    )
    # ``gems_assert_close`` expects a CPU reference in the ``--ref=cpu`` mode.
    device = "cpu" if utils.TO_CPU else inp.device
    return (
        ref_grad_input.to(dtype=dtype, device=device),
        ref_grad_grid.to(dtype=dtype, device=device),
    )


def _make_inputs(shape, dtype, grid_range, seed=None):
    N, C, H_in, W_in, H_out, W_out = shape
    if seed is not None:
        torch.manual_seed(seed)
    inp = torch.randn((N, C, H_in, W_in), dtype=dtype, device=flag_gems.device)
    grid = torch.empty(
        (N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device
    ).uniform_(*grid_range)
    grad_output = torch.randn(
        (N, C, H_out, W_out), dtype=dtype, device=flag_gems.device
    )
    return inp, grid, grad_output


def _assert_grads(
    res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, C
):
    """grad_input is a pure scatter, grad_grid reduces over the channels."""
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    # The grid gradient reduces over C channels, so fp32 rounding accumulates
    # a little more than the default 1e-4 tolerance on large-channel shapes.
    utils.gems_assert_close(
        res_grad_grid, ref_grad_grid, dtype, reduce_dim=C, atol=2e-3
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward(shape, dtype):
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("shape", OUT_OF_BOUND_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_out_of_bound(shape, dtype):
    # Grid values in [-2, 2] exercise zero padding in every direction.
    inp, grid, grad_output = _make_inputs(shape, dtype, (-2.0, 2.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_edge_coordinates(dtype):
    # Coordinates sitting exactly on the corners of the image, plus values just
    # inside the border, must reproduce cuDNN's convention.
    shape = (2, 3, 6, 5, 4, 4)
    inp, _, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=0)

    corners = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-0.999, 0.999],
            [0.0, 0.0],
            [-0.5, 0.25],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.75, -0.25],
            [0.75, 0.25],
            [-0.25, -0.75],
            [0.25, 0.75],
            [-0.001, 0.001],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    grid = corners.reshape(1, 4, 4, 2).expand(shape[0], 4, 4, 2).contiguous()

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_all_out_of_bound(dtype):
    # Every sample point is outside of the image => zero gradients everywhere.
    shape = (2, 3, 5, 5, 3, 3)
    inp, _, _ = _make_inputs(shape, dtype, (-1.0, 1.0), seed=1)
    grad_output = torch.randn(
        (shape[0], shape[1], shape[4], shape[5]), dtype=dtype, device=flag_gems.device
    )
    grid = torch.full(
        (shape[0], shape[4], shape[5], 2), 3.0, dtype=dtype, device=flag_gems.device
    )

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    # to_reference moves the expected-zero buffers to the reference device in
    # the quick-cpu CI mode.
    zero_input = utils.to_reference(torch.zeros_like(res_grad_input))
    zero_grid = utils.to_reference(torch.zeros_like(res_grad_grid))
    utils.gems_assert_close(res_grad_input, zero_input, dtype, atol=0.0)
    utils.gems_assert_close(res_grad_grid, zero_grid, dtype, atol=0.0)


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_accumulation(dtype):
    # A single sample point only scatters into four input pixels; all other
    # pixels must stay exactly zero.
    shape = (1, 2, 8, 8, 1, 1)
    inp, _, grad_output = _make_inputs(shape, dtype, (-0.5, 0.5), seed=2)
    grid = torch.zeros((1, 1, 1, 2), dtype=dtype, device=flag_gems.device)

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_cudnn_grid_sampler_backward_non_finite_grid(dtype, bad_value):
    # A non-finite grid coordinate is kept out of the index arithmetic but still
    # reaches the grad_grid component it multiplies (cuDNN propagates NaN into
    # grad_grid, and only there: grad_input stays finite).
    shape = (2, 3, 6, 5, 4, 4)
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=4)

    grid = grid.clone()
    grid[0, 0, 0, 0] = bad_value
    grid[0, 1, 1, 1] = bad_value
    grid[1, 3, 2, :] = bad_value

    ref_grad_input, ref_grad_grid = _cpu_reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    assert torch.isnan(ref_grad_grid).any(), "reference did not propagate NaN"
    assert not torch.isnan(res_grad_input).any()
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype, equal_nan=True)
    utils.gems_assert_close(
        res_grad_grid,
        ref_grad_grid,
        dtype,
        reduce_dim=shape[1],
        atol=2e-3,
        equal_nan=True,
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cudnn_grid_sampler_backward_non_finite_grad_output(dtype):
    # cuDNN drops sample points that miss the image entirely, so a NaN
    # grad_output there contributes nothing to grad_grid.  A NaN grad_output at
    # a sample point inside the image does propagate into both gradients.
    # The CPU reference propagates the NaN in both cases, so this behaviour can
    # only be asserted against cuDNN itself.
    if not _supports_cudnn_backward(dtype) or utils.TO_CPU:
        pytest.skip("the OOB NaN-grad_output semantics are cuDNN-specific")

    shape = (1, 3, 6, 6, 4, 4)
    inp, grid, grad_output = _make_inputs(shape, dtype, (-0.5, 0.5), seed=5)
    grid = grid.clone()
    grad_output = grad_output.clone()

    grid[0, 0, 0, :] = 3.0  # fully outside of the image
    grid[0, 1, 1, :] = 0.0  # centre of the image
    grad_output[0, :, 0, 0] = float("nan")
    grad_output[0, :, 1, 1] = float("nan")

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    assert not torch.isnan(ref_grad_grid[0, 0, 0]).any(), "cuDNN zeroed the OOB point"
    assert torch.isnan(ref_grad_grid[0, 1, 1]).all(), "reference dropped the NaN"
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype, equal_nan=True)
    utils.gems_assert_close(
        res_grad_grid,
        ref_grad_grid,
        dtype,
        reduce_dim=shape[1],
        atol=2e-3,
        equal_nan=True,
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4, 4, 1, 1),
        (2, 3, 8, 8, 8, 8),
        (1, 3, 1, 1, 4, 5),
    ],
)
def test_cudnn_grid_sampler_backward_float64(shape):
    dtype = torch.float64
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.5, 1.5), seed=3)

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_non_contiguous(dtype):
    # Sliced and transposed views must be handled by the contiguous copies in
    # the wrapper, not read with flat offsets.
    shape = (2, 3, 8, 6, 5, 4)
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=6)
    inp_nc = inp.transpose(2, 3).contiguous().transpose(2, 3)
    grid_nc = grid.transpose(1, 2).contiguous().transpose(1, 2)
    grad_output_nc = grad_output.transpose(2, 3).contiguous().transpose(2, 3)

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)
    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp_nc, grid_nc, grad_output_nc
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_empty(dtype):
    # A zero-sized dimension means no sample point can touch the input.
    shape = (2, 3, 5, 5, 0, 4)
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=7)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    assert res_grad_input.shape == inp.shape
    assert res_grad_grid.shape == grid.shape
    assert res_grad_input.dtype == inp.dtype
    assert res_grad_grid.dtype == grid.dtype
    assert torch.count_nonzero(res_grad_input) == 0
    assert torch.count_nonzero(res_grad_grid) == 0


@pytest.mark.cudnn_grid_sampler_backward
def test_cudnn_grid_sampler_backward_invalid_inputs():
    dtype = torch.float32
    inp = torch.randn(2, 3, 5, 5, dtype=dtype, device=flag_gems.device)
    grid = torch.zeros(2, 4, 4, 2, dtype=dtype, device=flag_gems.device)
    grad_output = torch.zeros(2, 3, 4, 4, dtype=dtype, device=flag_gems.device)

    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward(inp[0], grid, grad_output)
    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward(
            inp, torch.zeros(2, 4, 4, dtype=dtype, device=flag_gems.device), grad_output
        )
    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward(
            inp,
            torch.zeros(3, 4, 4, 2, dtype=dtype, device=flag_gems.device),
            grad_output,
        )
    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward(inp, grid, grad_output[:, :, :3])
    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward(inp, grid, grad_output.double())


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_out(shape, dtype):
    N, C, _, _, H_out, W_out = shape
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    out0 = torch.empty_like(inp)
    out1 = torch.empty((N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device)

    res_out0, res_out1 = flag_gems.cudnn_grid_sampler_backward_out(
        inp, grid, grad_output, out0, out1
    )

    assert res_out0.data_ptr() == out0.data_ptr()
    assert res_out1.data_ptr() == out1.data_ptr()
    _assert_grads(res_out0, res_out1, ref_grad_input, ref_grad_grid, dtype, C)


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
@pytest.mark.parametrize("index", [0, 1])
def test_cudnn_grid_sampler_backward_out_resize(dtype, index):
    # ATen's resize_output(): a mis-shaped out tensor is resized in place,
    # warning first unless it had zero elements.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, grad_output = _make_inputs(
        (N, C, H_in, W_in, H_out, W_out), dtype, (-1.0, 1.0), seed=8
    )
    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    out0 = torch.empty((N, C, H_in, W_in), dtype=dtype, device=flag_gems.device)
    out1 = torch.empty((N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device)
    outs = [out0, out1]
    outs[index] = torch.empty(0, dtype=dtype, device=flag_gems.device)

    with warnings.catch_warnings():
        # An empty out is resized without a deprecation warning.
        warnings.simplefilter("error", UserWarning)
        res_out0, res_out1 = flag_gems.cudnn_grid_sampler_backward_out(
            inp, grid, grad_output, outs[0], outs[1]
        )

    assert tuple(res_out0.shape) == (N, C, H_in, W_in)
    assert tuple(res_out1.shape) == (N, H_out, W_out, 2)
    _assert_grads(res_out0, res_out1, ref_grad_input, ref_grad_grid, dtype, C)


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
@pytest.mark.parametrize("index", [0, 1])
def test_cudnn_grid_sampler_backward_out_resize_warns(dtype, index):
    # A non-empty out with the wrong shape is resized but warns first, exactly
    # like ATen's resize_output().
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, grad_output = _make_inputs(
        (N, C, H_in, W_in, H_out, W_out), dtype, (-1.0, 1.0), seed=9
    )
    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    out0 = torch.empty((N, C, H_in, W_in), dtype=dtype, device=flag_gems.device)
    out1 = torch.empty((N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device)
    outs = [out0, out1]
    outs[index] = torch.empty(4, dtype=dtype, device=flag_gems.device)

    with pytest.warns(UserWarning):
        res_out0, res_out1 = flag_gems.cudnn_grid_sampler_backward_out(
            inp, grid, grad_output, outs[0], outs[1]
        )

    assert tuple(res_out0.shape) == (N, C, H_in, W_in)
    assert tuple(res_out1.shape) == (N, H_out, W_out, 2)
    _assert_grads(res_out0, res_out1, ref_grad_input, ref_grad_grid, dtype, C)


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_out_strided(dtype):
    # A strided out cannot be written with flat offsets, so the kernel stages
    # through a contiguous buffer and copies into the view.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, grad_output = _make_inputs(
        (N, C, H_in, W_in, H_out, W_out), dtype, (-1.0, 1.0), seed=10
    )
    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    out0_base = torch.empty((N, C, W_in, H_in), dtype=dtype, device=flag_gems.device)
    out0 = out0_base.transpose(2, 3)
    out1_base = torch.empty((N, W_out, H_out, 2), dtype=dtype, device=flag_gems.device)
    out1 = out1_base.transpose(1, 2)

    res_out0, res_out1 = flag_gems.cudnn_grid_sampler_backward_out(
        inp, grid, grad_output, out0, out1
    )

    assert res_out0.data_ptr() == out0.data_ptr()
    assert res_out1.data_ptr() == out1.data_ptr()
    assert not out0.is_contiguous()
    assert not out1.is_contiguous()
    _assert_grads(out0, out1, ref_grad_input, ref_grad_grid, dtype, C)


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index", [0, 1])
def test_cudnn_grid_sampler_backward_out_dtype_mismatch(dtype, index):
    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp, grid, grad_output = _make_inputs(
        (N, C, H_in, W_in, H_out, W_out), dtype, (-1.0, 1.0), seed=11
    )
    mismatched = torch.float32 if dtype != torch.float32 else torch.float64

    out0 = torch.empty((N, C, H_in, W_in), dtype=dtype, device=flag_gems.device)
    out1 = torch.empty((N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device)
    outs = [out0, out1]
    outs[index] = torch.empty(
        outs[index].shape, dtype=mismatched, device=flag_gems.device
    )

    with pytest.raises(ValueError):
        flag_gems.cudnn_grid_sampler_backward_out(
            inp, grid, grad_output, outs[0], outs[1]
        )


@pytest.mark.cudnn_grid_sampler_backward
def test_cudnn_grid_sampler_backward_dispatch(monkeypatch):
    # Calling the wrapper directly only proves the wrapper runs; this case
    # proves the dispatcher path.  ``only_enable`` is the sanctioned
    # registration API (``use_gems()`` is forbidden in tests) and registers the
    # impl the registrar resolves through ``FULL_CONFIG_BY_FUNC``.  Spying on
    # both entries proves the aten call lands in the Triton implementation
    # instead of silently falling back to cuDNN.
    if utils.TO_CPU:
        pytest.skip("cudnn_grid_sampler_backward is a device operator")

    dtype = torch.float32
    shape = (2, 3, 6, 5, 4, 4)
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=12)
    expected = flag_gems.cudnn_grid_sampler_backward(inp, grid, grad_output)

    calls = {"base": 0, "out": 0}
    real_base = flag_gems.cudnn_grid_sampler_backward
    real_out = flag_gems.cudnn_grid_sampler_backward_out

    def spy_base(input, grid, grad_output):
        calls["base"] += 1
        return real_base(input, grid, grad_output)

    def spy_out(input, grid, grad_output, out0, out1):
        calls["out"] += 1
        return real_out(input, grid, grad_output, out0, out1)

    mapping = dict(flag_gems.FULL_CONFIG_BY_FUNC)
    mapping["cudnn_grid_sampler_backward"] = [("cudnn_grid_sampler_backward", spy_base)]
    mapping["cudnn_grid_sampler_backward_out"] = [
        ("cudnn_grid_sampler_backward.out", spy_out)
    ]
    monkeypatch.setattr(flag_gems, "FULL_CONFIG_BY_FUNC", mapping)
    flag_gems.only_enable(
        include=["cudnn_grid_sampler_backward", "cudnn_grid_sampler_backward_out"]
    )

    via_dispatch = torch.ops.aten.cudnn_grid_sampler_backward(inp, grid, grad_output)
    assert calls["base"] == 1, "aten dispatch did not reach the Gems implementation"
    _assert_grads(via_dispatch[0], via_dispatch[1], *expected, dtype, shape[1])

    out0 = torch.empty_like(inp)
    out1 = torch.empty_like(grid)
    via_dispatch_out = torch.ops.aten.cudnn_grid_sampler_backward.out(
        inp, grid, grad_output, out0=out0, out1=out1
    )
    assert calls["out"] == 1, "aten dispatch did not reach the Gems .out implementation"
    assert via_dispatch_out[0].data_ptr() == out0.data_ptr()
    assert via_dispatch_out[1].data_ptr() == out1.data_ptr()
    _assert_grads(via_dispatch_out[0], via_dispatch_out[1], *expected, dtype, shape[1])
