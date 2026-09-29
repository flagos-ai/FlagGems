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

from . import test_utils as tu

pytestmark = pytest.mark.cudnn_affine_grid_generator

# The reference is the ATen cuDNN entry point itself; the generic affine_grid
# implementations disagree with it on unit extents, so they are never used.

# The kernel supports half, single and double precision. bfloat16, the integer
# types and float8 are rejected by the cuDNN dtype table.
FP64_SUPPORTED = bool(getattr(flag_gems.runtime.device, "support_fp64", False))
SUPPORTED_DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if FP64_SUPPORTED else []
)

# Rows are (N, C, H, W) sizes plus the theta layout. The size is fixed-rank, so
# the spec shape levels are expressed as 4-dim grid sizes.
#   * Unit H/W extents are valid and produce the kernel's clamped -1 coordinate.
#   * C is ignored metadata: zero, negative and large values are all valid.
#   * A row given a non-contiguous theta checks a strided (N, 2, 3) operand.
GRID_ROWS = [
    ((1, 3, 1, 1), "contiguous"),
    ((1, 3, 1, 7), "contiguous"),
    ((1, 3, 256, 1), "contiguous"),
    ((2, 3, 512, 512), "contiguous"),
    ((2, 3, 1024, 1024), "contiguous"),
    ((20, 3, 320, 15), "contiguous"),
    ((16, 128, 64, 60), "contiguous"),
    ((16, 7, 57, 32), "contiguous"),
    ((4, 0, 8, 5), "contiguous"),
    ((4, 1, 8, 5), "contiguous"),
    ((4, -1, 8, 5), "contiguous"),
    ((4, -3, 8, 5), "contiguous"),
    ((4, 32, 8, 5), "contiguous"),
    ((4, 1000, 8, 5), "contiguous"),
    ((4, 3, 8, 5), "strided"),
]

QUICK_GRID_ROWS = [((2, 3, 19, 7), "contiguous")]
GRID_CASES = tu.selected_cases(GRID_ROWS, quick=QUICK_GRID_ROWS)


def _make_theta(n, dtype, value_range, layout):
    """Build the (N, 2, 3) theta operand; a strided row returns a view."""
    if layout == "strided":
        return tu.make_input(dtype, (n, 2, 6), value_range)[:, :, ::2]
    return tu.make_input(dtype, (n, 2, 3), value_range)


@pytest.mark.parametrize("size,layout", GRID_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_cudnn_affine_grid_generator(size, layout, dtype, value_range):
    theta = _make_theta(size[0], dtype, value_range, layout)
    ref_theta = tu.to_reference(theta)

    ref_out = torch.ops.aten.cudnn_affine_grid_generator(ref_theta, *size)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, *size)

    assert res_out.device == theta.device
    tu.assert_result_close(res_out, ref_out)


OUT_SIZES = tu.selected_cases(
    [(1, 3, 1, 7), (2, 3, 16, 5), (20, 3, 32, 15)],
    quick=[],
)


@pytest.mark.parametrize("size", OUT_SIZES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cudnn_affine_grid_generator_out(size, dtype):
    theta = tu.make_input(dtype, (size[0], 2, 3), ["-1", "1"])
    ref_theta = tu.to_reference(theta)
    ref_out = torch.empty(
        size[0], size[2], size[3], 2, dtype=ref_theta.dtype, device=ref_theta.device
    )
    torch.ops.aten.cudnn_affine_grid_generator.out(ref_theta, *size, out=ref_out)

    out = torch.empty(
        size[0], size[2], size[3], 2, dtype=theta.dtype, device=theta.device
    )
    res_out = flag_gems.cudnn_affine_grid_generator(theta, *size, out=out)

    assert res_out is out
    tu.assert_result_close(res_out, ref_out)


BACKWARD_CASES = tu.selected_cases(
    [
        (size, dtype)
        for size in [(2, 3, 1, 5), (2, 3, 16, 8)]
        for dtype in SUPPORTED_DTYPES
    ],
    quick=[],
)


@pytest.mark.parametrize("size,dtype", BACKWARD_CASES)
def test_cudnn_affine_grid_generator_backward(size, dtype):
    n, _, h, w = size
    theta = tu.make_input(dtype, (n, 2, 3), ["-1", "1"]).requires_grad_(True)
    ref_theta = tu.to_reference(theta)
    upstream = tu.make_input(dtype, (n, h, w, 2), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    res_out = flag_gems.cudnn_affine_grid_generator(theta, *size)
    ref_out = torch.ops.aten.cudnn_affine_grid_generator(ref_theta, *size)
    tu.assert_result_close(res_out, ref_out)

    (res_grad,) = torch.autograd.grad(res_out, theta, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_theta, grad_outputs=ref_upstream)
    tu.assert_result_close(res_grad, ref_grad)


# nan / inf / mixed payloads per supported dtype. theta holds 6 values while the
# shared generator emits 5, so the pattern repeats to keep every special value.
SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(SUPPORTED_DTYPES),
    quick=[],
)


def _special_theta(dtype, scenario):
    return tu.make_special_input(dtype, scenario).repeat(2)[:6].reshape(1, 2, 3)


@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_cudnn_affine_grid_generator_special_values(dtype, scenario):
    theta = _special_theta(dtype, scenario)
    ref_theta = tu.to_reference(theta)

    ref_out = torch.ops.aten.cudnn_affine_grid_generator(ref_theta, 1, 3, 4, 5)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, 1, 3, 4, 5)

    tu.assert_result_close(res_out, ref_out)


# N must equal theta.size(0); theta is built independently so the rejection under
# test is the operator's own consistency check, not an input-construction error.
INVALID_BATCH = [0, -1, -2, 3, 4]


@pytest.mark.parametrize("n", INVALID_BATCH)
def test_cudnn_affine_grid_generator_invalid_batch(n):
    theta = tu.make_input(torch.float32, (1, 2, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator(theta, n, 3, 4, 4)


# A zero spatial extent is rejected by the kernel, a negative one overflows the
# size computation.
INVALID_EXTENTS = [
    ((2, 3, 4, 0), "w-zero"),
    ((2, 3, 0, 4), "h-zero"),
    ((2, 3, -4, 4), "h-negative"),
    ((2, 3, 4, -4), "w-negative"),
]


@pytest.mark.parametrize(
    "size",
    [size for size, _ in INVALID_EXTENTS],
    ids=[reason for _, reason in INVALID_EXTENTS],
)
def test_cudnn_affine_grid_generator_invalid_extent(size):
    theta = tu.make_input(torch.float32, (size[0], 2, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator(theta, *size)


INVALID_THETA_SHAPES = [
    ((2, 3), "rank-2"),
    ((1, 2, 3, 1), "rank-4"),
    ((2, 2, 4), "wrong-trailing-dims"),
]


@pytest.mark.parametrize(
    "theta_shape",
    [shape for shape, _ in INVALID_THETA_SHAPES],
    ids=[reason for _, reason in INVALID_THETA_SHAPES],
)
def test_cudnn_affine_grid_generator_invalid_theta(theta_shape):
    theta = tu.make_input(torch.float32, theta_shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator(theta, 2, 3, 4, 5)


# Dtypes the cuDNN dtype table rejects. The expectation is bound to the CUDA
# backend statically; other backends do not provide this cuDNN entry point, so
# their error behaviour is not asserted here.
CUDA_BACKEND = flag_gems.vendor_name == "nvidia"
UNSUPPORTED_DTYPES = (
    [
        torch.bfloat16,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    if CUDA_BACKEND
    else []
)


@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_cudnn_affine_grid_generator_unsupported_dtype(dtype):
    theta = torch.ones(2, 2, 3, dtype=torch.float32, device=flag_gems.device).to(dtype)
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator(theta, 2, 3, 4, 5)


# The out buffer must have theta's dtype; only supported dtypes are allocated.
OUT_DTYPE_MISMATCH = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
    (torch.float32, torch.int32),
]


@pytest.mark.parametrize("theta_dtype,out_dtype", OUT_DTYPE_MISMATCH)
def test_cudnn_affine_grid_generator_out_dtype_mismatch(theta_dtype, out_dtype):
    size = (1, 3, 4, 5)
    theta = tu.make_input(theta_dtype, (1, 2, 3), ["-1", "1"])
    out = torch.empty(1, 4, 5, 2, dtype=out_dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator(theta, *size, out=out)
