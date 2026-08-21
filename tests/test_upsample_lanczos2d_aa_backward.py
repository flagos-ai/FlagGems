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
from flag_gems.ops._upsample_lanczos2d_aa_backward import (
    _upsample_lanczos2d_aa_backward,
    _upsample_lanczos2d_aa_backward_out,
)

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

lanczos_backward_module = importlib.import_module(
    "flag_gems.ops._upsample_lanczos2d_aa_backward"
)

HAS_NATIVE_LANCZOS_BACKWARD = hasattr(torch.ops.aten, "_upsample_lanczos2d_aa_backward")

if QUICK_MODE:
    CASES = [(1, 2, 7, 9, 11, 13, False, None, None)]
    DTYPES = [torch.float32]
else:
    # Covers up/down sampling, align_corners, explicit scales, and both the
    # fused gather and two-pass precomputed-weight Triton paths.
    CASES = [
        (1, 2, 7, 9, 11, 13, False, None, None),
        (2, 3, 17, 19, 9, 31, False, None, None),
        (1, 1, 13, 15, 5, 7, True, None, None),
        (2, 4, 16, 20, 12, 34, False, 0.75, 1.7),
    ]
    DTYPES = [torch.float32]


def _scale(input_size, output_size, align_corners, explicit_scale):
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if explicit_scale is not None and explicit_scale > 0:
        return 1.0 / explicit_scale
    return input_size / output_size


def _weight_matrix(input_size, output_size, align_corners, explicit_scale, dtype):
    scale = _scale(input_size, output_size, align_corners, explicit_scale)
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    weight = torch.zeros((output_size, input_size), dtype=dtype)
    for output_index in range(output_size):
        center = scale * (output_index + 0.5)
        index_min = max(math.floor(center - support + 0.5), 0)
        index_max = min(math.floor(center + support + 0.5), input_size)
        values = []
        for input_index in range(index_min, index_max):
            x = abs((input_index - center + 0.5) * invscale)
            if x == 0.0:
                value = 1.0
            elif x < 3.0:
                pix = math.pi * x
                value = (math.sin(pix) / pix) * (math.sin(pix / 3.0) / (pix / 3.0))
            else:
                value = 0.0
            values.append(value)
        total = sum(values)
        for input_index, value in zip(range(index_min, index_max), values):
            weight[output_index, input_index] = value / total
    return weight


def _reference(grad_output, input_size, align_corners, scales_h, scales_w):
    _, _, input_h, input_w = input_size
    output_h, output_w = grad_output.shape[-2:]
    compute_dtype = (
        torch.float64 if grad_output.dtype == torch.float64 else torch.float32
    )
    weight_h = _weight_matrix(input_h, output_h, align_corners, scales_h, compute_dtype)
    weight_w = _weight_matrix(input_w, output_w, align_corners, scales_w, compute_dtype)
    return torch.einsum(
        "oh,ncow,wi->nchi", weight_h, grad_output.to(compute_dtype), weight_w
    ).to(grad_output.dtype)


def _disable_gemm_path(monkeypatch):
    # Accuracy tests should exercise the Triton kernels directly.  The GEMM
    # optimization uses torch.mm, whose FP32 precision depends on the runner's
    # global TF32 setting (enabled by default on the A100 CI runner).
    monkeypatch.setattr(
        lanczos_backward_module, "_should_use_gemm_path", lambda *args: False
    )


@pytest.mark.upsample_lanczos2d_aa_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n,c,input_h,input_w,output_h,output_w,align_corners,scales_h,scales_w",
    CASES,
)
def test_upsample_lanczos2d_aa_backward(
    monkeypatch,
    dtype,
    n,
    c,
    input_h,
    input_w,
    output_h,
    output_w,
    align_corners,
    scales_h,
    scales_w,
):
    _disable_gemm_path(monkeypatch)
    if not QUICK_MODE:
        # Keep the semantic matrix CI-safe: the separately tested fused path
        # has large shape-dependent static loops and is expensive to compile.
        monkeypatch.setattr(lanczos_backward_module, "_FUSE_THRESHOLD", 0)
    grad = torch.randn((n, c, output_h, output_w), dtype=dtype, device=flag_gems.device)
    grad_cpu = utils.to_reference(grad).cpu()
    input_size = (n, c, input_h, input_w)
    reference = _reference(grad_cpu, input_size, align_corners, scales_h, scales_w)
    result = _upsample_lanczos2d_aa_backward(
        grad,
        (output_h, output_w),
        input_size,
        align_corners,
        scales_h,
        scales_w,
    ).cpu()

    tolerance = {
        torch.float16: 3e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 2e-4,
        torch.float64: 1e-12,
    }[dtype]
    torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.upsample_lanczos2d_aa_backward
@pytest.mark.parametrize(
    "dtype,tolerance",
    [
        (torch.float16, 3e-3),
        (torch.bfloat16, 2e-2),
        (torch.float64, 1e-12),
    ],
)
def test_upsample_lanczos2d_aa_backward_dtypes(monkeypatch, dtype, tolerance):
    _disable_gemm_path(monkeypatch)
    monkeypatch.setattr(lanczos_backward_module, "_FUSE_THRESHOLD", 0)
    grad = torch.randn((1, 2, 11, 13), dtype=dtype, device=flag_gems.device)
    grad_cpu = utils.to_reference(grad).cpu()
    input_size = (1, 2, 7, 9)
    result = _upsample_lanczos2d_aa_backward(grad, (11, 13), input_size, False).cpu()
    reference = _reference(grad_cpu, input_size, False, None, None)
    torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.upsample_lanczos2d_aa_backward
def test_upsample_lanczos2d_aa_backward_precomputed_path(monkeypatch):
    # Force the two-pass path so CI compiles and validates its independent
    # precomputed-weight kernels in addition to the fused gather path.
    _disable_gemm_path(monkeypatch)
    monkeypatch.setattr(lanczos_backward_module, "_FUSE_THRESHOLD", 0)
    grad = torch.randn((2, 3, 9, 31), device=flag_gems.device)
    grad_cpu = utils.to_reference(grad).cpu()
    input_size = (2, 3, 17, 19)
    result = _upsample_lanczos2d_aa_backward(grad, (9, 31), input_size, False).cpu()
    reference = _reference(grad_cpu, input_size, False, None, None)
    torch.testing.assert_close(result, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.upsample_lanczos2d_aa_backward_grad_input
def test_upsample_lanczos2d_aa_backward_out(monkeypatch):
    _disable_gemm_path(monkeypatch)
    grad = torch.randn((1, 2, 11, 13), device=flag_gems.device)
    grad_cpu = utils.to_reference(grad).cpu()
    output = torch.empty(0, device=flag_gems.device)
    result = _upsample_lanczos2d_aa_backward_out(
        grad, (11, 13), (1, 2, 7, 9), False, grad_input=output
    )
    assert result is output
    reference = _reference(grad_cpu, (1, 2, 7, 9), False, None, None)
    torch.testing.assert_close(result.cpu(), reference, rtol=2e-4, atol=2e-4)


@pytest.mark.upsample_lanczos2d_aa_backward
def test_upsample_lanczos2d_aa_backward_noncontiguous_grad(monkeypatch):
    _disable_gemm_path(monkeypatch)
    grad = torch.randn((1, 2, 13, 11), device=flag_gems.device).transpose(-1, -2)
    assert not grad.is_contiguous()
    grad_cpu = utils.to_reference(grad).cpu()
    input_size = (1, 2, 7, 9)
    result = _upsample_lanczos2d_aa_backward(grad, (11, 13), input_size, False).cpu()
    reference = _reference(grad_cpu, input_size, False, None, None)
    torch.testing.assert_close(result, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.upsample_lanczos2d_aa_backward
@pytest.mark.skipif(
    not HAS_NATIVE_LANCZOS_BACKWARD,
    reason="ATen Lanczos schema was added after the local PyTorch 2.9 build",
)
def test_upsample_lanczos2d_aa_backward_dispatch():
    grad = torch.randn((1, 2, 11, 13), device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_lanczos2d_aa_backward(
            grad, (11, 13), (1, 2, 7, 9), False, None, None
        )
    direct = _upsample_lanczos2d_aa_backward(grad, (11, 13), (1, 2, 7, 9), False)
    torch.testing.assert_close(result, direct)
