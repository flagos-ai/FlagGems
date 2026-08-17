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
from packaging import version

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ["ascend"],
    reason="https://github.com/flagos-ai/FlagGems/issues/3387",
)

if QUICK_MODE:
    SCALED_MM_SHAPES = [(16, 16, 16)]
else:
    SCALED_MM_SHAPES = [
        (16, 16, 16),
        (17, 31, 32),
        (64, 48, 80),
    ]


def _float8_dtypes():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 89:
        return []
    return [torch.float8_e4m3fn] if hasattr(torch, "float8_e4m3fn") else []


def _scaled_mm_cases():
    cases = [
        (torch.float16, None, "scalar", False),
        (torch.float16, torch.float32, "rowwise_2d", True),
        (torch.float32, torch.float16, "rowwise_1d", True),
    ]
    if flag_gems.runtime.device.support_bf16:
        cases.append((torch.bfloat16, torch.float32, "rowwise_2d", False))
    for dtype in _float8_dtypes():
        cases.extend(
            [
                (dtype, None, "scalar", False),
                (dtype, torch.float16, "rowwise_1d", True),
                (dtype, torch.float32, "rowwise_2d", True),
            ]
        )
    return cases


def _case_id(case):
    dtype, out_dtype, scale_mode, use_bias = case
    out_name = "default" if out_dtype is None else str(out_dtype).split(".")[-1]
    return f"{str(dtype).split('.')[-1]}-{out_name}-{scale_mode}-bias_{use_bias}"


def _is_float8(dtype):
    return dtype in _float8_dtypes()


def _make_matrix(shape, dtype):
    base = torch.randn(shape, dtype=torch.float32, device=flag_gems.device) * 0.25
    try:
        return base.to(dtype)
    except RuntimeError as exc:
        pytest.skip(f"{dtype} is not supported on {flag_gems.device}: {exc}")


def _make_scales(rows, cols, mode):
    if mode == "scalar":
        return (
            torch.tensor([0.75], dtype=torch.float32, device=flag_gems.device),
            torch.tensor([1.25], dtype=torch.float32, device=flag_gems.device),
        )

    scale_a = torch.linspace(0.75, 1.25, rows, device=flag_gems.device)
    scale_b = torch.linspace(1.25, 0.75, cols, device=flag_gems.device)
    if mode == "rowwise_2d":
        return scale_a.reshape(rows, 1), scale_b.reshape(1, cols)

    return scale_a, scale_b


def _scale_for_output(scale, rows, cols, is_left_scale):
    if scale.numel() == 1:
        return scale
    if scale.ndim == 1:
        if is_left_scale and scale.shape[0] == rows:
            return scale.reshape(rows, 1)
        if not is_left_scale and scale.shape[0] == cols:
            return scale.reshape(1, cols)
    return scale


def _reference_scaled_mm(mat1, mat2, scale_a, scale_b, bias, out_dtype):
    rows = mat1.shape[0]
    cols = mat2.shape[1]
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_scale_a = utils.to_reference(scale_a, True)
    ref_scale_b = utils.to_reference(scale_b, True)

    ref = ref_mat1.mm(ref_mat2)
    ref = ref * _scale_for_output(ref_scale_a, rows, cols, True)
    ref = ref * _scale_for_output(ref_scale_b, rows, cols, False)
    if bias is not None:
        ref = ref + utils.to_reference(bias, True)
    return ref.to(out_dtype or mat1.dtype)


def _assert_scaled_mm_close(res, ref, dtype, reduce_dim):
    if _is_float8(dtype):
        res = res.float().cpu() if utils.TO_CPU else res.float()
        ref = ref.float() if utils.TO_CPU else ref.to(res.device).float()
        torch.testing.assert_close(res, ref, atol=1.25e-1, rtol=5e-1)
        return
    ref = ref if utils.TO_CPU else ref.to(flag_gems.device)
    utils.gems_assert_close(res, ref, dtype, reduce_dim=reduce_dim)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.5"),
    reason="aten._scaled_mm is unavailable before torch 2.5",
)
@pytest.mark.scaled_mm
@pytest.mark.parametrize("M, N, K", SCALED_MM_SHAPES)
@pytest.mark.parametrize("case", _scaled_mm_cases(), ids=_case_id)
def test_scaled_mm(M, N, K, case):
    dtype, out_dtype, scale_mode, use_bias = case
    mat1 = _make_matrix((M, K), dtype)
    mat2 = _make_matrix((K, N), dtype)
    scale_a, scale_b = _make_scales(M, N, scale_mode)
    bias = None
    if use_bias:
        bias_dtype = out_dtype or (torch.float32 if _is_float8(dtype) else dtype)
        bias = _make_matrix((N,), bias_dtype)

    ref = _reference_scaled_mm(mat1, mat2, scale_a, scale_b, bias, out_dtype)
    scale_result = torch.tensor([2.0], device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch._scaled_mm(
            mat1,
            mat2,
            scale_a,
            scale_b,
            bias=bias,
            scale_result=scale_result,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

    target_dtype = out_dtype or dtype
    _assert_scaled_mm_close(res, ref, target_dtype, reduce_dim=K)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.5"),
    reason="aten._scaled_mm.out is unavailable before torch 2.5",
)
@pytest.mark.scaled_mm_out
@pytest.mark.parametrize(
    "case",
    [
        (torch.float16, torch.float32, "rowwise_2d", True),
        *_scaled_mm_cases()[-1:],
    ],
    ids=_case_id,
)
def test_scaled_mm_out(case):
    dtype, out_dtype, scale_mode, use_bias = case
    M, N, K = SCALED_MM_SHAPES[0]
    mat1 = _make_matrix((M, K), dtype)
    mat2 = _make_matrix((K, N), dtype)
    scale_a, scale_b = _make_scales(M, N, scale_mode)
    bias = _make_matrix((1, N), out_dtype) if use_bias else None
    target_dtype = out_dtype or dtype
    out = torch.empty((M, N), dtype=target_dtype, device=flag_gems.device)

    ref = _reference_scaled_mm(mat1, mat2, scale_a, scale_b, bias, out_dtype)
    with flag_gems.use_gems():
        ret = torch.ops.aten._scaled_mm.out(
            mat1,
            mat2,
            scale_a,
            scale_b,
            bias=bias,
            scale_result=None,
            out_dtype=out_dtype,
            use_fast_accum=False,
            out=out,
        )

    assert ret is out
    _assert_scaled_mm_close(out, ref, target_dtype, reduce_dim=K)
