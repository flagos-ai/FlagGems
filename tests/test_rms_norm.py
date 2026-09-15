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

import numpy as np
import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.rms_norm import (
    _DW_COL_BLOCK_SIZE,
    _DW_ROW_BLOCK_SIZE,
    _DW_TARGET_LAYOUT,
    _DW_TLE_NUM_WARPS,
    _dw_tle_available,
    rms_norm_grad_dw_kernel,
    rms_norm_grad_dw_kernel_tle,
)
from flag_gems.utils.triton_version_utils import HAS_TLE

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
GROUP_SIZE = 128


def _cuda_fp8_e4m3fn_available():
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    # PPU can store and cast e4m3fn even though it reports sm_80.
    if flag_gems.vendor_name == "thead":
        return True
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _quantize_int8_weight(weight, group_size=GROUP_SIZE):
    grouped_weight = weight.float().reshape(-1, group_size)
    scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    weight_q = (
        (grouped_weight / scale)
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .reshape_as(weight)
        .contiguous()
    )
    return weight_q, scale.squeeze(-1).to(weight.dtype).contiguous()


def _quantize_fp8_weight(weight, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    grouped_weight = weight.float().reshape(-1, group_size)
    scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
        min=1e-8
    )
    weight_fp8 = (
        (grouped_weight / scale)
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
        .reshape_as(weight)
        .contiguous()
    )
    return weight_fp8, scale.squeeze(-1).to(weight.dtype).contiguous()


# ---------------------------------------------------------------------------
# Standard accuracy test: flag_gems.rms_norm vs a plain PyTorch reference,
# across the repo's standard shape/dtype matrix. This is the primary
# correctness gate for the op regardless of which internal kernel path
# (baseline or TLE) dispatch selects.
# ---------------------------------------------------------------------------


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


# ---------------------------------------------------------------------------
# TLE-specific tests below. These are additive to test_rms_norm above: they
# isolate the dw-reduction kernel's set_layout (TLE) path and verify it
# against the baseline (non-TLE) kernel, and separately verify the full
# rms_norm autograd path still matches PyTorch when TLE dispatch is active.
# Skipped entirely on hardware/software that doesn't support TLE.
# ---------------------------------------------------------------------------


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


_tle_skip = pytest.mark.skipif(
    not _has_tle_hw(),
    reason="requires Triton with TLE support on Hopper+ (capability >= 9)",
)


def _run_dw_kernel(kernel_fn, X, DY, INV_RMS, M, N, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    row_block_num = triton.cdiv(M, _DW_ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, _DW_COL_BLOCK_SIZE)
    DW = torch.empty((row_block_num, N), dtype=torch.float32, device=device)
    grid = (row_block_num, col_block_num)

    kernel_fn[grid](
        X,
        DY,
        INV_RMS,
        DW,
        N,
        1,
        N,
        1,
        M,
        N,
        _DW_ROW_BLOCK_SIZE,
        _DW_COL_BLOCK_SIZE,
        **extra_kwargs,
    )
    return torch.sum(DW, dim=0, dtype=torch.float32)


def _make_dw_inputs(M, N, dtype, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(M, N, dtype=dtype, device=device)
    DY = torch.randn(M, N, dtype=dtype, device=device)
    INV_RMS = torch.rand(M, dtype=torch.float32, device=device) + 0.5
    return X, DY, INV_RMS


DW_SHAPES = [
    (16, 256),
    (1024, 4096),
    (1000, 4096),
    (1024, 4000),
    (17, 300),
    (4096, 4096),
]


@_tle_skip
@pytest.mark.rms_norm
@pytest.mark.parametrize("M,N", DW_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dw_tle_matches_baseline(M, N, dtype):
    X, DY, INV_RMS = _make_dw_inputs(M, N, dtype)

    dw_base = _run_dw_kernel(rms_norm_grad_dw_kernel, X, DY, INV_RMS, M, N)
    dw_tle = _run_dw_kernel(
        rms_norm_grad_dw_kernel_tle,
        X,
        DY,
        INV_RMS,
        M,
        N,
        extra_kwargs={
            "TARGET_LAYOUT": _DW_TARGET_LAYOUT,
            "num_warps": _DW_TLE_NUM_WARPS,
        },
    )

    # Compare TLE kernel output against baseline kernel output (both float32).
    utils.gems_assert_close(dw_tle, dw_base, dtype)


@_tle_skip
@pytest.mark.rms_norm
def test_dw_tle_available_reflects_hardware():
    x_cuda = torch.zeros(1, device=device)
    assert _dw_tle_available(x_cuda) is True

    x_cpu = torch.zeros(1, device="cpu")
    assert _dw_tle_available(x_cpu) is False


def _torch_rms_norm_simple(x, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
    hidden_states = hidden_states.to(x.dtype)
    return weight * hidden_states


@_tle_skip
@pytest.mark.rms_norm
@pytest.mark.parametrize("M,N", [(1024, 4096), (1000, 4096), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_end_to_end_with_tle_dispatch(M, N, dtype):
    torch.manual_seed(0)
    eps = 1e-5

    inp = torch.randn(M, N, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn(M, N, dtype=dtype, device=device)

    ref_inp = inp.detach().clone().float().requires_grad_()
    ref_weight = weight.detach().clone().float().requires_grad_()

    res_out = flag_gems.rms_norm(inp, [N], weight=weight, eps=eps)
    ref_out = _torch_rms_norm_simple(ref_inp, ref_weight, eps).to(dtype)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), grad_out)
    ref_grad, ref_weight_grad_f32 = torch.autograd.grad(
        ref_out.float(), (ref_inp, ref_weight), grad_out.float()
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad_f32, dtype, reduce_dim=N)


# ---------------------------------------------------------------------------
# W8A16 tests (merged from upstream/master): a shared helper drives both the
# FP8 and INT8 weight-quantized variants of rms_norm.
# ---------------------------------------------------------------------------
@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "cudagraph"])
@pytest.mark.skipif(
    flag_gems.vendor_name != "thead" or not _cuda_fp8_e4m3fn_available(),
    reason="Regression test for THead W8A16 weight dequantization",
)
def test_rms_norm_w8a16_fp8_weight_updates(dtype, capture):
    n = 4096
    inp = torch.randn((512, n), device=flag_gems.device, dtype=dtype)
    weight_q = torch.ones(n, device=inp.device, dtype=dtype).to(FP8_DTYPE)
    weight_scale = torch.ones(n // GROUP_SIZE, device=inp.device, dtype=dtype)

    def run():
        return flag_gems.rms_norm_w8a16_fp8(
            inp, (n,), weight_q, weight_scale, eps=1e-5, group_size=GROUP_SIZE
        )

    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res_out = run()

    # Keep storage addresses fixed while changing each input independently.
    # A warmed cache must not hide updates in eager execution or graph replay.
    for update in ("initial", "weight", "scale"):
        if update == "weight":
            weight_q.copy_((weight_q.float() * 2).to(FP8_DTYPE))
        elif update == "scale":
            weight_scale.mul_(2)
        if capture:
            graph.replay()
        else:
            res_out = run()
        dequant_weight = (
            weight_q.float() * weight_scale.float().repeat_interleave(GROUP_SIZE)
        ).to(dtype)
        ref_out = torch.nn.functional.rms_norm(
            utils.to_reference(inp),
            (n,),
            utils.to_reference(dequant_weight),
            eps=1e-5,
        )
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m,n,group_size",
    [(2, 256, 128), (2, 384, 128), (2, 32768, 64), (2, 33024, 128), (513, 4096, 128)],
)
@pytest.mark.skipif(
    flag_gems.vendor_name != "thead" or not _cuda_fp8_e4m3fn_available(),
    reason="THead E4M3FN byte decoding across all kernel paths",
)
def test_rms_norm_w8a16_fp8_encodings(dtype, m, n, group_size):
    # Cover all 256 encodings, including signed zero, subnormals, and NaNs.
    bits = (torch.arange(n, device=flag_gems.device) % 256).to(torch.uint8)
    if n == 384:
        bits = bits.repeat_interleave(2)[::2]
    weight_q = bits.view(FP8_DTYPE)
    scales = torch.linspace(
        0.001, 0.01, n // group_size, device=flag_gems.device, dtype=dtype
    )
    if n == 384:
        # Strided input weights/scales were accepted by the unfused path.
        scales = scales.repeat_interleave(2)[::2]
    inp = torch.ones((m, n), device=flag_gems.device, dtype=dtype)
    result = flag_gems.rms_norm_w8a16_fp8(
        inp, (n,), weight_q, scales, eps=0.0, group_size=group_size
    )
    expected = (
        (weight_q.float().reshape(-1, group_size) * scales.float()[:, None])
        .flatten()
        .to(dtype)
    )
    # All-one inputs with eps=0 normalize to exactly one, so the public
    # operator exposes the decoded/scaled weights without normalization error.
    torch.testing.assert_close(
        result, expected.expand_as(inp), rtol=0, atol=0, equal_nan=True
    )
    zero = expected == 0
    assert torch.equal(
        torch.signbit(result[:, zero]), torch.signbit(expected[zero]).expand(m, -1)
    )


W8A16_SHAPES = [
    (1, 4096),
    (128, 4096),
    (512, 4096),
    (64, 8192),
    (1, 16384),
    (1, 32768),
]


def _run_rms_norm_w8a16_test(shape, quantize_weight, op):
    dtype = torch.bfloat16
    m, n = shape
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, (m, n)).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, (n,)).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    weight_q, weight_scale = quantize_weight(weight)
    dequant_weight = (
        (weight_q.float().reshape(-1, GROUP_SIZE) * weight_scale.float().unsqueeze(-1))
        .reshape_as(weight)
        .to(dtype)
    )

    eps = 1e-5
    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(dequant_weight)
    ref_out = torch.nn.functional.rms_norm(ref_inp, (n,), ref_weight, eps=eps)
    res_out = op(
        inp,
        (n,),
        weight_q,
        weight_scale,
        eps=eps,
        group_size=GROUP_SIZE,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.parametrize("shape", W8A16_SHAPES)
@pytest.mark.skipif(
    not _cuda_fp8_e4m3fn_available(),
    reason="RMSNorm W8A16 FP8 requires CUDA float8_e4m3fn",
)
def test_rms_norm_w8a16_fp8(shape):
    _run_rms_norm_w8a16_test(shape, _quantize_fp8_weight, flag_gems.rms_norm_w8a16_fp8)


@pytest.mark.rms_norm_w8a16_int8
@pytest.mark.parametrize("shape", W8A16_SHAPES)
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="RMSNorm W8A16 INT8 is only available on Ascend",
)
def test_rms_norm_w8a16_int8(shape):
    _run_rms_norm_w8a16_test(
        shape, _quantize_int8_weight, flag_gems.rms_norm_w8a16_int8
    )
