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

import logging
from functools import lru_cache
from itertools import product

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

from .conftest import QUICK_MODE

logger = logging.getLogger(__name__)
FP8_NAMES = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
# output dtype, scaling scheme, bias. Matrix inputs are always FP8.
CONFIGS = (
    (torch.float16, "scalar", True),
    (torch.bfloat16, "scalar", True),
    (torch.float32, "scalar", False),
    (torch.bfloat16, "rowwise", True),
    (None, "scalar", False),
)


@triton.jit
def _convert_probe(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0.0).to(tl.float32)
    tl.store(Y + i, x, i < N)


@triton.jit
def _dot_probe(A, B, C):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 32)
    a = tl.load(A + m[:, None] * 32 + k[None, :]).to(tl.float16)
    b = tl.load(B + k[:, None] * 16 + n[None, :]).to(tl.float16)
    tl.store(C + m[:, None] * 16 + n[None, :], tl.dot(a, b))


def _device_key():
    return str(flag_gems.device), torch_device_fn.current_device()


def _unsupported_reason(exc):
    text = str(exc)
    # Resource exhaustion and device faults are not evidence of dtype support.
    if any(
        s in text.lower()
        for s in ("out of memory", "device-side assert", "illegal memory")
    ):
        raise exc
    if not any(
        s in text.lower()
        for s in (
            "not support",
            "doesn't support",
            "unsupported",
            "not implement",
            "cannot convert",
            "cannot cast",
            "only support",
            "has no attribute",
            "divisible",
            "expected b.dtype",
            "not_supported",
        )
    ):
        raise exc
    return f"{type(exc).__name__}: {text[-1600:]}"


@lru_cache(None)
def _dtype_reason(dtype, device_key, output=False):
    device, index = device_key
    try:
        with torch_device_fn.device(index):
            # CPU quantization avoids requiring an unrelated vendor cast operator.
            cpu = ((torch.arange(512).float() % 13 - 6) / 16).reshape(16, 32)
            a = cpu.to(dtype).to(device)
            y = torch.empty_like(cpu, device=device)
            _convert_probe[(2,)](a, y, 511, 256)
            torch_device_fn.synchronize()
            torch.testing.assert_close(
                y.cpu().flatten()[:511],
                cpu.to(dtype).float().flatten()[:511],
                rtol=0,
                atol=0,
            )
            if output:
                _convert_probe[(2,)](y, a, 511, 256)
                torch_device_fn.synchronize()
                torch.testing.assert_close(
                    a.cpu().float().flatten()[:511],
                    cpu.to(dtype).float().flatten()[:511],
                    rtol=0,
                    atol=0,
                )
            else:
                b = cpu.t().contiguous().to(dtype).to(device)
                c = torch.empty((16, 16), dtype=torch.float32, device=device)
                _dot_probe[(1,)](a, b, c)
                torch_device_fn.synchronize()
                torch.testing.assert_close(
                    c.cpu(),
                    cpu.to(dtype).float() @ cpu.t().to(dtype).float(),
                    rtol=1e-4,
                    atol=1e-4,
                )
    except Exception as exc:
        reason = _unsupported_reason(exc)
        logger.info(
            "FP8 feature check failed: device=%s dtype=%s output=%s reason=%s",
            device_key,
            dtype,
            output,
            reason,
        )
        return reason
    return None


def dtype_reason(dtype, output=False):
    return _dtype_reason(dtype, _device_key(), output)


def cases():
    dtypes = [getattr(torch, name) for name in FP8_NAMES if hasattr(torch, name)]
    return [
        (a, b, out, mode, bias)
        for a, b in product(dtypes, repeat=2)
        for out, mode, bias in CONFIGS
    ]


def case_id(case):
    return "-".join(str(x).replace("torch.", "") for x in case)


def case_reason(case):
    a, b, out, _, _ = case
    reason = dtype_reason(a) or dtype_reason(b)
    if reason is None and out is None:
        reason = dtype_reason(a, output=True)
    return reason


def make_inputs(case, shape, layout="column", fast=False, scale_result=None):
    a_dtype, b_dtype, out_dtype, mode, has_bias = case
    M, N, K = shape
    # Seed locally, without changing the application's global RNG state.
    gen = torch.Generator().manual_seed(2026)
    a = (torch.randn((M, K), generator=gen) * 0.25).to(a_dtype).to(flag_gems.device)
    b = (torch.randn((K, N), generator=gen) * 0.25).to(b_dtype)
    if layout == "column":
        b = b.t().contiguous().t()
    b = b.to(flag_gems.device)
    if mode == "scalar":
        sa = torch.tensor([0.75], device=flag_gems.device)
        sb = torch.tensor([1.25], device=flag_gems.device)
    else:
        sa = torch.linspace(0.5, 1.0, M).reshape(M, 1).to(flag_gems.device)
        sb = torch.linspace(1.0, 1.5, N).reshape(1, N).to(flag_gems.device)
    bias = None
    if has_bias:
        bias = (
            (torch.randn((N,), generator=gen) * 0.1).to(out_dtype).to(flag_gems.device)
        )
    return (a, b, sa, sb), dict(
        bias=bias, scale_result=scale_result, out_dtype=out_dtype, use_fast_accum=fast
    )


def comparison_tolerance(dtype):
    if str(dtype).removeprefix("torch.") in FP8_NAMES:
        return 0.125
    return 0.02 if dtype == torch.bfloat16 else 0.002


SHAPES = (
    [(16, 16, 16)]
    if QUICK_MODE
    else [
        (16, 16, 16),
        (32, 32, 32),
        (17, 31, 80),
        (64, 128, 128),
        (128, 128, 128),
        (512, 512, 512),
    ]
)


def _params():
    result = []
    for case in cases():
        reason = case_reason(case)
        marks = [pytest.mark.skip(reason=f"FP8 capability: {reason}")] if reason else []
        result.append(pytest.param(case, id=case_id(case), marks=marks))
    return result


CASES = _params()


def golden(args, kwargs):
    a, b, sa, sb = args
    result = a.cpu().float() @ b.cpu().float()
    result = result * sa.cpu() * sb.cpu()
    if kwargs["bias"] is not None and a.shape[1] > 0:
        result += kwargs["bias"].cpu().float()
    # ATen GPU _scaled_mm currently ignores scale_result (unlike its CPU kernel).
    return result.to(kwargs["out_dtype"] or a.dtype)


def _check(case, shape, layout, fast, use_out):
    args, kwargs = make_inputs(case, shape, layout=layout, fast=fast)
    expected = golden(args, kwargs)
    if use_out:
        out = torch.empty(
            (shape[1], shape[0]), dtype=expected.dtype, device=flag_gems.device
        ).t()
        actual = flag_gems.scaled_mm_out(*args, **kwargs, out=out)
        assert actual is out
    else:
        actual = flag_gems.scaled_mm(*args, **kwargs)
    assert actual.dtype == expected.dtype
    tol = comparison_tolerance(expected.dtype)
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=tol, atol=tol
    )


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout,fast", [("column", False), ("row", True)])
def test_scaled_mm(case, shape, layout, fast):
    _check(case, shape, layout, fast, False)


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", SHAPES)
def test_scaled_mm_out(case, shape):
    _check(case, shape, "column", False, True)


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", [(0, 16, 32), (16, 0, 32), (16, 16, 0)])
def test_scaled_mm_empty(case, shape):
    args, kwargs = make_inputs(case, shape)
    actual = flag_gems.scaled_mm(*args, **kwargs)
    torch.testing.assert_close(
        actual.cpu().float(), golden(args, kwargs).float(), rtol=0.02, atol=0.02
    )


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
def test_scaled_mm_schema(case):
    args, kwargs = make_inputs(case, (16, 16, 32))
    with pytest.raises(RuntimeError, match="Float32"):
        flag_gems.scaled_mm(args[0], args[1], args[2].half(), args[3], **kwargs)
    with pytest.raises(RuntimeError, match="float scalar"):
        flag_gems.scaled_mm(
            *args,
            **{**kwargs, "scale_result": torch.ones(2, device=flag_gems.device)},
        )
    actual = flag_gems.scaled_mm(
        *args,
        **{**kwargs, "scale_result": torch.tensor([2.0], device=flag_gems.device)},
    )
    tol = comparison_tolerance(actual.dtype)
    torch.testing.assert_close(
        actual.cpu().float(), golden(args, kwargs).float(), rtol=tol, atol=tol
    )


@pytest.mark.scaled_mm
def test_scaled_mm_fast_route(monkeypatch):
    import importlib

    module = importlib.import_module("flag_gems.ops.scaled_mm")
    case = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16, "scalar", True)
    reason = case_reason(case)
    if reason:
        pytest.skip(reason)
    calls = []
    original = module._csmm

    def tracked(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module, "_csmm", tracked)
    for layout in ("column", "row"):
        args, kwargs = make_inputs(case, (32, 32, 32), layout)
        expected_fast = (
            layout == "column"
            and flag_gems.vendor_name == "nvidia"
            and torch.cuda.get_device_capability()[0] == 9
        )
        calls.clear()
        actual = module.scaled_mm(*args, **kwargs)
        assert bool(calls) == expected_fast
        torch.testing.assert_close(
            actual.cpu().float(), golden(args, kwargs).float(), rtol=0.02, atol=0.002
        )


@pytest.mark.scaled_mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="MThreads descriptor cache"
)
def test_scaled_mm_descriptor_dtype_switch():
    dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for dtype in dtypes:
        reason = dtype_reason(dtype)
        if reason:
            pytest.skip(reason)
    # Keep shape, strides and output dtype identical while switching input
    # encodings, including switching back to a previously compiled encoding.
    for dtype in (*dtypes, dtypes[0]):
        case = (dtype, dtype, torch.bfloat16, "scalar", True)
        args, kwargs = make_inputs(case, (512, 512, 512))
        expected = golden(args, kwargs)
        actual = flag_gems.scaled_mm(*args, **kwargs)
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=0.02, atol=0.02
        )
        out = torch.empty_like(actual)
        actual = flag_gems.scaled_mm_out(*args, **kwargs, out=out)
        assert actual is out
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=0.02, atol=0.02
        )
