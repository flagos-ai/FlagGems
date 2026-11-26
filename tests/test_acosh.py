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

# ---------------------------------------------------------------------------
# acosh spec grid
#
# Value ranges follow the operator-test spec: [-1,1], [0,1], [-1,0], [0,max],
# [min,0]. acosh's domain is [1, inf), so three of the five ranges sit mostly
# outside it: x < 1 produces NaN (x == 1 produces 0). Those cases pin the
# kernel's out-of-domain path; the comparison runs with equal_nan=True.
#
# The kernel guards its domain explicitly (see src/flag_gems/ops/acosh.py),
# so the full unbounded grid agrees with ATen's NaN contract and no
# magnitude cap is needed.
#
# Shapes are the spec's REQUIRED_SHAPES (0-5 dims). dtypes cover the float
# sweep (fp16/bf16/fp32), the fp8 sweep (e4m3/e5m2, supported by the Triton
# kernel but not by ATen — the reference is computed in float32), and the
# integer sweep (int8/uint8/int32/int64, promoted to float32 exactly as
# ATen does).
# ---------------------------------------------------------------------------

ACOSH_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]

ACOSH_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),  # None = dtype max, resolved per dtype
    ("[min,0]", None, 0.0),
]

# FP8_DTYPES is arch-gated in accuracy_utils.py (sm89+): Triton's CUDA backend
# only compiles fp8e4nv pointers on sm89+.
FP8_DTYPES = utils.FP8_DTYPES

FLOAT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


def _make_float_input(dtype, shape, low, high):
    """Sample a float tensor in [low, high] with both ends pinned.

    FP8 has no uniform_ implementation, so all float inputs are sampled in
    float32 and cast; the pinned bounds travel through the same cast.
    """
    if high is None:
        high = float(torch.finfo(dtype).max)
    if low is None:
        low = float(torch.finfo(dtype).min)
    values = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
    inp = (values * (high - low) + low).to(dtype)
    if inp.numel() > 1:
        edges = torch.tensor(
            [low, high], dtype=torch.float32, device=flag_gems.device
        ).to(dtype)
        inp.view(-1)[:2] = edges
    return inp


def _make_int_input(dtype, shape, low, high):
    """Sample an integer tensor in [low, high] with both ends pinned.

    Integer inputs are built on CPU (GPU random init has no integer
    implementation) and promoted by acosh to float32, matching ATen.
    torch.randint rejects bounds outside the dtype's own range, so the
    bounds are clamped to the representable window first (e.g. uint8
    cannot hold -1, so [-1,0] degenerates to [0,0]). For int64 the exclusive
    upper bound (high + 1) would overflow the signed window torch.randint
    samples in, so the inclusive form is clamped to iinfo.max as well.
    """
    if high is None:
        high = torch.iinfo(dtype).max
    if low is None:
        low = torch.iinfo(dtype).min
    info = torch.iinfo(dtype)
    low = max(int(low), int(info.min))
    high = min(int(high), int(info.max))
    low = min(low, high)
    high = min(high, torch.iinfo(torch.int64).max - 1)
    inp = torch.randint(low, high + 1, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    if inp.numel() > 1:
        edges = torch.tensor([low, high], dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp.view(-1)[:2] = edges
    return inp


@pytest.mark.acosh
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + FP8_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh(shape, dtype, value_range):
    _, low, high = value_range
    inp = _make_float_input(dtype, shape, low, high)
    if dtype in FP8_DTYPES:
        # FP8 has no ATen acosh kernel on any backend; the reference is the
        # float32 upcast of the fp8 input, compared at fp8 resolution.
        res_out = flag_gems.acosh(inp)
        ref_out = torch.acosh(inp.float()).to(dtype)
        torch.testing.assert_close(
            res_out.float(),
            ref_out.float(),
            rtol=torch.finfo(dtype).eps,
            atol=1e-4,
            equal_nan=True,
        )
    else:
        ref_inp = utils.to_reference(inp)

        ref_out = torch.acosh(ref_inp)
        res_out = flag_gems.acosh(inp)

        # Three of the five ranges sit outside acosh's domain [1, inf), so the
        # comparison must ignore NaNs on both sides.
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh_int_promote(shape, dtype, value_range):
    # Integer inputs are promoted to float32 by acosh, exactly as ATen does.
    _, low, high = value_range
    inp = _make_int_input(dtype, shape, low, high)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.acosh(ref_inp)
    res_out = flag_gems.acosh(inp)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh_(shape, dtype, value_range):
    _, low, high = value_range
    inp = _make_float_input(dtype, shape, low, high)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh_(ref_inp)
    res_out = flag_gems.acosh_(inp)

    # Out-of-domain inputs yield NaN on both sides; equal_nan=True.
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh__fp8(shape, dtype, value_range):
    # The in-place kernel accepts fp8 as well; the reference is the float32
    # upcast, compared at fp8 resolution.
    _, low, high = value_range
    inp = _make_float_input(dtype, shape, low, high)
    # Snapshot before the in-place call: acosh_ mutates inp, so the reference
    # must be computed from a pre-mutation copy.
    snapshot = inp.float().clone()
    res_out = flag_gems.acosh_(inp)
    ref_out = torch.acosh(snapshot).to(dtype)
    torch.testing.assert_close(
        res_out.float(),
        ref_out.float(),
        rtol=torch.finfo(dtype).eps,
        atol=1e-4,
        equal_nan=True,
    )


@pytest.mark.acosh
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "scenario,payload",
    [
        ("nan", [float("nan"), 1.0, 0.5, -1.0, float("inf")]),
        ("inf", [float("inf"), float("-inf"), 1.0, 2.0, 0.0]),
        ("mixed", [float("nan"), float("inf"), float("-inf"), 1.0, 0.0]),
    ],
)
def test_acosh_special_values(dtype, scenario, payload):
    # Special float values through acosh: NaN propagates, +inf stays +inf,
    # everything below 1 becomes NaN. Computed on both sides in the input
    # dtype so the comparison runs at native resolution.
    inp = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh(ref_inp)
    res_out = flag_gems.acosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", [(), (8,), (4, 4)])
@pytest.mark.parametrize("dtype", INT_DTYPES + [torch.bool])
def test_acosh__integer_rejects(shape, dtype):
    # ATen raises for in-place acosh on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    if dtype is torch.bool:
        inp = torch.zeros(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.randint(0, 3, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.acosh_(inp)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.1
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone().requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    direct_grad = flag_gems.acosh_backward(grad, res_inp)
    utils.gems_assert_close(direct_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_registered_dispatch(dtype):
    # A small 2-D tensor is enough to verify the Autograd-key routing itself.
    inp = (
        torch.rand((64, 64), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["acosh"])
        res_out = torch.acosh(inp)
        assert "AcoshFunction" in type(res_out.grad_fn).__name__
        res_grad = torch.autograd.grad(res_out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_special_values(dtype):
    values = [
        -float("inf"),
        -2.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        float("inf"),
        float("nan"),
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    grad = torch.ones_like(inp)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone().requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_equal(res_grad, ref_grad, equal_nan=True)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_noncontiguous(dtype):
    # A transposed 2-D input exercises the single kernel after normalization.
    inp = (torch.rand((32, 64), dtype=dtype, device=flag_gems.device) + 1.1).transpose(
        0, 1
    )
    assert not inp.is_contiguous()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(
        inp.detach().clone(memory_format=torch.preserve_format)
    ).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().clone(memory_format=torch.preserve_format).requires_grad_()

    ref_grad = torch.autograd.grad(torch.acosh(ref_inp), ref_inp, ref_grad_output)[0]
    res_grad = torch.autograd.grad(flag_gems.acosh(res_inp), res_inp, grad)[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.empty_like(inp)

    res_grad = torch.autograd.grad(flag_gems.acosh(inp), inp, grad)[0]

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == dtype


@pytest.mark.acosh_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_backward_fused_kernel_count(dtype):
    """The backward replaces ATen's four-kernel formula with one fused kernel."""
    from torch.profiler import ProfilerActivity, profile

    # This size is large enough for the profiler to record the Triton launch.
    inp = (
        torch.rand((4096,), dtype=dtype, device=flag_gems.device) + 1.1
    ).requires_grad_()
    out = flag_gems.acosh(inp)
    grad = torch.randn_like(inp)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "acosh_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
