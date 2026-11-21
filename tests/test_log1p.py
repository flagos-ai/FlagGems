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

# log1p is defined over floats in ATen, but the CUDA kernel promotes integral
# inputs to float first, so 1-byte ints are also accepted. Byte-int inputs stay
# in [0, 100) to keep log1p finite and away from the domain boundary.
LOG1P_DTYPES = utils.FLOAT_DTYPES + utils.BYTE_INT_DTYPES


@pytest.mark.log1p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", LOG1P_DTYPES)
def test_log1p(shape, dtype):
    torch.manual_seed(0)
    if dtype in utils.BYTE_INT_DTYPES:
        inp = torch.randint(0, 100, shape, dtype=torch.int32, device="cpu").to(
            dtype=dtype, device=flag_gems.device
        )
    else:
        inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.log1p(ref_inp)
    res_out = flag_gems.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, res_out.dtype)


@pytest.mark.log1p_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.log1p_()
    res_out = flag_gems.log1p_(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


QUICK_MODE = utils.QUICK_MODE

BACKWARD_SHAPES = (
    [(), (0,), (257,), (19, 37)]
    if QUICK_MODE
    else [(), (0,), (1,), (257,), (19, 37), (64, 512, 512)]
)

# `1 + x` is exactly representable at these points, so the ATen formula
# (rounded denominator, float32 opmath quotient) and the analytic dx agree.
BACKWARD_SPECIAL_VALUES = [
    -2.0,  # outside the forward domain; backward still uses grad / (1 + x)
    -1.0,  # log1p domain boundary: 1 + x == 0 -> +inf gradient
    -0.9999999403953552,  # nearest float32 below -1 + 2**-24
    -0.9999,
    -0.9,
    -0.5,
    -1e-8,
    -0.0,
    0.0,
    1e-8,
    0.5,
    1.0,
    1e10,
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.log1p_backward
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    if x.ndim == 2:
        # Non-contiguous input: the backward must still return x's logical
        # gradient, not a flat copy of the transposed buffer.
        x = x.t()
    x.requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)

    out = flag_gems.log1p(x)
    ref_out = torch.log1p(ref_x)
    (dx,) = torch.autograd.grad(out, x, grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_x, utils.to_reference(grad))

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.log1p_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward_expanded_grad(dtype):
    x = torch.rand((7, 13), dtype=dtype, device=flag_gems.device) + 0.5
    x.requires_grad_()
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    # Zero-stride grad_output: the kernel must read the broadcast value, not
    # walk a dense buffer of expanded memory.
    grad = torch.randn((7, 1), dtype=dtype, device=x.device).expand_as(x)

    (dx,) = torch.autograd.grad(flag_gems.log1p(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.log1p(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.log1p_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward_special_values(dtype):
    x = torch.tensor(
        BACKWARD_SPECIAL_VALUES,
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    ref_x = utils.to_reference(x.detach()).requires_grad_()
    grad = torch.ones_like(x)

    (dx,) = torch.autograd.grad(flag_gems.log1p(x), x, grad)
    (ref_dx,) = torch.autograd.grad(torch.log1p(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype, equal_nan=True)


@pytest.mark.log1p_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward_matches_aten_rounding(dtype):
    """The fused kernel reproduces ATen's CUDA backward bit for bit.

    ATen rounds `1 + x` to the input dtype before dividing with float32
    opmath; computing the quotient directly in float32 differs by up to a few
    ulp for fp16/bf16. Both sides run on the device here on purpose: this is an
    exactness check, so it must not be compared against the float64 CPU
    reference used by ``gems_assert_close``.
    """
    x = torch.cat(
        [
            torch.rand(1 << 16, dtype=dtype, device=flag_gems.device) * 1.99 - 0.995,
            torch.rand(1 << 16, dtype=dtype, device=flag_gems.device) * 8.0,
        ]
    )
    x = x[torch.isfinite(1.0 + x)]
    gem_x = x.clone().requires_grad_()
    ref_x = x.clone().requires_grad_()
    grad = torch.randn_like(gem_x)

    (dx,) = torch.autograd.grad(flag_gems.log1p(gem_x), gem_x, grad)
    (ref_dx,) = torch.autograd.grad(torch.log1p(ref_x), ref_x, grad)

    assert dx.dtype == dtype
    # Device-to-device on purpose: `--ref cpu` must not silently compare the
    # fused kernel against the float64 CPU reference here.
    torch.testing.assert_close(dx, ref_dx, atol=0, rtol=0, equal_nan=True)


@pytest.mark.log1p_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward_via_torch_dispatch(dtype):
    """`torch.log1p` under `use_gems()` must take the FlagGems backward."""
    x = torch.rand((32, 17), dtype=dtype, device=flag_gems.device) + 0.5
    ref_x = utils.to_reference(x.detach().clone()).requires_grad_()
    grad = torch.randn_like(x, memory_format=torch.contiguous_format)

    inp = x.detach().clone().requires_grad_()
    out = flag_gems.log1p(inp)
    (dx,) = torch.autograd.grad(out, inp, grad)

    assert "Log1pFunction" in type(out.grad_fn).__name__, type(out.grad_fn).__name__

    ref_x.requires_grad_()
    (ref_dx,) = torch.autograd.grad(torch.log1p(ref_x), ref_x, utils.to_reference(grad))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.log1p_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log1p_backward_fused_kernel_count(dtype):
    """The backward launches a single kernel (no ATen add + div composite)."""
    from torch.profiler import ProfilerActivity, profile

    x = torch.rand((4096,), dtype=dtype, device=flag_gems.device) + 0.5
    x.requires_grad_()
    out = flag_gems.log1p(x)
    grad = torch.randn_like(x)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (x,), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "log1p_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    # ATen's composite backward shows up as a separate add and div kernel.
    composite = [name for name, _ in launches if "_add" in name or "_div" in name]
    assert not composite, launches
