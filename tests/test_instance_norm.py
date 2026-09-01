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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NORM_SHAPES = [
        (2, 1, 2, 1),
    ]
    WEIGTH_BIAS = [True]
    USE_INPUT_BIAS = [True]
    HAS_RUN_STATS = [False]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NORM_SHAPES = [
        (1, 1, 2, 2),
        (2, 1, 2, 2),
        (2, 3, 2, 2),
        (2, 3, 128, 128),
        (4, 16, 8, 8),
        (2, 3, 1024),
        (2, 3, 2048),
        (2, 3, 4096),
        (2, 3, 8192),
        (2, 3, 10240),
    ]
    WEIGTH_BIAS = [False, True]
    USE_INPUT_BIAS = [False, True]
    HAS_RUN_STATS = [False, True]

device = flag_gems.device


@pytest.mark.instance_norm
@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("has_weight_bias", WEIGTH_BIAS)
@pytest.mark.parametrize("use_input_stats", USE_INPUT_BIAS)
@pytest.mark.parametrize("has_running_stats", HAS_RUN_STATS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_instance_norm(
    shape, dtype, has_weight_bias, use_input_stats, has_running_stats
):
    if use_input_stats is False and has_running_stats is False:
        return

    B, C = shape[:2]
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

    weight = None
    bias = None
    if has_weight_bias:
        weight = torch.randn(size=(C,), dtype=dtype, device=device, requires_grad=True)
        bias = torch.randn(size=(C,), dtype=dtype, device=device, requires_grad=True)

    running_mean = None
    running_var = None
    if has_running_stats:
        running_mean = torch.randn(size=(C,), dtype=torch.float32, device=device)
        r = torch.randn(size=(C,), dtype=torch.float32, device=device).abs()
        running_var = r + 1e-5

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_running_mean = utils.to_reference(None, True)
    ref_running_var = utils.to_reference(None, True)
    if has_running_stats:
        ref_running_mean = utils.to_reference(running_mean.clone(), True)
        ref_running_var = utils.to_reference(running_var.clone(), True)

    ref_out = torch.nn.functional.instance_norm(
        ref_inp,
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )

    res_out = flag_gems.instance_norm(
        inp,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    if has_running_stats:
        utils.gems_assert_close(running_mean, ref_running_mean, running_mean.dtype)
        utils.gems_assert_close(running_var, ref_running_var, running_var.dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = utils.to_reference(out_grad, True)

    if has_weight_bias:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)

    M = B * C
    N = inp.numel() // M

    if use_input_stats:
        utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)

        if has_weight_bias:
            utils.gems_assert_close(
                res_weight_grad, ref_weight_grad, dtype, reduce_dim=B * N
            )
            utils.gems_assert_close(
                res_bias_grad, ref_bias_grad, dtype, reduce_dim=B * N
            )


# Regression for issue #4885: the running-stats update kernel recovered the
# biased variance as (1 / rstd**2 + eps) instead of (1 / rstd**2 - eps), so
# running_var was stored as (var + 2 * eps) * N / (N - 1) instead of
# var * N / (N - 1). The default eps=1e-5 keeps the error under the accuracy
# tolerance, so this test uses a larger eps to expose it. The shapes span all
# three forward kernels that store rstd: N <= 128 (multiline persistent),
# 128 < N <= 4096 (persistent), and N > 4096 (loop).
INSTANCE_NORM_LARGE_EPS = [0.1, 1.0]
INSTANCE_NORM_EPS_SHAPES = [
    (4, 16, 8, 8),  # N=64, multiline persistent
    (2, 3, 2, 2),  # N=4, small
    (2, 2, 64, 64),  # N=4096, persistent
    (2, 2, 72, 72),  # N=5184, loop
]


@pytest.mark.instance_norm
@pytest.mark.parametrize("shape", INSTANCE_NORM_EPS_SHAPES)
@pytest.mark.parametrize("eps", INSTANCE_NORM_LARGE_EPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_instance_norm_running_var_large_eps(shape, eps, dtype):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    running_mean = torch.randn(size=(C,), dtype=torch.float32, device=device)
    running_var = (
        torch.randn(size=(C,), dtype=torch.float32, device=device).abs() + 1e-5
    )
    momentum = 0.1

    ref_inp = utils.to_reference(inp, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    torch.nn.functional.instance_norm(
        ref_inp,
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )
    flag_gems.instance_norm(
        inp,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )

    utils.gems_assert_close(running_mean, ref_running_mean, running_mean.dtype)
    utils.gems_assert_close(running_var, ref_running_var, running_var.dtype)


@pytest.mark.instance_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_instance_norm_running_var_near_zero(dtype):
    # A near-constant channel has ~zero variance. With momentum=1.0 and a zero
    # initial running_var, the stored running_var equals the freshly recovered
    # variance directly (running_var = new_var / B), so it reflects the update
    # kernel's output without being diluted by the previous value. The clamp
    # tl.maximum(1 / rstd**2 - eps, 0) then guarantees running_var stays
    # non-negative under subtractive cancellation. See issue #4885.
    shape = (2, 4, 8, 8)
    C = shape[1]
    inp = torch.ones(shape, dtype=dtype, device=device)
    inp += 1e-4 * torch.randn(shape, dtype=dtype, device=device)
    running_mean = torch.zeros(size=(C,), dtype=torch.float32, device=device)
    running_var = torch.zeros(size=(C,), dtype=torch.float32, device=device)
    eps = 1.0
    momentum = 1.0

    ref_inp = utils.to_reference(inp, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    torch.nn.functional.instance_norm(
        ref_inp,
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )
    flag_gems.instance_norm(
        inp,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )

    assert torch.all(running_var >= 0), "running_var must never be negative"
    utils.gems_assert_close(running_var, ref_running_var, running_var.dtype)
