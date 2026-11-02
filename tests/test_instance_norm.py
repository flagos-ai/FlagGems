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


# Regression tests for issue #4885: the running-stats update kernel recovered
# the biased variance from rstd as (1 / rstd**2 + eps), but 1 / rstd**2 is
# already var + eps (rstd = rsqrt(var + eps)), so running_var was updated with
# (var + 2 * eps) instead of var. The default eps=1e-5 keeps the error below
# the accuracy tolerance, so these tests use a large eps to expose it. The
# shapes span all three forward kernels that store rstd: N <= 128 (multiline
# persistent), 128 < N <= 4096 (persistent) and N > 4096 (loop).
INSTANCE_NORM_RUNNING_STATS_EPS = [0.1, 1.0]
INSTANCE_NORM_RUNNING_STATS_SHAPES = [
    (4, 16, 8, 8),  # N=64, multiline persistent kernel
    (2, 3, 1024),  # N=1024, persistent kernel
    (2, 3, 128, 128),  # N=16384, loop kernel
]


@pytest.mark.instance_norm_running_stats
@pytest.mark.parametrize("shape", INSTANCE_NORM_RUNNING_STATS_SHAPES)
@pytest.mark.parametrize("eps", INSTANCE_NORM_RUNNING_STATS_EPS)
def test_instance_norm_running_var_recovery(shape, eps):
    torch.manual_seed(0)
    x = torch.randn(shape, device=device)
    C = shape[1]
    momentum = 0.1

    ref_running_mean = torch.zeros(C)
    ref_running_var = torch.ones(C)
    torch.nn.functional.instance_norm(
        x.cpu(),
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )

    running_mean = torch.zeros(C, device=device)
    running_var = torch.ones(C, device=device)
    flag_gems.instance_norm(
        x,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )

    utils.gems_assert_close(running_mean.cpu(), ref_running_mean, torch.float32)
    utils.gems_assert_close(running_var.cpu(), ref_running_var, torch.float32)


@pytest.mark.instance_norm_running_stats
@pytest.mark.parametrize("shape", [(2, 3, 16), (2, 3, 64)])
def test_instance_norm_running_var_non_negative(shape):
    """Near-constant channels: subtractive cancellation must not go negative."""
    torch.manual_seed(1)
    x = torch.ones(shape, device=device) + torch.randn(shape, device=device) * 1e-6
    C = shape[1]
    running_mean = torch.zeros(C, device=device)
    running_var = torch.ones(C, device=device)
    flag_gems.instance_norm(
        x,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=True,
        momentum=0.1,
        eps=1e-5,
    )
    assert bool(torch.all(running_var >= 0)), "running_var must stay non-negative"
