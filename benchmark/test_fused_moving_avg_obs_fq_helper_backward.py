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

import math

import pytest
import torch

from flag_gems.ops.fused_moving_avg_obs_fq_helper_backward import (
    fused_moving_avg_obs_fq_helper_backward,
)

from . import base, consts

# Keep every shape at or below 2**24 elements (~64MB per fp32 tensor): the
# backward needs grad + self + output simultaneously, and the per-channel
# qparams stay tiny.
_MAX_ELEMENTS = 2**24


def fused_moving_avg_obs_fq_helper_backward_input_fn(shape, dtype, device):
    # QAT backward inputs: a gradient, the forward's fp32 input values, and
    # the final qparams (recomputed from running min/max = -0.5/0.5 with
    # qmin/qmax = 0/255, i.e. scale = 1/255, zero_point = 127).  Per-tensor
    # for 1-d shapes, per-channel ch_axis=0 otherwise.
    grad = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    self_ = (torch.randn(shape, dtype=torch.float32, device=device) * 3.0).to(dtype)
    n = shape[0] if len(shape) > 1 else 1
    scale = torch.full((n,), 1 / 255, dtype=torch.float32, device=device)
    zero_point = torch.full((n,), 127, dtype=torch.int32, device=device)
    yield grad, self_, scale, zero_point, 0, 255, True, True


class FusedMovingAvgObsFqHelperBackwardBenchmark(base.GenericBenchmark):
    def init_user_config(self):
        super().init_user_config()
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= _MAX_ELEMENTS
        ]


def _torch_backward(grad, self_, scale, zero_point, qmin, qmax, fq, obs):
    """ATen baseline driven through the FusedMovingAvgObsFqHelperBackward0 node.

    The forward is float32-only, so the reference chain runs in fp32 on the
    same values; the running stats are re-initialized per call (the closure
    runs under the repeat loop).
    """
    dev = self_.device
    obs_t = torch.tensor(0, dtype=torch.long, device=dev)
    fq_t = torch.tensor(1 if fq else 0, dtype=torch.long, device=dev)
    per_channel = scale.numel() > 1 and self_.dim() > 1
    n = scale.numel() if per_channel else 1
    rmin = torch.full((n,), -0.5, dtype=torch.float32, device=dev)
    rmax = torch.full((n,), 0.5, dtype=torch.float32, device=dev)
    sc = torch.ones((n,), dtype=torch.float32, device=dev)
    zp = torch.zeros((n,), dtype=torch.int32, device=dev)
    x = self_.to(torch.float32).requires_grad_(True)
    out, _ = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        x,
        obs_t,
        fq_t,
        rmin,
        rmax,
        sc,
        zp,
        0.01,
        qmin,
        qmax,
        0,
        per_channel,
        False,
    )
    return torch.autograd.grad(out, x, grad.to(torch.float32))[0].to(grad.dtype)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward():
    bench = FusedMovingAvgObsFqHelperBackwardBenchmark(
        op_name="fused_moving_avg_obs_fq_helper_backward",
        torch_op=_torch_backward,
        gems_op=fused_moving_avg_obs_fq_helper_backward,
        input_fn=fused_moving_avg_obs_fq_helper_backward_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
