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

# (shape, ch_axis) pairs. ch_axis is only used in per-channel mode.
FMAOFQ_SHAPES = [
    ((4096,), 0),
    ((2048,), 0),
    ((16, 512), 0),
    ((32, 256), 0),
    ((8, 1024), 0),
]

UNSUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float64]


def _run_reference(x, observer_on, fake_quant_on, per_channel, ch_axis, symmetric):
    """Reference via native aten on the reference device (CPU or CUDA)."""
    n = x.shape[ch_axis] if per_channel else 1
    dev = x.device
    obs = torch.tensor(observer_on, dtype=torch.long, device=dev)
    fq = torch.tensor(fake_quant_on, dtype=torch.long, device=dev)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=dev)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=dev)
    scale = torch.ones((n,), dtype=torch.float32, device=dev)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=dev)
    qmin, qmax = (-128, 127) if symmetric else (0, 255)
    out, mask = torch.ops.aten._fused_moving_avg_obs_fq_helper(
        x,
        obs,
        fq,
        running_min,
        running_max,
        scale,
        zero_point,
        0.01,
        qmin,
        qmax,
        ch_axis,
        per_channel,
        symmetric,
    )
    return out, mask, running_min, running_max, scale, zero_point


@pytest.mark.fused_moving_avg_obs_fq_helper
@pytest.mark.parametrize("shape,ch_axis", FMAOFQ_SHAPES)
@pytest.mark.parametrize("observer_on", [1, 0])
@pytest.mark.parametrize("fake_quant_on", [1, 0])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_fused_moving_avg_obs_fq_helper(
    shape, ch_axis, observer_on, fake_quant_on, per_channel, symmetric
):
    # per-channel needs at least 2 dims to index a channel axis.
    if per_channel and len(shape) < 2:
        pytest.skip("per-channel mode requires a channel axis")

    dtype = torch.float32
    # Seed for determinism: fake-quant compares the clamp mask with exact
    # equality, so an input value landing exactly on a quantization boundary can
    # flip a single mask bit via a 1-ULP difference between the native op's and
    # the kernel's x/scale. A fixed seed keeps the matrix reproducible.
    torch.manual_seed(0)
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Reference on the reference device (CPU when --ref=cpu, else CUDA).
    ref_x = utils.to_reference(base)
    (
        ref_out,
        ref_mask,
        ref_rmin,
        ref_rmax,
        ref_scale,
        ref_zp,
    ) = _run_reference(
        ref_x, observer_on, fake_quant_on, per_channel, ch_axis, symmetric
    )

    n = shape[ch_axis] if per_channel else 1
    dev = flag_gems.device
    obs = torch.tensor(observer_on, dtype=torch.long, device=dev)
    fq = torch.tensor(fake_quant_on, dtype=torch.long, device=dev)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=dev)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=dev)
    scale = torch.ones((n,), dtype=torch.float32, device=dev)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=dev)
    qmin, qmax = (-128, 127) if symmetric else (0, 255)

    with flag_gems.use_gems():
        res_out, res_mask = torch.ops.aten._fused_moving_avg_obs_fq_helper(
            base,
            obs,
            fq,
            running_min,
            running_max,
            scale,
            zero_point,
            0.01,
            qmin,
            qmax,
            ch_axis,
            per_channel,
            symmetric,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert torch.equal(res_mask.bool().cpu(), ref_mask.bool().cpu()), "mask mismatch"
    # In-place-mutated observer state must match the native op.
    utils.gems_assert_close(running_min, ref_rmin, torch.float32)
    utils.gems_assert_close(running_max, ref_rmax, torch.float32)
    utils.gems_assert_close(scale, ref_scale, torch.float32)
    assert torch.equal(zero_point.cpu(), ref_zp.cpu()), "zero_point mismatch"


@pytest.mark.fused_moving_avg_obs_fq_helper
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_fused_moving_avg_obs_fq_helper_rejects_non_fp32(dtype):
    """Match ATen's float32-only input contract."""
    dev = flag_gems.device
    x = torch.randn((16, 32), dtype=dtype, device=dev)
    obs = torch.tensor(1, dtype=torch.long, device=dev)
    fq = torch.tensor(1, dtype=torch.long, device=dev)
    running_min = torch.full((1,), -0.5, dtype=torch.float32, device=dev)
    running_max = torch.full((1,), 0.5, dtype=torch.float32, device=dev)
    scale = torch.ones((1,), dtype=torch.float32, device=dev)
    zero_point = torch.zeros((1,), dtype=torch.int32, device=dev)

    args = (
        obs,
        fq,
        running_min,
        running_max,
        scale,
        zero_point,
        0.01,
        0,
        255,
        0,
        False,
        False,
    )
    error = "expected scalar type Float but found"

    # The exact native error text varies by PyTorch version depending on which
    # dtype check runs first, but every supported version rejects these inputs.
    with pytest.raises(RuntimeError):
        torch.ops.aten._fused_moving_avg_obs_fq_helper(x, *args)
    with flag_gems.use_gems(), pytest.raises(RuntimeError, match=error):
        torch.ops.aten._fused_moving_avg_obs_fq_helper(x, *args)
