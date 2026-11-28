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
from flag_gems.ops.fused_moving_avg_obs_fq_helper_backward import (
    fused_moving_avg_obs_fq_helper_backward,
)

from . import accuracy_utils as utils

# The native forward is float32-only, so the autograd gradient chain it feeds
# is float32; the kernel itself preserves the incoming gradient's dtype (the
# historical ATen backward was ``mask * grad``, a dtype-local multiply), and
# low-precision cases are compared against the explicit formula.
DTYPES = utils.FLOAT_DTYPES

# Shapes for the dtype matrix.  The native-autograd cases cover a broad set
# separately; this matrix exercises the formula reference on every dtype.
MATRIX_SHAPES = [(1,), (16, 512), (1024, 1024), (2, 19, 7)]


def _case(shape, dtype, seed):
    """Shared input construction: x scaled past the quant range, random grads."""
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device) * 3.0
    if dtype is not torch.float32:
        x = x.to(dtype)
    go = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return x, go


def _native_fwd(x, observer_on, fake_quant_on, per_channel, qmin, qmax, ch_axis=0):
    """Run the native fused forward on ``x`` (float32, requires_grad)."""
    n = x.shape[ch_axis] if per_channel else 1
    dev = x.device
    obs = torch.tensor(observer_on, dtype=torch.long, device=dev)
    fq = torch.tensor(fake_quant_on, dtype=torch.long, device=dev)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=dev)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=dev)
    scale = torch.ones((n,), dtype=torch.float32, device=dev)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=dev)
    return torch.ops.aten._fused_moving_avg_obs_fq_helper(
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
        False,
    )


def _native_autograd_ref(
    x, go, observer_on, fake_quant_on, per_channel=False, qmin=0, qmax=255
):
    """Native autograd reference (float32 only): grad through the fused forward.

    Verified bit-exact against the mask formula below on 16M-element tensors.
    Returns (grad, mask).
    """
    ref_x = utils.to_reference(x).requires_grad_(True)
    ref_go = utils.to_reference(go)
    out, mask = _native_fwd(ref_x, observer_on, fake_quant_on, per_channel, qmin, qmax)
    (grad,) = torch.autograd.grad(out, ref_x, ref_go)
    return grad, mask


def _formula_ref(x, go, scale, zero_point, quant_min, quant_max, fq=True):
    """Explicit mask formula in the gradient's dtype.

    The mask is ``round(x/scale) + zero_point`` inside ``[quant_min,
    quant_max]`` (computed in float32, IEEE division as on the device), and
    the gradient is a dtype-local ``grad * {0, 1}`` multiply.  This mirrors
    the historical ATen backward and was verified bit-exact against the
    native autograd chain for the configurations exercised here.
    """
    if not fq:
        return go.clone(), torch.ones_like(go, dtype=torch.bool)
    ref_x = utils.to_reference(x).to(torch.float32)
    ref_go = utils.to_reference(go).to(torch.float32)
    ref_scale = utils.to_reference(scale).to(torch.float32)
    ref_zp = utils.to_reference(zero_point).to(torch.float32)
    if ref_scale.numel() == ref_x.shape[0] and ref_x.dim() >= 1:
        # Per-channel ch_axis=0 broadcast: one group per leading-dim row.
        shape = [1] * ref_x.dim()
        shape[0] = -1
        ref_scale = ref_scale.reshape(shape)
        ref_zp = ref_zp.reshape(shape)
    q = torch.round(ref_x / ref_scale) + ref_zp
    mask = (q >= quant_min) & (q <= quant_max)
    grad = (ref_go * mask.to(torch.float32)).to(go.dtype)
    return grad, mask


def _qparams(scale_val, zp_val, n=1, device=None):
    """Qparam tensors on the FlagGems device."""
    device = device or flag_gems.device
    scale = torch.full((n,), scale_val, dtype=torch.float32, device=device)
    zero_point = torch.full((n,), zp_val, dtype=torch.int32, device=device)
    return scale, zero_point


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
def test_fused_moving_avg_obs_fq_helper_backward_native_ref(shape):
    """Masked STE bit-exact against the FusedMovingAvgObsFqHelperBackward0 node.

    The scale/zero_point are the *final* qparams (recomputed from the same
    running min/max the forward uses), matching the exact QAT call order
    where the backward consumes the qparams the forward updated in place.
    The native forward rejects 0-dim inputs
    (``ch_axis must be < self.dim()``), so there is no gradient there.
    """
    if len(shape) == 0:
        pytest.skip("native forward rejects 0-dim inputs")
    x, go = _case(shape, torch.float32, 0)
    ref_grad, _ = _native_autograd_ref(x, go, observer_on=0, fake_quant_on=1)

    scale, zero_point = _qparams(0.003921568859368563, 127)
    res_grad, dummy = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)
    assert dummy.shape == x.shape
    assert torch.all(dummy == 1)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
def test_fused_moving_avg_obs_fq_helper_backward(dtype, shape):
    """Every dtype: mask-recomputed multiply against the explicit formula."""
    x, go = _case(shape, dtype, 0)
    scale, zero_point = _qparams(0.004, 120)
    ref_grad, _ = _formula_ref(x, go, scale, zero_point, 0, 255)

    res_grad, dummy = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert res_grad.dtype == dtype
    assert dummy.shape == x.shape
    assert torch.all(dummy == 1)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_moving_avg_obs_fq_helper_backward_per_channel(dtype):
    """ch_axis=0 rows take their own scale/zero_point mask."""
    shape = (4, 64)
    x, go = _case(shape, dtype, 1)
    n = shape[0]
    scale = torch.linspace(0.001, 0.02, n, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.arange(n, dtype=torch.int32, device=flag_gems.device) * 10
    ref_grad, _ = _formula_ref(x, go, scale, zero_point, 0, 255)

    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("fq", [False, True])
def test_fused_moving_avg_obs_fq_helper_backward_fq_switch(fq):
    """fake_quant off passes the gradient through; on applies the mask.

    The kernel qparams match the ones the native forward recomputes from the
    running min/max (scale 2*0.5/255, zero_point 127), so the reference
    gradient and the kernel mask see the same quantization grid.
    """
    dtype = torch.float32
    shape = (16, 512)
    x, go = _case(shape, dtype, 2)
    ref_grad, _ = _native_autograd_ref(
        x, go, observer_on=0, fake_quant_on=1 if fq else 0
    )

    scale, zero_point = _qparams(0.5 / 127.5, 127)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, fq, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)
    if not fq:
        assert torch.equal(res_grad, go)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_mask_parts():
    """All-ones / all-zeros / mixed masks multiply exactly as advertised."""
    dtype = torch.float32
    dev = flag_gems.device
    x = torch.tensor([0.1, 10.0, 0.2, 10.0], dtype=dtype, device=dev)
    go = torch.tensor([1.0, 2.0, float("inf"), float("nan")], dtype=dtype, device=dev)
    ref_grad, _ = _native_autograd_ref(x, go, observer_on=0, fake_quant_on=1)

    scale = torch.full((1,), 1 / 255, dtype=torch.float32, device=dev)
    zero_point = torch.full((1,), 127, dtype=torch.int32, device=dev)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32, equal_nan=True)
    # x=0.1 in range -> full grad; x=10 clamped -> grad * 0; the inf/nan
    # gradients sit at in-range positions and pass through unchanged, while
    # the nan gradient at an out-of-range position becomes 0 * nan = nan the
    # same way the native multiply does.
    assert res_grad[0].item() == 1.0
    assert res_grad[1].item() == 0.0
    assert torch.isinf(res_grad[2])
    assert torch.isnan(res_grad[3])


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_moving_avg_obs_fq_helper_backward_mask_extremes(dtype):
    """Mask all-ones (range containing the data) and all-zeros (empty range)."""
    shape = (128,)
    x, go = _case(shape, dtype, 3)
    scale = torch.ones(1, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.zeros(1, dtype=torch.int32, device=flag_gems.device)

    # All-ones mask: range wide enough for every element; gradient identical.
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, -1000, 1000, True, True
    )
    assert torch.equal(res_grad, go)

    # All-zero mask: the quantized range [qmax, qmin] sits far away from
    # round(x / 1.0) + 0 for every element.
    ref_grad, _ = _formula_ref(x, go, scale, zero_point, 900, 901)
    res_grad0, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 900, 901, True, True
    )
    utils.gems_assert_close(res_grad0, ref_grad, dtype)
    assert torch.equal(res_grad0, torch.zeros_like(go))


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_symmetric():
    """qmin=-128/qmax=127 with zero_point 128 (symmetric qparams)."""
    dtype = torch.float32
    shape = (16, 512)
    x, go = _case(shape, dtype, 4)
    ref_grad, _ = _native_autograd_ref(x, go, 0, 1, qmin=-128, qmax=127)

    # qparams the native forward recomputes from rmin=-0.5/rmax=0.5
    # (asymmetric branch: scale = 1/255, zero_point clamps to qmin=0).
    scale = torch.full((1,), 1 / 255, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.full((1,), 0, dtype=torch.int32, device=flag_gems.device)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, -128, 127, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_moving_avg_obs_fq_helper_backward_non_contiguous(dtype):
    """A transposed input/grad pair keeps the mask aligned by logical index."""
    shape = (48, 64)
    base, go_base = _case(shape, dtype, 5)
    x = base.t()
    go = go_base.t()
    scale = torch.full((1,), 0.004, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.full((1,), 120, dtype=torch.int32, device=flag_gems.device)
    ref_grad, _ = _formula_ref(base, go_base, scale, zero_point, 0, 255)

    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad.t(), dtype)
    assert res_grad.shape == x.shape


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_empty():
    """Empty tensors produce empty outputs without launching."""
    dev = flag_gems.device
    x = torch.empty(0, dtype=torch.float32, device=dev)
    go = torch.empty(0, dtype=torch.float32, device=dev)
    scale = torch.ones(1, dtype=torch.float32, device=dev)
    zero_point = torch.zeros(1, dtype=torch.int32, device=dev)
    res_grad, dummy = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    assert res_grad.shape == (0,)
    assert dummy.shape == (0,)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_moving_avg_obs_fq_helper_backward_0d(dtype):
    """0-dim tensors take the per-tensor path with scalar qparams."""
    x = torch.tensor(0.3, dtype=torch.float32, device=flag_gems.device)
    go = torch.tensor(2.0, dtype=torch.float32, device=flag_gems.device)
    scale = torch.full((), 1.0, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.zeros((), dtype=torch.int32, device=flag_gems.device)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, 0, 255, True, True
    )
    # 0.3 in range -> gradient passes.
    assert res_grad.item() == go.item()

    # An out-of-range element (q = 1e6 > qmax) receives zero gradient.
    x_out = torch.tensor(1e6, dtype=torch.float32, device=flag_gems.device)
    res_out, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x_out, scale, zero_point, 0, 255, True, True
    )
    assert res_out.item() == 0.0


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_boundary():
    """Elements landing exactly on quant_min/quant_max stay in the mask."""
    dtype = torch.float32
    shape = (1 << 16,)
    go = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Exact quantization-boundary values: q == qmin and q == qmax both belong
    # to the mask (inclusive range).
    x = (torch.arange(16, dtype=torch.float32, device=flag_gems.device) - 8.0)[
        torch.randint(0, 16, shape, device=flag_gems.device)
    ]
    ref_grad, _ = _native_autograd_ref(
        x, go, observer_on=0, fake_quant_on=1, qmin=-8, qmax=7
    )

    # qparams the native forward recomputes from rmin=-0.5/rmax=0.5:
    # scale = 1/15 (rounded by the device), zero_point = 0.
    scale = torch.full(
        (1,), 0.06666667014360428, dtype=torch.float32, device=flag_gems.device
    )
    zero_point = torch.full((1,), 0, dtype=torch.int32, device=flag_gems.device)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go, x, scale, zero_point, -8, 7, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_mismatched_qparams_rejected():
    """qparam shapes outside the per-tensor / ch_axis=0 contract fail."""
    dev = flag_gems.device
    x = torch.randn((4, 8), dtype=torch.float32, device=dev)
    go = torch.randn((4, 8), dtype=torch.float32, device=dev)
    bad_scale = torch.ones(3, dtype=torch.float32, device=dev)
    bad_zp = torch.zeros(3, dtype=torch.int32, device=dev)
    with pytest.raises(RuntimeError):
        fused_moving_avg_obs_fq_helper_backward(
            go, x, bad_scale, bad_zp, 0, 255, True, True
        )


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_int_rejected():
    """Integer tensors cannot carry a gradient."""
    dev = flag_gems.device
    x = torch.ones(4, dtype=torch.int32, device=dev)
    go = torch.ones(4, dtype=torch.int32, device=dev)
    scale = torch.ones(1, dtype=torch.float32, device=dev)
    zero_point = torch.zeros(1, dtype=torch.int32, device=dev)
    with pytest.raises(RuntimeError):
        fused_moving_avg_obs_fq_helper_backward(
            go, x, scale, zero_point, 0, 255, True, True
        )


@pytest.mark.fused_moving_avg_obs_fq_helper_backward
def test_fused_moving_avg_obs_fq_helper_backward_updated_qparams():
    """End-to-end: final qparams of the forward feed the backward.

    Runs the native fused forward (which updates scale/zero_point in place)
    and feeds the *updated* qparams to the kernel, comparing against the
    native autograd gradient -- the exact QAT call sequence.
    """
    shape = (16, 512)
    x32, go32 = _case(shape, torch.float32, 7)

    # The native forward runs on the reference device (CPU under --ref cpu)
    # on the same fp32 values the kernel receives as ``self``; the kernel
    # then consumes the qparams the forward updated in place.
    ref_x = utils.to_reference(x32).requires_grad_(True)
    n = 1
    dev = ref_x.device
    obs = torch.tensor(1, dtype=torch.long, device=dev)
    fq = torch.tensor(1, dtype=torch.long, device=dev)
    running_min = torch.full((n,), -0.5, dtype=torch.float32, device=dev)
    running_max = torch.full((n,), 0.5, dtype=torch.float32, device=dev)
    scale = torch.ones((n,), dtype=torch.float32, device=dev)
    zero_point = torch.zeros((n,), dtype=torch.int32, device=dev)
    out, _ = _native_fwd_final(
        ref_x, obs, fq, running_min, running_max, scale, zero_point
    )
    (ref_grad,) = torch.autograd.grad(out, ref_x, utils.to_reference(go32))

    scale = scale.to(flag_gems.device)
    zero_point = zero_point.to(flag_gems.device)
    res_grad, _ = fused_moving_avg_obs_fq_helper_backward(
        go32, x32, scale, zero_point, 0, 255, True, True
    )
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)


def _native_fwd_final(x, obs, fq, running_min, running_max, scale, zero_point):
    return torch.ops.aten._fused_moving_avg_obs_fq_helper(
        x,
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
