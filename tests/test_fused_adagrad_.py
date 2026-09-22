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

# The two registered ATen variants are ``aten::_fused_adagrad`` (out-of-place)
# and ``aten::_fused_adagrad_`` (in-place).  pytest forbids marker names that
# start with ``_``, so the leading-underscore ATen names cannot be used as
# marks; the stripped, pytest-legal marks (``fused_adagrad`` /
# ``fused_adagrad_``) are applied to the test functions below, matching the
# ``fused_adam`` convention.

# Adagrad optimizer parameter shapes covering small/large parameter tensors
# (attention head, embedding row, MLP weight, large embedding). All numels are
# multiples of 4 -- see ``_assert_grads_unchanged`` for why that matters.
FUSED_ADAGRAD_SHAPES = [
    (1024,),
    (4096,),
    (256, 256),
    (1024, 256),
    (2048, 512),
]

# Per-dtype tolerance for the Adagrad update; fp16/bf16 accumulate in fp32 but
# store back into the low-precision tensor, so allow a slightly looser bound.
TOLS = {
    torch.float32: 1e-5,
    torch.float16: 1e-3,
    torch.bfloat16: 2e-2,
    torch.float64: 1e-12,
}

# dtypes accepted by aten::_fused_adagrad_.
ADAGRAD_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


def _native_reference(params, grads, state_sums, steps, **kwargs):
    """Run the native ``torch._fused_adagrad_`` on clones and return the results.

    Using the native op as the accuracy reference (rather than a hand-written
    formula) keeps the test honest about ATen's exact semantics, including the
    corrected-lr expression and the ``found_inf`` comparison.
    """
    p = [t.clone() for t in params]
    g = [t.clone() for t in grads]
    s = [t.clone() for t in state_sums]
    st = [t.clone() for t in steps]
    torch._fused_adagrad_(p, g, s, st, **kwargs)
    return p, s


def _assert_close(got, expected, dtype):
    atol = TOLS[dtype]
    utils.gems_assert_close(
        utils.to_reference(got), utils.to_reference(expected), dtype, atol=atol
    )


def _assert_grads_unchanged(grads, originals):
    """``grads`` must be read-only.

    The native CUDA kernel happens to leave the unscaled gradient behind for
    tail elements when numel is not a multiple of its vector width (4), but that
    is a vectorization artifact rather than part of the operator contract, so
    FlagGems never writes to ``grad``. Every shape used here has a numel that is
    a multiple of 4, where the native op leaves ``grad`` untouched as well.
    """
    for i, (g, g0) in enumerate(zip(grads, originals)):
        assert torch.equal(g, g0), f"grads[{i}] was modified in place"


def _make_inputs(shape, dtype, device, *, nonzero_state=False, step=3.0):
    """Build a single-tensor optimizer state returned as 1-element lists.

    ``aten::_fused_adagrad_`` / ``aten::_fused_adagrad`` expect lists of tensors
    (foreach semantics); we exercise the single-tensor case through a 1-element
    list and the multi-tensor case through an explicit 3-element list.
    """
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    if nonzero_state:
        # A fresh optimizer starts at zero, but a resumed run does not; a zero
        # state hides sign/accumulation errors in the state_sum update.
        state_sum = torch.rand(shape, dtype=dtype, device=device) + 0.5
    else:
        state_sum = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([step], dtype=torch.float32, device=device)
    return [param], [grad], [state_sum], [state_step]


# ---------------------------------------------------------------------------
# In-place variant: aten::_fused_adagrad_  (mutates params & state_sums)
# ---------------------------------------------------------------------------


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", FUSED_ADAGRAD_SHAPES)
@pytest.mark.parametrize("dtype", ADAGRAD_DTYPES)
def test_fused_adagrad__basic(shape, dtype):
    """Basic in-place Adagrad step against the native op."""
    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(shape, dtype, flag_gems.device)
    grads_before = [g.clone() for g in grads]
    opts = dict(
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[0], ref_p[0], dtype)
    _assert_close(state_sums[0], ref_s[0], dtype)
    _assert_grads_unchanged(grads, grads_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "lr_decay,weight_decay,maximize",
    [
        (0.1, 0.0, False),
        (0.0, 0.01, False),
        (0.1, 0.01, True),
        (0.5, 0.001, False),
    ],
)
@pytest.mark.parametrize("nonzero_state", [False, True])
def test_fused_adagrad__options(
    shape, dtype, lr_decay, weight_decay, maximize, nonzero_state
):
    """In-place Adagrad with various lr_decay / weight_decay / maximize combos."""
    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(
        shape, dtype, flag_gems.device, nonzero_state=nonzero_state
    )
    grads_before = [g.clone() for g in grads]
    opts = dict(
        lr=0.05,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=1e-10,
        maximize=maximize,
        grad_scale=None,
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[0], ref_p[0], dtype)
    _assert_close(state_sums[0], ref_s[0], dtype)
    _assert_grads_unchanged(grads, grads_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scale", [2.0, 1024.0])
@pytest.mark.parametrize("weight_decay,maximize", [(0.0, False), (0.01, True)])
def test_fused_adagrad__grad_scale(shape, dtype, scale, weight_decay, maximize):
    """AMP gradient unscaling must match the native op, and must not touch grads.

    The unscaled gradient feeds both the state_sum accumulation and the parameter
    update, so a missing or misordered division shows up in both outputs.
    """
    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(
        shape, dtype, flag_gems.device, nonzero_state=True
    )
    grads_before = [g.clone() for g in grads]
    opts = dict(
        lr=0.01,
        lr_decay=0.1,
        weight_decay=weight_decay,
        eps=1e-10,
        maximize=maximize,
        grad_scale=torch.tensor(scale, device=flag_gems.device),
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[0], ref_p[0], dtype)
    _assert_close(state_sums[0], ref_s[0], dtype)
    _assert_grads_unchanged(grads, grads_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("dtype", [torch.float32])
# ATen compares found_inf against 1 exactly: 0.0 and 2.0 both still update.
@pytest.mark.parametrize("found_inf_val", [0.0, 1.0, 2.0])
def test_fused_adagrad__found_inf(dtype, found_inf_val):
    """``found_inf == 1`` skips the step; any other value applies it."""
    shape = (1024,)
    torch.manual_seed(42)
    params, grads, state_sums, steps = _make_inputs(
        shape, dtype, flag_gems.device, nonzero_state=True
    )
    params_before = [p.clone() for p in params]
    state_before = [s.clone() for s in state_sums]
    grads_before = [g.clone() for g in grads]
    opts = dict(
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=torch.tensor(found_inf_val, device=flag_gems.device),
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[0], ref_p[0], dtype)
    _assert_close(state_sums[0], ref_s[0], dtype)
    _assert_grads_unchanged(grads, grads_before)

    if found_inf_val == 1.0:
        # Buffers must be left exactly as they were, not merely close.
        assert torch.equal(params[0], params_before[0])
        assert torch.equal(state_sums[0], state_before[0])
    else:
        assert not torch.equal(params[0], params_before[0])


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_adagrad__empty(dtype):
    """Empty parameters are a no-op and must not raise or launch a null grid."""
    params = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    grads = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    state_sums = [torch.empty(0, dtype=dtype, device=flag_gems.device)]
    steps = [torch.tensor([1.0], dtype=torch.float32, device=flag_gems.device)]

    flag_gems._fused_adagrad_(
        params,
        grads,
        state_sums,
        steps,
        lr=0.01,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )

    assert params[0].numel() == 0
    assert state_sums[0].numel() == 0


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_adagrad__mixed_empty_and_nonempty(dtype):
    """An empty entry must not stop the remaining parameters from updating."""
    shape = (256,)
    torch.manual_seed(42)
    params = [
        torch.empty(0, dtype=dtype, device=flag_gems.device),
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
    ]
    grads = [
        torch.empty(0, dtype=dtype, device=flag_gems.device),
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
    ]
    state_sums = [
        torch.empty(0, dtype=dtype, device=flag_gems.device),
        torch.rand(shape, dtype=dtype, device=flag_gems.device),
    ]
    steps = [
        torch.tensor([1.0], dtype=torch.float32, device=flag_gems.device),
        torch.tensor([4.0], dtype=torch.float32, device=flag_gems.device),
    ]
    opts = dict(
        lr=0.01,
        lr_decay=0.2,
        weight_decay=0.01,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[1], ref_p[1], dtype)
    _assert_close(state_sums[1], ref_s[1], dtype)


@pytest.mark.fused_adagrad_
def test_fused_adagrad__mismatched_list_lengths():
    """Shorter companion lists must be rejected before any buffer is touched."""
    dtype = torch.float32
    dev = flag_gems.device
    params = [torch.randn(64, dtype=dtype, device=dev) for _ in range(2)]
    grads = [torch.randn(64, dtype=dtype, device=dev)]  # too short
    state_sums = [torch.zeros(64, dtype=dtype, device=dev) for _ in range(2)]
    steps = [torch.tensor([1.0], device=dev) for _ in range(2)]
    params_before = [p.clone() for p in params]

    with pytest.raises(RuntimeError, match="same length"):
        flag_gems._fused_adagrad_(
            params,
            grads,
            state_sums,
            steps,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
            grad_scale=None,
            found_inf=None,
        )

    # Validation happens up front, so no parameter may have been updated.
    for p, p0 in zip(params, params_before):
        assert torch.equal(p, p0)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize(
    "bad_kind,match",
    [
        ("shape", "does not match param shape"),
        ("stride", "do not match param strides"),
        ("dtype", "must have same dtype, device, and layout"),
        ("device", "must have same dtype, device, and layout"),
        ("layout", "non-overlapping-and-dense"),
    ],
)
def test_fused_adagrad__mismatched_tensor_properties(bad_kind, match):
    """Shape/dtype/device/layout mismatches must raise before any update."""
    dtype = torch.float32
    dev = flag_gems.device
    params = [torch.randn(64, dtype=dtype, device=dev)]
    grads = [torch.randn(64, dtype=dtype, device=dev)]
    state_sums = [torch.zeros(64, dtype=dtype, device=dev)]
    if bad_kind == "shape":
        grads = [torch.randn(32, dtype=dtype, device=dev)]
    elif bad_kind == "stride":
        # Both sides are non-overlapping and dense, but a contiguous param and a
        # transposed grad do not share strides, so the storage-order walk would
        # pair up the wrong elements.
        params = [torch.randn(8, 8, dtype=dtype, device=dev)]
        grads = [torch.randn(8, 8, dtype=dtype, device=dev).t()]
        state_sums = [torch.zeros(8, 8, dtype=dtype, device=dev)]
    elif bad_kind == "dtype":
        grads = [torch.randn(64, dtype=torch.float16, device=dev)]
    elif bad_kind == "device":
        grads = [torch.randn(64, dtype=dtype, device="cpu")]
    elif bad_kind == "layout":
        grads = [torch.randn(1, dtype=dtype, device=dev).expand(64)]
    steps = [torch.tensor([1.0], device=dev)]
    params_before = [p.clone() for p in params]

    with pytest.raises(RuntimeError, match=match):
        flag_gems._fused_adagrad_(
            params,
            grads,
            state_sums,
            steps,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
            grad_scale=None,
            found_inf=None,
        )

    for p, p0 in zip(params, params_before):
        assert torch.equal(p, p0)


def _assert_buffers_untouched(params, params_before, state_sums, sums_before):
    """Every param and state_sum must be bit-identical after a rejected call.

    A validation that ran lazily inside the launch loop would leave the entries
    before the bad one already updated; asserting on both lists catches that.
    """
    for i, (p, p0) in enumerate(zip(params, params_before)):
        assert torch.equal(p, p0), f"params[{i}] was partially updated"
    for i, (s, s0) in enumerate(zip(state_sums, sums_before)):
        assert torch.equal(s, s0), f"state_sums[{i}] was partially updated"


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize(
    "bad_kind,match",
    [
        # ATen's check_fast_path_restrictions requires one common dtype and
        # device across params/grads/state_sums; mixing them per entry is
        # rejected. The invalid entry is the *second* one, so a lazy check
        # would already have applied the first entry's update.
        ("params_dtype", "must have same dtype, device, and layout"),
        ("params_device", "must have same dtype, device, and layout"),
        ("grads_dtype", "must have same dtype, device, and layout"),
        ("state_sums_dtype", "must have same dtype, device, and layout"),
    ],
)
def test_fused_adagrad__later_entry_bad_dtype_or_device(bad_kind, match):
    """An invalid later entry must raise before any earlier entry is updated."""
    dev = flag_gems.device
    dtype = torch.float32
    shape = (256,)

    def make(dt=dtype, device=dev):
        return torch.randn(shape, dtype=dt, device=device)

    params = [make(), make()]
    grads = [make(), make()]
    state_sums = [torch.zeros(shape, dtype=dtype, device=dev) for _ in range(2)]
    steps = [torch.tensor([1.0], device=dev) for _ in range(2)]

    if bad_kind == "params_dtype":
        params[1] = make(torch.float16)
    elif bad_kind == "params_device":
        params[1] = make(device="cpu")
    elif bad_kind == "grads_dtype":
        grads[1] = make(torch.float16)
    elif bad_kind == "state_sums_dtype":
        state_sums[1] = torch.zeros(shape, dtype=torch.float16, device=dev)

    params_before = [p.clone() for p in params]
    sums_before = [s.clone() for s in state_sums]

    with pytest.raises(RuntimeError, match=match):
        flag_gems._fused_adagrad_(
            params,
            grads,
            state_sums,
            steps,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
            grad_scale=None,
            found_inf=None,
        )

    _assert_buffers_untouched(params, params_before, state_sums, sums_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize(
    "bad_kind,match",
    [
        ("dtype_int32", "must contain float32 scalar tensors"),
        ("dtype_int64", "must contain float32 scalar tensors"),
        ("dtype_fp64", "must contain float32 scalar tensors"),
        ("numel_2", "must contain 1-element tensors"),
        ("numel_0", "must contain 1-element tensors"),
        ("device_cpu", "different from other tensors"),
    ],
)
def test_fused_adagrad__later_entry_bad_state_step(bad_kind, match):
    """state_steps are validated (dtype/numel/device) before the first launch.

    ATen reads the step through a raw float pointer, so a non-float32 or
    multi-element step silently produces a wrong corrected_lr (and an empty one
    reads out of bounds). The bad entry is second, so nothing may be updated.
    """
    dev = flag_gems.device
    dtype = torch.float32
    shape = (256,)
    params = [torch.randn(shape, dtype=dtype, device=dev) for _ in range(2)]
    grads = [torch.randn(shape, dtype=dtype, device=dev) for _ in range(2)]
    state_sums = [torch.zeros(shape, dtype=dtype, device=dev) for _ in range(2)]
    steps = [torch.tensor([1.0], device=dev) for _ in range(2)]

    if bad_kind == "dtype_int32":
        steps[1] = torch.tensor([1], dtype=torch.int32, device=dev)
    elif bad_kind == "dtype_int64":
        steps[1] = torch.tensor([1], dtype=torch.int64, device=dev)
    elif bad_kind == "dtype_fp64":
        steps[1] = torch.tensor([1.0], dtype=torch.float64, device=dev)
    elif bad_kind == "numel_2":
        steps[1] = torch.tensor([1.0, 2.0], dtype=torch.float32, device=dev)
    elif bad_kind == "numel_0":
        steps[1] = torch.tensor([], dtype=torch.float32, device=dev)
    elif bad_kind == "device_cpu":
        steps[1] = torch.tensor([1.0], dtype=torch.float32, device="cpu")

    params_before = [p.clone() for p in params]
    sums_before = [s.clone() for s in state_sums]

    with pytest.raises(RuntimeError, match=match):
        flag_gems._fused_adagrad_(
            params,
            grads,
            state_sums,
            steps,
            lr=0.01,
            lr_decay=0.1,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
            grad_scale=None,
            found_inf=None,
        )

    _assert_buffers_untouched(params, params_before, state_sums, sums_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("shape", [(1024,), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_adagrad__multi_tensor(shape, dtype):
    """Adagrad over a list of multiple parameter tensors (foreach semantics)."""
    torch.manual_seed(42)
    params = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    grads = [torch.randn(shape, dtype=dtype, device=flag_gems.device) for _ in range(3)]
    state_sums = [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    # Distinct steps per tensor: a shared step would hide an indexing bug in the
    # per-tensor state_step read.
    steps = [
        torch.tensor([float(i + 1)], dtype=torch.float32, device=flag_gems.device)
        for i in range(3)
    ]
    grads_before = [g.clone() for g in grads]
    opts = dict(
        lr=0.01,
        lr_decay=0.3,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    for p, rp, s, rs in zip(params, ref_p, state_sums, ref_s):
        _assert_close(p, rp, dtype)
        _assert_close(s, rs, dtype)
    _assert_grads_unchanged(grads, grads_before)


@pytest.mark.fused_adagrad_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_adagrad__transposed_parity(dtype):
    # ATen accepts strided tensors that are non-overlapping and dense, so a
    # matching transposed layout must be supported (the kernel walks the
    # storage order) rather than rejected by an ``is_contiguous()`` check.
    utils.init_seed(0)
    shape = (256, 256)

    def make_tr(values=None):
        p = (
            torch.randn(shape, dtype=dtype, device=flag_gems.device)
            if values is None
            else values
        )
        return p.t().contiguous().t()

    params = [make_tr()]
    grads = [make_tr()]
    # ``state_sum`` is a running sum of squared gradients, so it must be
    # non-negative -- a negative entry would make the reference itself NaN.
    state_sums = [make_tr(torch.rand(shape, dtype=dtype, device=flag_gems.device))]
    steps = [torch.tensor([3.0], device=flag_gems.device, dtype=torch.float32)]
    grads_before = [g.clone() for g in grads]

    opts = dict(
        lr=0.1,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )

    ref_p, ref_s = _native_reference(params, grads, state_sums, steps, **opts)
    flag_gems._fused_adagrad_(params, grads, state_sums, steps, **opts)

    _assert_close(params[0], ref_p[0], dtype)
    _assert_close(state_sums[0], ref_s[0], dtype)
    _assert_grads_unchanged(grads, grads_before)


@pytest.mark.fused_adagrad_
def test_fused_adagrad__internal_overlap_rejected():
    # An expanded tensor has no well-defined element order; ATen's
    # non-overlapping-and-dense requirement rejects it too.
    base = torch.randn(4, dtype=torch.float32, device=flag_gems.device)
    expanded = base.expand(4, 4)
    steps = [torch.tensor([1.0], device=flag_gems.device, dtype=torch.float32)]
    with pytest.raises(RuntimeError, match="non-overlapping-and-dense"):
        flag_gems._fused_adagrad_(
            [expanded],
            [expanded.clone()],
            [torch.zeros(4, 4, device=flag_gems.device)],
            steps,
            lr=0.1,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
        )
