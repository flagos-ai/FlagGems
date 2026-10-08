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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The registered ATen variant exercised here is ``aten::_fused_sgd_`` (in-place).
# pytest forbids marker names that start with ``_``, so the leading-underscore
# ATen name cannot be used as a mark; the stripped, pytest-legal mark
# (``fused_sgd_``) is applied to the test functions below, matching the
# ``fused_adam`` convention.

# SGD fused optimizer tensors are exercised on the flag_gems device (CUDA).
# Several parameter combinations drive the kernel through every branch:
#   - plain momentum update
#   - dampening (1 - tau) factor
#   - nesterov look-ahead
#   - L2 weight decay
#   - maximize (gradient ascent)
#   - is_first_step (momentum buffer initialisation)
# The native CUDA fused-SGD kernel requires momentum > 0, so every case
# below keeps momentum strictly positive to remain comparable to the
# reference implementation.
SGD_CASES = [
    # (weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step)
    (0.0, 0.9, 0.1, 0.0, False, False, False),  # plain momentum
    (0.0, 0.9, 0.1, 0.0, True, False, False),  # nesterov
    (0.01, 0.9, 0.1, 0.0, False, False, False),  # weight decay
    (0.01, 0.9, 0.1, 0.0, False, True, False),  # maximize + weight decay
    (0.0, 0.9, 0.1, 0.5, False, False, False),  # dampening
    (0.01, 0.9, 0.1, 0.0, False, False, True),  # first step (init momentum buffer)
    (0.01, 0.9, 0.05, 0.2, True, True, False),  # everything together
]

SHAPES = utils.POINTWISE_SHAPES


def _skip_if_cpu_ref():
    """Skip when the CPU reference path is requested.

    ``torch._fused_sgd_`` only has a native CUDA fused-SGD implementation, so
    when CI runs the second pass with ``--ref=cpu --quick`` (``utils.TO_CPU``),
    the reference cannot run on CPU. Skip those cases rather than attempting a
    cross-device comparison that cannot succeed.
    """
    if utils.TO_CPU:
        pytest.skip("fused SGD has no native CPU reference (CUDA-only op)")


def _make_inputs(shape, dtype, device, momentum):
    """Build a single-tensor (param, grad, momentum_buffer) triple."""
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    # momentum_buffer is the running buffer; initialise with non-trivial data
    momentum_buf = torch.randn(shape, dtype=dtype, device=device)
    return param, grad, momentum_buf


def _run_ref(op, params, grads, mbufs, **kwargs):
    """Run the reference (native aten) implementation.

    ``torch._fused_sgd_`` dispatches to the FlagGems registered kernel when
    lists are passed (the FlagGems wrapper accepts list inputs), but the native
    CUDA fused-SGD op expects tuple inputs. Passing tuples therefore reaches
    the native path directly, without an explicit dispatch context. The
    reference inputs are routed through ``utils.to_reference`` so the
    cross-device reference path is consistent with the rest of the suite
    (callers guard ``utils.TO_CPU`` so the native CUDA op never runs on CPU).
    """
    ref = lambda xs: [utils.to_reference(x) for x in xs]
    return op(tuple(ref(params)), tuple(ref(grads)), tuple(ref(mbufs)), **kwargs)


def _run_gems(op, params, grads, mbufs, **kwargs):
    """Run the FlagGems implementation directly.

    ``op`` is the FlagGems wrapper (``flag_gems._fused_sgd_``); calling it
    directly invokes the Triton kernel without going through the aten dispatch
    machinery.
    """
    return op(params, grads, mbufs, **kwargs)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize(
    "wd,momentum,lr,dampening,nesterov,maximize,is_first_step", SGD_CASES
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_sgd_(
    shape, dtype, wd, momentum, lr, dampening, nesterov, maximize, is_first_step
):
    """In-place fused SGD step: compare gems vs native on params/grads/momentum."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    p, g, mb = _make_inputs(shape, dtype, device, momentum)
    # Reference clones
    ref_p = p.clone()
    ref_g = g.clone()
    ref_mb = mb.clone()
    # Gems clones (mutated in-place by the inplace op)
    gems_p = p.clone()
    gems_g = g.clone()
    gems_mb = mb.clone()

    _run_ref(
        torch._fused_sgd_,
        [ref_p],
        [ref_g],
        [ref_mb],
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        [gems_p],
        [gems_g],
        [gems_mb],
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )

    utils.gems_assert_close(gems_p, ref_p, dtype)
    utils.gems_assert_close(gems_mb, ref_mb, dtype)
    utils.gems_assert_close(gems_g, ref_g, dtype)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize(
    "wd,momentum,lr,dampening,nesterov,maximize,is_first_step", SGD_CASES
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_sgd__multitensor(
    shape, dtype, wd, momentum, lr, dampening, nesterov, maximize, is_first_step
):
    """In-place fused SGD over a list of tensors of differing shapes.

    Differing shapes are the point: the whole list is packed into a *single*
    launch (see ``test_fused_sgd__single_launch``), so this is what exercises the
    per-program tensor lookup rather than one grid per parameter.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    shapes = [shape, (64, 64), (128,), (7, 7, 7)]
    tensors = [_make_inputs(s, dtype, device, momentum) for s in shapes]
    ref_p = [t[0].clone() for t in tensors]
    ref_g = [t[1].clone() for t in tensors]
    ref_mb = [t[2].clone() for t in tensors]
    res_p = [t[0].clone() for t in tensors]
    res_g = [t[1].clone() for t in tensors]
    res_mb = [t[2].clone() for t in tensors]

    _run_ref(
        torch._fused_sgd_,
        ref_p,
        ref_g,
        ref_mb,
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        res_p,
        res_g,
        res_mb,
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )

    for rp, sp, rg, sg, rmb, smb in zip(ref_p, res_p, ref_g, res_g, ref_mb, res_mb):
        utils.gems_assert_close(sp, rp, dtype)
        utils.gems_assert_close(smb, rmb, dtype)
        utils.gems_assert_close(sg, rg, dtype)


@pytest.mark.fused_sgd_
def test_fused_sgd__grad_scale():
    """AMP gradient scaling: grad is divided by grad_scale in-place."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    dtype = torch.float32
    # A single representative size is enough for the grad_scale / found_inf code
    # paths, which are independent of shape and dtype.
    shape = (256, 256)
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    ref_p, ref_g, ref_mb = p.clone(), g.clone(), mb.clone()
    res_p, res_g, res_mb = p.clone(), g.clone(), mb.clone()
    gs = torch.tensor([4.0], device=device)

    _run_ref(
        torch._fused_sgd_,
        [ref_p],
        [ref_g],
        [ref_mb],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=gs,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        [res_p],
        [res_g],
        [res_mb],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=gs,
    )

    utils.gems_assert_close(res_p, ref_p, dtype)
    utils.gems_assert_close(res_mb, ref_mb, dtype)
    utils.gems_assert_close(res_g, ref_g, dtype)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("found_inf_val", [0.0, 1.0, 2.0])
def test_fused_sgd__found_inf(found_inf_val):
    """``found_inf == 1`` skips the step; any other value applies it.

    ATen compares against 1 exactly rather than ``> 0``, so 2.0 must still
    update. Testing only 1.0 passes under either comparison and cannot catch a
    kernel that loosened the condition to ``> 0``.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    dtype = torch.float32
    # A single representative size is enough: the skip path is independent of
    # shape and dtype.
    shape = (128, 128)
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    fi = torch.tensor([found_inf_val], device=device)
    opts = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        found_inf=fi,
    )

    ref_p, ref_g, ref_mb = p.clone(), g.clone(), mb.clone()
    _run_ref(torch._fused_sgd_, [ref_p], [ref_g], [ref_mb], **opts)

    res_p, res_g, res_mb = p.clone(), g.clone(), mb.clone()
    _run_gems(flag_gems._fused_sgd_, [res_p], [res_g], [res_mb], **opts)

    utils.gems_assert_close(res_p, ref_p, dtype)
    utils.gems_assert_close(res_g, ref_g, dtype)
    utils.gems_assert_close(res_mb, ref_mb, dtype)

    if found_inf_val == 1.0:
        # Buffers must be untouched, not merely close to the reference.
        assert torch.equal(res_p, p)
        assert torch.equal(res_mb, mb)
    else:
        assert not torch.equal(res_p, p)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("lr_shape", [(), (1,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_sgd__tensor_lr(lr_shape, dtype):
    """``_fused_sgd_.tensor_lr`` matches the native overload.

    fp64 is included deliberately: the float overload passes ``lr`` as a Python
    double, which Triton would narrow to fp32 for an unannotated argument, so a
    double-precision param has to either read it from a tensor or receive it
    through a ``tl.float64``-annotated scalar. Both overflow-independent paths
    must agree with ATen.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    shape = (256, 128)
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    # ATen requires an fp32 lr tensor regardless of the param dtype (an fp64 lr
    # raises "expected scalar type Float but found Double"), so the kernel must
    # widen it rather than assume it matches the params.
    lr = torch.full(lr_shape, 0.07, dtype=torch.float32, device=device)
    opts = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=lr,
        dampening=0.0,
        nesterov=True,
        maximize=False,
        is_first_step=False,
    )

    ref_p, ref_g, ref_mb = p.clone(), g.clone(), mb.clone()
    _run_ref(
        torch.ops.aten._fused_sgd_.tensor_lr,
        [ref_p],
        [ref_g],
        [ref_mb],
        **opts,
    )

    res_p, res_g, res_mb = p.clone(), g.clone(), mb.clone()
    _run_gems(flag_gems._fused_sgd__tensor_lr, [res_p], [res_g], [res_mb], **opts)

    utils.gems_assert_close(res_p, ref_p, dtype)
    utils.gems_assert_close(res_mb, ref_mb, dtype)


# ---------------------------------------------------------------------------
# Whole-list validation: a list spanning dtypes/devices must be rejected before
# anything is updated, matching ATen's fast-path restrictions.
# ---------------------------------------------------------------------------


def _step_kwargs(**overrides):
    kwargs = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    kwargs.update(overrides)
    return kwargs


def _assert_untouched(before, after):
    """Assert every tensor is byte-identical to its snapshot."""
    for b, a in zip(before, after):
        assert torch.equal(b, a), "a tensor was updated before the error was raised"


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("order", ["fp32_first", "fp16_first"])
def test_fused_sgd__rejects_mixed_dtype_list(order):
    """A param list spanning two dtypes raises before any tensor is touched.

    ATen enforces this whole-list rule in ``check_fast_path_restrictions`` and
    applies nothing when it fires, so neither the shape of the error nor the
    order of the mismatched pair may leave a partial update behind.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    shape = (64, 64)
    dtypes = [torch.float32, torch.float16]
    if order == "fp16_first":
        dtypes.reverse()

    ps = [torch.randn(shape, dtype=d, device=device) for d in dtypes]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    before = [t.clone() for t in ps + gs + mbs]

    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems._fused_sgd_(ps, gs, mbs, **_step_kwargs())

    _assert_untouched(before, ps + gs + mbs)


@pytest.mark.fused_sgd_
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices"
)
def test_fused_sgd__rejects_mixed_device_list():
    """A param list spanning two devices raises before any tensor is touched.

    Without a list-wide device check the tensors on the first device would be
    updated and the mismatch would only surface later, leaving a half-applied
    optimizer step.
    """
    _skip_if_cpu_ref()
    ps = [
        torch.randn(64, 64, device="cuda:0"),
        torch.randn(64, 64, device="cuda:1"),
    ]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    before = [t.clone() for t in ps + gs + mbs]

    with pytest.raises(RuntimeError, match="device"):
        flag_gems._fused_sgd_(ps, gs, mbs, **_step_kwargs())

    _assert_untouched(before, ps + gs + mbs)


@pytest.mark.fused_sgd_
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices"
)
def test_fused_sgd__launches_on_param_device():
    """A list on a non-current device runs there, matching ATen.

    ATen accepts a list on any one device; the launch must target the
    parameters' device rather than whatever the caller left current.
    """
    _skip_if_cpu_ref()
    device = "cuda:1"
    with torch.cuda.device("cuda:0"):
        # Build and run while cuda:0 is current, so only an explicit device
        # guard can make this work.
        shapes = [(256, 256), (1000,)]
        ps = [torch.randn(s, device=device) for s in shapes]
        gs = [torch.randn_like(p) for p in ps]
        mbs = [torch.randn_like(p) for p in ps]
        ref_p = [t.clone() for t in ps]
        ref_g = [t.clone() for t in gs]
        ref_mb = [t.clone() for t in mbs]
        torch.ops.aten._fused_sgd_(ref_p, ref_g, ref_mb, **_step_kwargs())
        flag_gems._fused_sgd_(ps, gs, mbs, **_step_kwargs())
        torch.cuda.synchronize()
    for i in range(len(shapes)):
        utils.gems_assert_close(ps[i], ref_p[i], torch.float32)
        utils.gems_assert_close(gs[i], ref_g[i], torch.float32)
        utils.gems_assert_close(mbs[i], ref_mb[i], torch.float32)


@pytest.mark.fused_sgd_
def test_fused_sgd__rejects_grad_dtype_mismatch():
    """A grad whose dtype differs from its param is rejected before launch.

    This is the within-index half of the same rule: ATen compares grads and
    buffers against the shared dtype rather than only against their own param.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    p = torch.randn(64, 64, dtype=torch.float32, device=device)
    g = torch.randn(64, 64, dtype=torch.float64, device=device)
    mb = torch.randn(64, 64, dtype=torch.float32, device=device)
    before = [p.clone(), g.clone(), mb.clone()]

    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems._fused_sgd_([p], [g], [mb], **_step_kwargs())

    _assert_untouched(before, [p, g, mb])


# ---------------------------------------------------------------------------
# Single-launch batching: the list must be one launch, not one per parameter.
# ---------------------------------------------------------------------------


def _count_launches(n_params, shape=(256, 256)):
    """Run one step over ``n_params`` tensors and count Triton kernel launches.

    The kernel wrapper is patched at module level for the duration of the call,
    so the count is exactly what ``flag_gems._fused_sgd_`` issues for one step.
    """
    kernel = importlib.import_module("flag_gems.ops._fused_sgd_")._fused_sgd_kernel
    original_run = kernel.run
    counter = {"n": 0}

    def counting_run(*args, **kwargs):
        counter["n"] += 1
        return original_run(*args, **kwargs)

    device = flag_gems.device
    ps = [torch.randn(shape, device=device) for _ in range(n_params)]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    try:
        kernel.run = counting_run
        flag_gems._fused_sgd_(ps, gs, mbs, **_step_kwargs())
    finally:
        kernel.run = original_run
    torch.cuda.synchronize()
    return counter["n"]


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("n_params", [1, 4, 16])
def test_fused_sgd__single_launch(n_params):
    """The whole parameter list costs one kernel launch, whatever its length."""
    _skip_if_cpu_ref()
    assert _count_launches(n_params) == 1


@pytest.mark.fused_sgd_
def test_fused_sgd__heterogeneous_batch_accuracy():
    """Accuracy on a heterogeneous list, including masked tail blocks.

    The shapes disagree in their remainder modulo the block size (including a
    single-element tensor), so a batched kernel that mis-derives the owning
    tensor or its mask shows up as a shape-indexed mismatch rather than noise.
    """
    _skip_if_cpu_ref()
    device = flag_gems.device
    shapes = [(4096, 4096), (1000,), (7, 7, 7), (1,), (513, 17)]
    tensors = [_make_inputs(s, torch.float32, device, 0.9) for s in shapes]
    ref_p = [t[0].clone() for t in tensors]
    ref_g = [t[1].clone() for t in tensors]
    ref_mb = [t[2].clone() for t in tensors]
    res_p = [t[0].clone() for t in tensors]
    res_g = [t[1].clone() for t in tensors]
    res_mb = [t[2].clone() for t in tensors]

    opts = _step_kwargs(
        weight_decay=0.01,
        dampening=0.2,
        nesterov=True,
    )
    _run_ref(torch._fused_sgd_, ref_p, ref_g, ref_mb, **opts)
    _run_gems(flag_gems._fused_sgd_, res_p, res_g, res_mb, **opts)

    for i in range(len(shapes)):
        utils.gems_assert_close(res_p[i], ref_p[i], torch.float32)
        utils.gems_assert_close(res_g[i], ref_g[i], torch.float32)
        utils.gems_assert_close(res_mb[i], ref_mb[i], torch.float32)


# ---------------------------------------------------------------------------
# CUDA graph capture: the step must not allocate or synchronize.
# ---------------------------------------------------------------------------


def _capture_step(op, params, grads, mbs, opts):
    """Warm up on a side stream and capture one step; return the graph."""
    for _ in range(3):
        op(params, grads, mbs, **opts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(params, grads, mbs, **opts)
    torch.cuda.synchronize()
    return graph


def _assert_replay_matches_eager(op, params, grads, mbs, opts):
    """Capture, then assert one replay equals one extra eager step.

    ``params``/``grads``/``mbs`` are stepped by the capture warmup; the eager
    reference is taken from that state, then the graph is replayed once and the
    result compared against a second eager step taken from the same state.
    """
    _skip_if_cpu_ref()
    graph = _capture_step(op, params, grads, mbs, opts)

    # Eager step from the post-warmup state == what one replay must produce.
    expected_p = [t.clone() for t in params]
    expected_g = [t.clone() for t in grads]
    expected_mb = [t.clone() for t in mbs]
    op(expected_p, expected_g, expected_mb, **opts)

    graph.replay()
    torch.cuda.synchronize()
    return params, expected_p, grads, expected_g, mbs, expected_mb


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_sgd__cuda_graph_capture(dtype):
    """One captured step replays correctly, including the fp64 float-lr path.

    fp64 is the regression guard for the per-call ``torch.tensor(lr, ...)`` the
    float-lr path used to allocate for every parameter: an allocation or a
    pageable host-to-device copy inside the step makes ``torch.cuda.graph``
    fail outright, so this test fails without the fix, not merely numerically.
    """
    device = flag_gems.device
    shapes = [(256, 256), (64, 64), (1000,)]
    ps = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    opts = _step_kwargs(nesterov=True)

    got = _assert_replay_matches_eager(flag_gems._fused_sgd_, ps, gs, mbs, opts)
    got_p, exp_p, got_g, exp_g, got_mb, exp_mb = got
    for i in range(len(shapes)):
        utils.gems_assert_close(got_p[i], exp_p[i], dtype)
        utils.gems_assert_close(got_g[i], exp_g[i], dtype)
        utils.gems_assert_close(got_mb[i], exp_mb[i], dtype)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_sgd__cuda_graph_capture_amp(dtype):
    """An AMP step (grad_scale + found_inf) captures and replays correctly.

    Both scalars are read inside the kernel on device; a host-side ``.item()``
    would abort the capture.
    """
    device = flag_gems.device
    shapes = [(256, 256), (64, 64)]
    ps = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    opts = _step_kwargs(
        grad_scale=torch.tensor([4.0], device=device),
        found_inf=torch.tensor([0.0], device=device),
    )

    got = _assert_replay_matches_eager(flag_gems._fused_sgd_, ps, gs, mbs, opts)
    got_p, exp_p, got_g, exp_g, got_mb, exp_mb = got
    for i in range(len(shapes)):
        utils.gems_assert_close(got_p[i], exp_p[i], dtype)
        utils.gems_assert_close(got_g[i], exp_g[i], dtype)
        utils.gems_assert_close(got_mb[i], exp_mb[i], dtype)


@pytest.mark.fused_sgd_
def test_fused_sgd__cuda_graph_capture_tensor_lr():
    """The tensor-lr overload also captures and replays correctly."""
    device = flag_gems.device
    shapes = [(256, 256), (1000,)]
    ps = [torch.randn(s, device=device) for s in shapes]
    gs = [torch.randn_like(p) for p in ps]
    mbs = [torch.randn_like(p) for p in ps]
    opts = _step_kwargs(lr=torch.tensor(0.1, device=device), nesterov=True)

    got = _assert_replay_matches_eager(
        flag_gems._fused_sgd__tensor_lr, ps, gs, mbs, opts
    )
    got_p, exp_p, got_g, exp_g, got_mb, exp_mb = got
    for i in range(len(shapes)):
        utils.gems_assert_close(got_p[i], exp_p[i], torch.float32)
        utils.gems_assert_close(got_g[i], exp_g[i], torch.float32)
        utils.gems_assert_close(got_mb[i], exp_mb[i], torch.float32)
