from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

# `cudnn_ctc_loss` only accepts float32 `log_probs` (cuDNN restriction).
CTC_DTYPES = [torch.float32]

# (T, N, C, S) tuples: time steps, batch size, number of classes, max target
# length. `cudnn_ctc_loss` only accepts float32 `log_probs` and int32 targets.
CTC_SHAPES = [
    (1, 1, 2, 1),
    (4, 1, 8, 2),
    (16, 4, 16, 5),
    (32, 8, 32, 8),
    (64, 16, 64, 16),
    (128, 32, 128, 32),
]


def _make_inputs(shape, device, input_lengths=None, target_lengths=None):
    """Build log_probs/targets for a CTC problem.

    The returned targets are a single 1D int32 tensor, which is the layout
    `cudnn_ctc_loss` expects (targets are kept on the CPU since cuDNN requires
    the CPU backend for them).
    """
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    if input_lengths is None:
        input_lengths = [t_steps] * batch
    if target_lengths is None:
        target_lengths = [max_target] * batch
    pieces = [
        torch.randint(1, classes, (length,), device=device, dtype=torch.int32)
        for length in target_lengths
    ]
    targets = (
        torch.cat(pieces).cpu()
        if pieces
        else torch.empty(0, dtype=torch.int32, device="cpu")
    )
    return log_probs, targets, input_lengths, target_lengths


def _reference(log_probs, targets, input_lengths, target_lengths):
    """Reference result using `F.ctc_loss`, comparing only the valid region.

    cuDNN leaves the gradient of the padded time steps uninitialized, so the
    rows `t >= input_lengths[b]` are not part of the reference.
    """
    losses = F.ctc_loss(
        log_probs.detach(),
        targets.to(log_probs.device),
        input_lengths,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    lp = log_probs.detach().clone().requires_grad_(True)
    ref = F.ctc_loss(
        lp,
        targets.to(log_probs.device),
        input_lengths,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    (grads,) = torch.autograd.grad(ref.sum(), lp, allow_unused=True)
    return losses, grads, lp


def _assert_valid_region_close(res_grad, ref_grad, input_lengths):
    """Compare only the rows the cuDNN alignment actually visits.

    The alignment accumulates `log_alpha` over `input_lengths[b]` steps, so the
    absolute error grows with the sequence length. Scale the tolerance the same
    way the `ctc_loss` tests do.
    """
    batch = ref_grad.shape[1]
    reduce_dim = max(1, ref_grad.shape[0] * (max(input_lengths) + 1))
    for b in range(batch):
        valid = input_lengths[b]
        utils.gems_assert_close(
            res_grad[:valid, b],
            ref_grad[:valid, b],
            torch.float32,
            reduce_dim=reduce_dim,
        )


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize("shape", CTC_SHAPES)
def test_cudnn_ctc_loss_forward(shape):
    device = flag_gems.device
    torch.manual_seed(0)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    ref_loss, _, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    assert res_loss.shape == (shape[1],)
    assert res_loss.dtype == torch.float32
    utils.gems_assert_close(res_loss, ref_loss, torch.float32)


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize("shape", CTC_SHAPES)
def test_cudnn_ctc_loss_backward(shape):
    device = flag_gems.device
    torch.manual_seed(0)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    _, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    _, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    assert res_grad.shape == log_probs.shape
    assert res_grad.dtype == torch.float32
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize(
    "input_lengths,target_lengths",
    [
        ([16, 16, 16, 16], [5, 5, 5, 5]),
        ([16, 12, 8, 15], [5, 3, 1, 4]),
        ([16, 16, 16, 16], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
    ],
)
def test_cudnn_ctc_loss_varying_lengths(input_lengths, target_lengths):
    device = flag_gems.device
    torch.manual_seed(0)
    # Fixed (T, N, C, S) host shape; only the per-batch lengths vary.
    shape = (16, 4, 16, 5)
    log_probs, targets, _, _ = _make_inputs(
        shape, device, input_lengths, target_lengths
    )
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    ref_loss, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_empty_target():
    """An empty label sequence matches cuDNN on loss and on the non-blank gradient.

    cuDNN reports a zero loss for the empty label sequence and seeds the
    gradient with `exp(log_probs)`, which is what the kernel returns. One
    column does not match: cuDNN subtracts `prod_t exp(log_probs[t, blank])`
    from the blank entry, so its gradient is not `exp(log_probs)` there and its
    rows do not sum to zero. That is not a usable gradient -- nothing
    accumulates posterior mass into state 0 -- so it is pinned as a known
    difference rather than reproduced. The gap shrinks with `input_lengths`,
    which is why the earlier form of this test compared whole rows and only
    passed at `input_lengths=[5, ...]`.

    Only the valid time steps are compared, since cuDNN leaves the padded rows
    uninitialized.
    """
    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([3], dtype=torch.int32)
    # Short enough that the blank-column difference is far above the tolerance,
    # so the split assertion below is actually exercised.
    input_lengths = [2, 8]
    target_lengths = [0, 1]

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    # The reference is the native op: no other test in this file leaves a global
    # registration behind (the dispatch test is scoped and tears its own down).
    ref_loss, ref_grad = torch.ops.aten._cudnn_ctc_loss(
        log_probs,
        targets.to(device),
        torch.tensor(input_lengths, dtype=torch.int32, device=device),
        torch.tensor(target_lengths, dtype=torch.int32, device=device),
        0,
        False,
        False,
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    valid = input_lengths[0]
    # The empty-label gradient is the `exp(log_probs)` seed verbatim, not zeros.
    utils.gems_assert_close(
        res_grad[:valid, 0], log_probs[:valid, 0].exp(), torch.float32
    )
    # ...and it agrees with cuDNN on every column but the blank one.
    utils.gems_assert_close(
        res_grad[:valid, 0, 1:], ref_grad[:valid, 0, 1:], torch.float32
    )
    # Blank column: cuDNN subtracts the all-blank path probability.
    blank_path = log_probs[:valid, 0, 0].exp().prod()
    utils.gems_assert_close(
        ref_grad[:valid, 0, 0],
        log_probs[:valid, 0, 0].exp() - blank_path,
        torch.float32,
    )
    # The two implementations differ here, and by more than the tolerance --
    # this pins the split so the assertion above cannot silently start passing
    # because the difference shrank.
    assert (
        res_grad[:valid, 0, 0] - ref_grad[:valid, 0, 0]
    ).abs().max() > 1e-6, "blank column now matches cuDNN; collapse the split"


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_autograd(monkeypatch):
    """The loss must be differentiable end to end through `log_probs`.

    Differentiation has to go through the dispatcher: the registered entry is
    only the CUDA kernel, and PyTorch's own autograd rule for
    `_cudnn_ctc_loss` supplies the backward. A direct `flag_gems.cudnn_ctc_loss`
    call returns plain tensors with no `grad_fn`.
    """
    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 16, 3, 8
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_lengths = [2, 1, 1]

    ref_lp = utils.to_reference(log_probs, True).clone().requires_grad_(True)
    ref = F.ctc_loss(
        ref_lp,
        targets.to(ref_lp.device),
        [t_steps] * batch,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    (ref_grad,) = torch.autograd.grad(ref.sum(), ref_lp)

    gems_lp = log_probs.clone().requires_grad_(True)
    _install_spy(monkeypatch, [])
    with _registered_cudnn_ctc_loss(["cudnn_ctc_loss"]):
        gems_loss = torch.ops.aten._cudnn_ctc_loss(
            gems_lp, targets, [t_steps] * batch, target_lengths, 0, False, False
        )[0]
        (gems_grad,) = torch.autograd.grad(gems_loss.sum(), gems_lp)

    utils.gems_assert_close(gems_loss, ref.detach(), torch.float32)
    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_weighted_grad(monkeypatch):
    """Each batch entry must be scaled by its own upstream gradient.

    A plain `.sum()` upstream folds into a uniform gradient, which hides a
    per-batch indexing bug: every sample would be scaled by `grad_output[0]`
    and still match. Weighting the upstream by a per-batch vector makes the
    scaling observable. The scaling itself is done by PyTorch's native
    `_cudnn_ctc_loss_backward`, so this pins the interaction between the Triton
    forward and that rule.
    """
    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 16, 3, 8
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_lengths = [2, 1, 1]
    # One weight vector per side: under --ref=cpu the reference works on the
    # host while the operator stays on the device, and the upstream gradient has
    # to sit next to the loss it scales.
    upstream = [1.0, 0.0, -2.0]

    ref_lp = utils.to_reference(log_probs, True).clone().requires_grad_(True)
    ref = F.ctc_loss(
        ref_lp,
        targets.to(ref_lp.device),
        [t_steps] * batch,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    ref_weights = torch.tensor(upstream, dtype=ref_lp.dtype, device=ref_lp.device)
    (ref_grad,) = torch.autograd.grad((ref * ref_weights).sum(), ref_lp)

    gems_lp = log_probs.clone().requires_grad_(True)
    gems_weights = torch.tensor(upstream, dtype=log_probs.dtype, device=device)
    _install_spy(monkeypatch, [])
    with _registered_cudnn_ctc_loss(["cudnn_ctc_loss"]):
        gems_loss = torch.ops.aten._cudnn_ctc_loss(
            gems_lp, targets, [t_steps] * batch, target_lengths, 0, False, False
        )[0]
        (gems_grad,) = torch.autograd.grad((gems_loss * gems_weights).sum(), gems_lp)

    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_tensor_lengths():
    """`input_lengths`/`target_lengths` may be passed as tensors."""
    device = flag_gems.device
    torch.manual_seed(0)
    # Fixed (T, N, C, S) shape; only the lengths-as-tensors path is exercised.
    shape = (16, 4, 16, 5)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    length_tensors = (
        torch.tensor(input_lengths, dtype=torch.int32),
        torch.tensor(target_lengths, dtype=torch.int32),
    )

    ref_loss, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, length_tensors[0], length_tensors[1], 0, False, False
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_invalid_blank():
    """`blank` must be 0, matching the cuDNN restriction."""
    device = flag_gems.device
    log_probs, targets, input_lengths, target_lengths = _make_inputs(
        (8, 2, 6, 2), device
    )
    with pytest.raises(RuntimeError, match="blank must be label 0"):
        flag_gems.cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 1, False, False
        )


def _install_spy(monkeypatch, calls):
    """Route the registered `cudnn_ctc_loss` entry through a counter.

    A numeric comparison cannot tell gems apart from the native kernel; only
    the fact that this wrapper ran can.
    """
    real = flag_gems.cudnn_ctc_loss

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    def tensor_spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    mapping = dict(flag_gems.FULL_CONFIG_BY_FUNC)
    mapping["cudnn_ctc_loss"] = [
        ("_cudnn_ctc_loss", spy, None),
        ("_cudnn_ctc_loss.Tensor", tensor_spy, None),
    ]
    monkeypatch.setattr(flag_gems, "FULL_CONFIG_BY_FUNC", mapping)
    return spy


@contextmanager
def _registered_cudnn_ctc_loss(include=("cudnn_ctc_loss",)):
    """Scope a `cudnn_ctc_loss` registration to the calling `with` block.

    Test files must not use `flag_gems.use_gems()`, and `only_enable` alone
    registers into the global aten library where it outlives the test: a
    `finally: enable()` would then leave *every* op registered, so the
    `torch.ops.aten._cudnn_ctc_loss` references in the other tests would turn
    into self-comparisons. A private library plus restoring the registrar keeps
    the registration local, the same way `test_special_zeta` does it.
    """
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = flag_gems.current_work_registrar
    if isinstance(include, str):
        include = [include]
    try:
        flag_gems.only_enable(
            lib=library,
            include=list(include),
            registrar=flag_gems.GeneralOpRegistrar,
        )
        yield
    finally:
        if hasattr(library, "_destroy"):
            library._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_dispatch_overloads(monkeypatch):
    """Both functional overloads must be reachable, and must hit gems.

    `_cudnn_ctc_loss` (int[] lengths) and `_cudnn_ctc_loss.Tensor` have to be in
    `_FULL_CONFIG`; a registration regression is otherwise silent because
    `register_error` only logs. Comparing numbers alone cannot catch it either:
    an unregistered overload falls back to native cuDNN, which agrees with gems
    to ~1e-5, well inside the comparison tolerance. So each path is checked
    with a spy on the registered entry before any value assertion.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    # One (T, N, C, S) shape is enough: this test is about which overload the
    # dispatcher routes to, not about size coverage.
    shape = (16, 4, 16, 5)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    inp_len = torch.tensor(input_lengths, dtype=torch.int32, device=device)
    tgt_len = torch.tensor(target_lengths, dtype=torch.int32, device=device)

    direct_loss, direct_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, inp_len, tgt_len, 0, False, False
    )

    for overload, args in (
        ("_cudnn_ctc_loss", (log_probs, targets, input_lengths, target_lengths)),
        ("_cudnn_ctc_loss.Tensor", (log_probs, targets, inp_len, tgt_len)),
    ):
        calls = []
        _install_spy(monkeypatch, calls)
        entry = (
            torch.ops.aten._cudnn_ctc_loss
            if overload == "_cudnn_ctc_loss"
            else torch.ops.aten._cudnn_ctc_loss.Tensor
        )

        with _registered_cudnn_ctc_loss(["cudnn_ctc_loss"]):
            dispatch_loss, dispatch_grad = entry(*args, 0, False, False)

        assert calls, f"{overload} did not reach the Gems implementation"
        utils.gems_assert_close(dispatch_loss, direct_loss, torch.float32)
        utils.gems_assert_close(dispatch_grad, direct_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_out_reuses_registered_default(monkeypatch):
    """`.out` is served by the autogenerated composite over `.default`.

    There is no separate `.out` registration: enabling `_cudnn_ctc_loss` makes
    the native `.out` wrapper call it, then resize and copy the results. The
    spy proving that is the point -- with nothing registered, `.out` still
    returns correct-looking values from native cuDNN.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    # One (T, N, C, S) shape is enough: the assertion is about `.out` reusing
    # the registered `.default`, not about size coverage.
    shape = (16, 4, 16, 5)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_loss, ref_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    calls = []
    _install_spy(monkeypatch, calls)

    with _registered_cudnn_ctc_loss(["cudnn_ctc_loss"]):
        out0 = torch.empty_like(ref_loss)
        out1 = torch.empty_like(ref_grad)
        res0, res1 = torch.ops.aten._cudnn_ctc_loss.out(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            0,
            False,
            False,
            out0=out0,
            out1=out1,
        )

    assert calls, ".out fell back to native cuDNN instead of .default"
    # The composite must write and return the caller's buffers.
    assert res0.data_ptr() == out0.data_ptr()
    assert res1.data_ptr() == out1.data_ptr()
    utils.gems_assert_close(out0, ref_loss, torch.float32)
    utils.gems_assert_close(out1, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_native_autograd_scales_by_batch(monkeypatch):
    """Under dispatch, PyTorch's own autograd rule produces the gradient.

    With only the CUDA kernel registered, `_cudnn_ctc_loss_backward` scales the
    forward's raw gradient by each batch entry's upstream gradient and applies
    the `zero_infinity` mask. A uniform `.sum()` upstream hides a per-batch
    indexing bug, so the weights here are non-uniform.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 16, 3, 8
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_lengths = [2, 1, 1]
    weights = [1.0, 0.0, -2.0]

    ref_lp = log_probs.clone().requires_grad_(True)
    # cuDNN requires the targets on the CPU.
    ref_loss = torch.ops.aten._cudnn_ctc_loss(
        ref_lp,
        targets,
        [t_steps] * batch,
        target_lengths,
        0,
        True,
        False,
    )[0]
    ref_w = torch.tensor(weights, dtype=ref_lp.dtype, device=ref_lp.device)
    (ref_grad,) = torch.autograd.grad((ref_loss * ref_w).sum(), ref_lp)

    gems_lp = log_probs.clone().requires_grad_(True)
    _install_spy(monkeypatch, [])
    gems_w = torch.tensor(weights, dtype=log_probs.dtype, device=device)
    with _registered_cudnn_ctc_loss(["cudnn_ctc_loss"]):
        gems_loss = torch.ops.aten._cudnn_ctc_loss(
            gems_lp, targets, [t_steps] * batch, target_lengths, 0, True, False
        )[0]
        (gems_grad,) = torch.autograd.grad((gems_loss * gems_w).sum(), gems_lp)

    utils.gems_assert_close(gems_loss, ref_loss, torch.float32)
    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_zero_infinity():
    """`zero_infinity` must not change the reported loss.

    cuDNN ignores the flag: an infeasible alignment already comes back as a
    zero loss with a zero gradient whether it is set or not, unlike
    `F.ctc_loss`, which reports `inf` and needs the flag to zero it. This pins
    the cuDNN contract so a future "fix" toward the `F.ctc_loss` convention
    cannot land silently.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    # Three labels into two time steps can never be aligned: each label needs a
    # step, and adjacent repeats need a separating blank on top.
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    input_lengths = [2, 8]
    target_lengths = [3, 2]

    ref = torch.ops.aten._cudnn_ctc_loss(
        log_probs,
        targets.to(device),
        torch.tensor(input_lengths, dtype=torch.int32, device=device),
        torch.tensor(target_lengths, dtype=torch.int32, device=device),
        0,
        True,
        False,
    )

    for zero_infinity in (False, True):
        res_loss, res_grad = flag_gems.cudnn_ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            0,
            True,
            zero_infinity,
        )
        # Infeasible alignment: zero loss, zero gradient, no NaN.
        assert res_loss[0].item() == 0.0
        assert res_grad[:, 0].count_nonzero().item() == 0
        assert not torch.isnan(res_grad).any()
        utils.gems_assert_close(res_loss, ref[0], torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_infeasible_alignment():
    """An infeasible alignment yields zero loss and zero gradient.

    Both shapes below reach the terminal-state reduction with every state still
    `-inf`: a label sequence longer than the input, and adjacent repeated
    labels with no time step left for the separating blank. The loss must come
    back as zero (not `inf`), and the gradient must be all zeros (not `NaN`
    from `inf - inf`, and not the `exp(log_probs)` seed the empty-target path
    leaves in place).

    The gradient of the *infeasible* batch entry is asserted directly rather
    than compared against cuDNN: cuDNN does not write that entry's gradient
    buffer at all, so its value is uninitialized memory (measured to be
    whatever the allocator last handed out). The feasible trailing entry is
    still compared against cuDNN.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)

    cases = [
        # (targets, input_lengths, target_lengths)
        ([1, 2, 3, 4, 5], [2, 8], [3, 2]),
        ([2, 2, 4], [2, 8], [2, 1]),
    ]
    for targets, input_lengths, target_lengths in cases:
        tgt = torch.tensor(targets, dtype=torch.int32)
        inp_len = torch.tensor(input_lengths, dtype=torch.int32, device=device)
        tgt_len = torch.tensor(target_lengths, dtype=torch.int32, device=device)
        ref_loss, ref_grad = torch.ops.aten._cudnn_ctc_loss(
            log_probs, tgt.to(device), inp_len, tgt_len, 0, True, False
        )

        res_loss, res_grad = flag_gems.cudnn_ctc_loss(
            log_probs, tgt, input_lengths, target_lengths, 0, True, False
        )

        assert res_loss[0].item() == 0.0, (targets, res_loss.tolist())
        assert res_grad[:, 0].count_nonzero().item() == 0
        assert not torch.isnan(res_grad).any()
        utils.gems_assert_close(res_loss, ref_loss, torch.float32)
        # Batch 1 is alignable, so cuDNN does define its gradient.
        utils.gems_assert_close(res_grad[:, 1], ref_grad[:, 1], torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_empty_input():
    """An empty input with an empty target is the one infinite-loss case."""

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1], dtype=torch.int32)
    input_lengths = [0, 8]
    target_lengths = [0, 1]

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, True, False
    )

    assert torch.isinf(res_loss[0])
    assert res_grad[:, 0].count_nonzero().item() == 0


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize("empty_axis", ["time", "batch"])
def test_cudnn_ctc_loss_empty_axis_rejected(empty_axis):
    """Degenerate `log_probs` must raise instead of corrupting the context.

    cuDNN rejects both with `CUDNN_STATUS_BAD_PARAM` and leaves the device
    usable. The Triton path cannot match that by accident: `log_alpha` is
    `(N, T, BLOCK_S)`, so a zero `T` sends every batch entry to offset 0 of an
    empty buffer and the forward kernel's store faults, which poisons the whole
    CUDA context rather than failing one call. Checking the error type is
    enough here -- the point is that the raise happens before any kernel is
    launched, so the assertion after it proves the context survived.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    # Both degenerate axes, with the other dimension kept non-empty so the
    # rejection is attributable to the axis under test.
    shape = (0, 2, 6) if empty_axis == "time" else (8, 0, 6)
    log_probs = torch.empty(*shape, dtype=torch.float32, device=device)
    lengths = [0, 0] if empty_axis == "time" else []

    with pytest.raises(RuntimeError, match="positive"):
        flag_gems.cudnn_ctc_loss(
            log_probs,
            torch.empty(0, dtype=torch.int32),
            lengths,
            lengths,
            0,
            True,
            False,
        )

    # No kernel ran, so the device is still usable.
    assert torch.zeros(4, device=device).sum().item() == 0
