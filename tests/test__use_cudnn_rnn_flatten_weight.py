# Copyright 2026, The FlagGems Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness tests for ``torch.ops.aten._use_cudnn_rnn_flatten_weight``.

``aten::_use_cudnn_rnn_flatten_weight() -> bool`` declares no parameter, so the
spec's dtype / value-range / shape / broadcast / backward grids have nothing to
attach to. The contract it does have is checked directly: strict ``bool``
agreement with the native predicate under the ambient configurations a caller
may have set, repeatability, no configuration side effects, and rejection of
every argument form (the numeric grid's special values appear as rejected
arguments).
"""

import contextlib
import itertools

import pytest
import torch

import flag_gems

from . import test_utils as tu

_GRAD_CONTEXTS = {
    "grad": torch.enable_grad,
    "no_grad": torch.no_grad,
    "inference_mode": torch.inference_mode,
}

_SWITCH_PATTERNS = (
    (False, False, False, False, False),
    (True, True, True, True, True),
    (True, False, True, False, True),
    (False, True, False, True, False),
)

# One case is one ambient configuration: grad mode, cudnn enabled, cudnn
# deterministic, cudnn benchmark, cudnn allow_tf32 and
# torch.use_deterministic_algorithms. The predicate reports a property of the
# build, not of the caller, so the candidate must answer it the same way in all
# of them.
_CONTEXT_CASES = [
    (grad_mode, *switches)
    for grad_mode in _GRAD_CONTEXTS
    for switches in itertools.product((True, False), repeat=5)
]

# Quick keeps four cheap configurations per grad mode: a query allocates and
# computes nothing, so narrowing the switch sweep costs no accuracy coverage.
_QUICK_CONTEXTS = [
    (grad_mode, *switches)
    for grad_mode in _GRAD_CONTEXTS
    for switches in _SWITCH_PATTERNS
]

_REJECTED_SCALARS = (True, 0, 1, -1, 0.0, 1.5, -1.5, "cudnn", None)

_REJECTED_KEYWORDS = ("inp", "training", "enabled")


def _ambient_state():
    # The switches a capability query must leave exactly as it found them.
    return (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.is_grad_enabled(),
    )


@contextlib.contextmanager
def _ambient_context(case):
    """Apply one ambient configuration and restore it exactly afterwards."""
    grad_mode, enabled, deterministic, benchmark, allow_tf32, det_algos = case
    saved = _ambient_state()
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = allow_tf32
    # Keep the ambient warn_only while setting the mode, so the row only varies
    # the enabled half.
    torch.use_deterministic_algorithms(det_algos, warn_only=saved[5])
    try:
        with _GRAD_CONTEXTS[grad_mode]():
            yield
    finally:
        (
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
        ) = saved[:4]
        # Enabling the mode clears warn_only, so restore both halves together.
        torch.use_deterministic_algorithms(saved[4], warn_only=saved[5])


@contextlib.contextmanager
def _cudnn_toggle(mechanism, enabled):
    """Flip cuDNN through each switch a caller may use."""
    if mechanism == "attribute":
        saved = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = enabled
        try:
            yield
        finally:
            torch.backends.cudnn.enabled = saved
    elif mechanism == "flags":
        with torch.backends.cudnn.flags(enabled=enabled):
            yield
    else:
        saved = torch.backends.cudnn.enabled
        torch._C._set_cudnn_enabled(enabled)
        try:
            yield
        finally:
            torch._C._set_cudnn_enabled(saved)


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("case", _CONTEXT_CASES)
def test__use_cudnn_rnn_flatten_weight_matches_native(case):
    with _ambient_context(case):
        ref = torch.ops.aten._use_cudnn_rnn_flatten_weight()
        res = flag_gems._use_cudnn_rnn_flatten_weight()
    assert type(res) is bool
    assert res == ref


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("mechanism", ("attribute", "flags", "c_api"))
@pytest.mark.parametrize("enabled", (True, False))
def test__use_cudnn_rnn_flatten_weight_with_cudnn_toggle(mechanism, enabled):
    # A caller may disable cuDNN through any of these switches; the query still
    # reports the build capability, so the candidate must agree with native.
    with _cudnn_toggle(mechanism, enabled):
        ref = torch.ops.aten._use_cudnn_rnn_flatten_weight()
        res = flag_gems._use_cudnn_rnn_flatten_weight()
    assert type(res) is bool
    assert res == ref


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("repeats", (2, 4, 8))
def test__use_cudnn_rnn_flatten_weight_is_repeatable(repeats):
    # Consecutive calls, each made under a different ambient configuration, must
    # agree: the predicate is a global capability query, not a stateful decision
    # cached from the first call.
    ref = torch.ops.aten._use_cudnn_rnn_flatten_weight()
    results = []
    for index in range(repeats):
        with _ambient_context(_CONTEXT_CASES[index]):
            results.append(flag_gems._use_cudnn_rnn_flatten_weight())
    assert [type(value) for value in results] == [bool] * repeats, results
    assert all(value is ref for value in results), results


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("case", _QUICK_CONTEXTS)
def test__use_cudnn_rnn_flatten_weight_has_no_side_effect(case):
    # A capability query must not perturb any switch it reports on, including the
    # warn_only half of the deterministic-algorithms state.
    with _ambient_context(case):
        before = _ambient_state()
        flag_gems._use_cudnn_rnn_flatten_weight()
        after = _ambient_state()
    assert after == before


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test__use_cudnn_rnn_flatten_weight_rejects_tensor_argument(dtype):
    # ``() -> bool`` takes no operand, so every dtype the spec lists is an
    # invalid argument. These rows make no claim about dtype support.
    inp = torch.zeros(2, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._use_cudnn_rnn_flatten_weight(inp)


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("value", (0.0, float("nan"), float("inf")))
def test__use_cudnn_rnn_flatten_weight_rejects_special_argument(value):
    # There is no numeric operand, so nan/inf cannot appear as an input value;
    # the spec's special-value dimension is kept as rejected arguments.
    inp = torch.full((2,), value, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._use_cudnn_rnn_flatten_weight(inp)


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("value", _REJECTED_SCALARS)
def test__use_cudnn_rnn_flatten_weight_rejects_scalar_argument(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._use_cudnn_rnn_flatten_weight(value)


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("arity", (1, 2, 3, 4))
def test__use_cudnn_rnn_flatten_weight_rejects_extra_positional(arity):
    args = [torch.empty(0, device=flag_gems.device) for _ in range(arity)]
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._use_cudnn_rnn_flatten_weight(*args)


@pytest.mark.use_cudnn_rnn_flatten_weight
@pytest.mark.parametrize("name", _REJECTED_KEYWORDS)
def test__use_cudnn_rnn_flatten_weight_rejects_keyword_argument(name):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._use_cudnn_rnn_flatten_weight(**{name: True})
