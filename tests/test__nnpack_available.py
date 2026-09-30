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

"""Correctness tests for ``aten::_nnpack_available``: ``() -> bool`` reports
whether this PyTorch build was compiled with NNPACK.

The operator is a host-side build-capability query with no operands, so the tests
pin the properties it actually has: the schema-declared Python bool result,
agreement with the native query under different ambient torch states, thread
independence, absence of global side effects, and the arity contract.
"""

import threading
from contextlib import contextmanager

import pytest
import torch

import flag_gems


def _ambient_state_snapshot():
    return {
        "default_dtype": torch.get_default_dtype(),
        "grad_enabled": torch.is_grad_enabled(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "num_threads": torch.get_num_threads(),
        # The runtime NNPACK flag has no public reader, so use the same accessor
        # torch.backends.nnpack itself reads.
        "nnpack_enabled": torch._C._get_nnpack_enabled(),
    }


@contextmanager
def _ambient(state, value):
    """Run the body under one ambient torch state and restore it afterwards."""
    if state == "plain":
        yield
    elif state == "no_grad":
        with torch.no_grad():
            yield
    elif state == "inference_mode":
        with torch.inference_mode():
            yield
    elif state == "default_dtype":
        previous = torch.get_default_dtype()
        torch.set_default_dtype(value)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)
    elif state == "num_threads":
        previous = torch.get_num_threads()
        torch.set_num_threads(value)
        try:
            yield
        finally:
            torch.set_num_threads(previous)
    elif state == "deterministic":
        # Restore both halves of the deterministic-algorithms setting.
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(value, warn_only=warn_only)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    elif state == "autocast":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
    elif state == "nnpack_flag":
        # The runtime "enabled" flag is not the build capability the schema
        # reports, so a candidate returning that flag instead is wrong. The
        # context manager restores the previous value on exit.
        with torch.backends.nnpack.flags(enabled=value):
            yield
    else:
        raise AssertionError(f"unknown ambient state {state!r}")


# The operator has no operand and no parameter, so no value-range, shape, dtype,
# broadcast, backward or param case can be built from it; the ambient states below
# are the operator-relevant dimension. All rows are tiny and run at both levels.
_AMBIENT_ROWS = [
    pytest.param("plain", None, id="plain"),
    pytest.param("no_grad", None, id="no_grad"),
    pytest.param("inference_mode", None, id="inference_mode"),
    pytest.param("default_dtype", torch.float16, id="default_dtype_fp16"),
    pytest.param("default_dtype", torch.float64, id="default_dtype_fp64"),
    pytest.param("default_dtype", torch.bfloat16, id="default_dtype_bf16"),
    pytest.param("num_threads", 1, id="single_thread"),
    pytest.param("num_threads", 2, id="two_threads"),
    pytest.param("deterministic", True, id="deterministic_algorithms"),
    pytest.param("autocast", None, id="autocast_cpu_bf16"),
    pytest.param("nnpack_flag", False, id="nnpack_runtime_flag_disabled"),
    pytest.param("nnpack_flag", True, id="nnpack_runtime_flag_enabled"),
]

AMBIENT_CASES = _AMBIENT_ROWS


@pytest.mark.nnpack_available
def test__nnpack_available():
    ref = torch.ops.aten._nnpack_available()
    res = flag_gems._nnpack_available()

    assert type(res) is bool
    assert res == ref


@pytest.mark.nnpack_available
def test__nnpack_available_is_stable_across_calls():
    # The answer describes the build, so repeated queries in one process must not
    # drift and must not be served from a stale cached value.
    ref = torch.ops.aten._nnpack_available()

    for _ in range(4):
        res = flag_gems._nnpack_available()
        assert type(res) is bool
        assert res == ref


@pytest.mark.nnpack_available
@pytest.mark.parametrize("state,value", AMBIENT_CASES)
def test__nnpack_available_ignores_ambient_state(state, value):
    with _ambient(state, value):
        ref = torch.ops.aten._nnpack_available()
        res = flag_gems._nnpack_available()

        assert type(res) is bool
        assert res == ref


@pytest.mark.nnpack_available
def test__nnpack_available_from_worker_thread():
    # A build capability must not be reported from main-thread-only state.
    ref = torch.ops.aten._nnpack_available()
    observed = []
    errors = []

    def _query():
        try:
            observed.append(flag_gems._nnpack_available())
        except Exception as exc:  # surfaced below instead of vanishing in the thread
            errors.append(exc)

    thread = threading.Thread(target=_query)
    thread.start()
    thread.join()

    assert not errors, errors
    assert len(observed) == 1
    res = observed[0]
    assert type(res) is bool
    assert res == ref


@pytest.mark.nnpack_available
def test__nnpack_available_has_no_global_side_effects():
    # A capability query must not perturb the ambient state it observes, nor
    # consume global RNG state.
    before = _ambient_state_snapshot()
    rng_before = torch.get_rng_state()

    ref = torch.ops.aten._nnpack_available()
    res = flag_gems._nnpack_available()

    assert type(res) is bool
    assert res == ref
    assert _ambient_state_snapshot() == before
    assert torch.equal(torch.get_rng_state(), rng_before)


# The native schema declares no argument: any operand is rejected at dispatch
# (probed: RuntimeError "aten::_nnpack_available() expected at most 0 argument(s)
# but received 1 argument(s)"). A candidate must reject the same calls instead of
# silently ignoring an operand. These rows are kept in both levels.
REJECTED_CALLS = [
    pytest.param((7,), {}, id="positional_int"),
    pytest.param((1.5,), {}, id="positional_float"),
    pytest.param((True,), {}, id="positional_bool"),
    pytest.param(("cpu",), {}, id="positional_str"),
    pytest.param((None,), {}, id="positional_none"),
    pytest.param((7, 8), {}, id="two_positionals"),
    pytest.param((), {"device": torch.device("cpu")}, id="keyword_argument"),
]


@pytest.mark.nnpack_available
@pytest.mark.parametrize("args,kwargs", REJECTED_CALLS)
def test__nnpack_available_rejects_arguments(args, kwargs):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._nnpack_available(*args, **kwargs)


@pytest.mark.nnpack_available
def test__nnpack_available_rejects_tensor_operand():
    # A tensor operand is the realistic mistake for a pointwise-style operator.
    operand = torch.zeros(4, device=flag_gems.device)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._nnpack_available(operand)
