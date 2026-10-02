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

"""Correctness test for ``aten::is_vulkan_available``.

The operator is nullary (``aten::is_vulkan_available() -> bool``) and reports a
build-time property of the installed stack, so the dtype / shape / value-range /
broadcast / backward grid has no dimension to act on. The axes that do carry
information are swept instead: the ambient process state the query is issued in,
and the call index, because a pure capability query must answer the same Python
bool on every call. The schema-typed result is compared directly against the
native bool; no tensor buffer is fabricated for it.
"""

import contextlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import flag_gems

AMBIENT_STATES = (
    "plain",
    "no_grad",
    "inference_mode",
    "seeded",
    "single_thread",
    "after_cpu_allocation",
    "after_device_allocation",
    "after_device_kernel_launch",
    "default_dtype_float64",
    "default_dtype_float16",
    "default_device_active",
    "deterministic_algorithms",
    "matmul_precision_high",
    "under_tensor_dispatch_mode",
)

# The Nth call in one ambient state must answer like the first one.
CALL_INDICES = (1, 2, 3, 4, 5, 6, 7, 8)

AMBIENT_ROWS = [(state, index) for state in AMBIENT_STATES for index in CALL_INDICES]
# The schema takes no argument and has no ``out`` overload, so the negative rows
# exercise the call contract itself.
NEGATIVE_CALLS = [
    ((1,), {}),
    ((None,), {}),
    ((1, 2), {}),
    ((), {"device": torch.device("cpu")}),
    ((), {"out": None}),
]


class _PassThroughDispatchMode(TorchDispatchMode):
    """Tensor dispatch mode installed while the query runs."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def _snapshot_process_globals():
    """Capture every process-wide flag the states below touch.

    The deterministic-algorithms warn_only bit is captured next to the main flag
    because ``use_deterministic_algorithms()`` would otherwise reset it while
    restoring; the default generator state is captured with the seed it feeds.
    """
    return (
        torch.get_default_dtype(),
        torch.get_default_device(),
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.random.get_rng_state(),
    )


def _restore_process_globals(snapshot):
    dtype, device, threads, deterministic, warn_only, precision, rng_state = snapshot
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.random.set_rng_state(rng_state)


@contextlib.contextmanager
def _changed_global(apply):
    """Apply one process-wide setting, then restore all captured flags."""
    snapshot = _snapshot_process_globals()
    apply()
    try:
        yield
    finally:
        _restore_process_globals(snapshot)


@contextlib.contextmanager
def _held_allocation(device):
    """Keep a live allocation on `device` while the body runs."""
    held = torch.zeros(4, dtype=torch.float32, device=device)
    try:
        yield
    finally:
        del held


@contextlib.contextmanager
def _launched_device_kernel():
    """Run a real device kernel before the body."""
    left = torch.ones(8, dtype=torch.float32, device=flag_gems.device)
    total = left + 1
    try:
        yield
    finally:
        del left, total


def _ambient_state(name):
    """Enter the named ambient process state for the duration of the block."""
    if name == "plain":
        return contextlib.nullcontext()
    if name == "no_grad":
        return torch.no_grad()
    if name == "inference_mode":
        return torch.inference_mode()
    if name == "seeded":
        return _changed_global(lambda: torch.random.default_generator.manual_seed(0))
    if name == "single_thread":
        return _changed_global(lambda: torch.set_num_threads(1))
    if name == "after_cpu_allocation":
        return _held_allocation(torch.device("cpu"))
    if name == "after_device_allocation":
        return _held_allocation(flag_gems.device)
    if name == "after_device_kernel_launch":
        return _launched_device_kernel()
    if name == "default_dtype_float64":
        return _changed_global(lambda: torch.set_default_dtype(torch.float64))
    if name == "default_dtype_float16":
        return _changed_global(lambda: torch.set_default_dtype(torch.float16))
    if name == "default_device_active":
        return _changed_global(lambda: torch.set_default_device(flag_gems.device))
    if name == "deterministic_algorithms":
        return _changed_global(lambda: torch.use_deterministic_algorithms(True))
    if name == "matmul_precision_high":
        return _changed_global(lambda: torch.set_float32_matmul_precision("high"))
    if name == "under_tensor_dispatch_mode":
        return _PassThroughDispatchMode()
    raise ValueError(f"unknown ambient state: {name}")


@pytest.mark.is_vulkan_available
@pytest.mark.parametrize("state,index", AMBIENT_ROWS)
def test_is_vulkan_available_matches_native(state, index):
    with _ambient_state(state):
        expected = torch.ops.aten.is_vulkan_available()
        results = [flag_gems.is_vulkan_available() for _ in range(index)]

    # The schema returns a Python bool; every call must match the native answer.
    for result in results:
        assert type(result) is bool, f"expected bool, got {type(result).__name__}"
        assert result == expected, f"{result!r} != native {expected!r}"


@pytest.mark.is_vulkan_available
@pytest.mark.parametrize("args,kwargs", NEGATIVE_CALLS)
def test_is_vulkan_available_rejects_invalid_call(args, kwargs):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_vulkan_available(*args, **kwargs)
