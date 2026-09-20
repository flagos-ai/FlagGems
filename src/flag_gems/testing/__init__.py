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

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Callable, Iterator, Optional

import torch

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

_GEMS_OP_OVERRIDE_LOCK = RLock()
_GEMS_OP_OVERRIDES = {}
_MISSING = object()
_CURRENT_GEMS_OP_CASE = ContextVar("flag_gems_current_op_case", default=None)


def _validate_gems_op(operator: str, function: Optional[Callable] = None) -> None:
    if not isinstance(operator, str) or not operator:
        raise ValueError("operator must be a non-empty public operator name.")
    if function is not None and not callable(function):
        raise TypeError("gems_op must be callable.")


@contextmanager
def override_gems_op(operator: str, replacement: Callable) -> Iterator[Callable]:
    """Temporarily override the direct callable resolved for one public operator.

    This registry is independent of the PyTorch dispatcher registration tables.
    It is intended for native tests and benchmarks that need to run the same
    candidate callable through a deterministic, process-local route.
    """

    _validate_gems_op(operator, replacement)
    with _GEMS_OP_OVERRIDE_LOCK:
        previous = _GEMS_OP_OVERRIDES.get(operator, _MISSING)
        _GEMS_OP_OVERRIDES[operator] = replacement
    try:
        yield replacement
    finally:
        with _GEMS_OP_OVERRIDE_LOCK:
            if previous is _MISSING:
                _GEMS_OP_OVERRIDES.pop(operator, None)
            else:
                _GEMS_OP_OVERRIDES[operator] = previous


def resolve_gems_op(operator: str, default: Optional[Callable] = None) -> Callable:
    """Resolve an override or the operator's normal direct FlagGems callable."""

    _validate_gems_op(operator, default)
    with _GEMS_OP_OVERRIDE_LOCK:
        override = _GEMS_OP_OVERRIDES.get(operator, _MISSING)
    if override is not _MISSING:
        return override
    if default is None:
        # Imported lazily because this module is imported while flag_gems is
        # still constructing its public namespace.
        import flag_gems as package

        default = getattr(package, operator, None)
    if not callable(default):
        raise LookupError(f"No direct FlagGems callable for '{operator}'.")
    return default


def gems_op_source(operator: str, function: Callable) -> str:
    """Describe whether ``function`` came from the process-local override."""

    _validate_gems_op(operator, function)
    with _GEMS_OP_OVERRIDE_LOCK:
        override = _GEMS_OP_OVERRIDES.get(operator, _MISSING)
    return "override" if override is function else "default"


@contextmanager
def gems_op_case(operator: str, case_id: Optional[str]) -> Iterator[None]:
    """Expose the active timing case to an optional candidate observer."""

    _validate_gems_op(operator)
    token = _CURRENT_GEMS_OP_CASE.set((operator, case_id))
    try:
        yield
    finally:
        _CURRENT_GEMS_OP_CASE.reset(token)


def current_gems_op_case(operator: Optional[str] = None) -> Optional[str]:
    """Return the active timing case ID, if one is being materialized."""

    current = _CURRENT_GEMS_OP_CASE.get()
    if current is None:
        return None
    current_operator, case_id = current
    if operator is not None and operator != current_operator:
        return None
    return case_id


if runtime.device.vendor_name == "kunlunxin":
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
        torch.complex128: 1e-7,
    }
else:
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
        torch.complex128: 1e-7,
    }


def _maybe_move_to_cpu(res, ref):
    if res.device.type == "cpu" or ref.device.type == "cpu":
        return res, ref

    required = res.numel() * res.element_size()

    free_mem = None
    try:
        free_mem, _ = torch_device_fn.mem_get_info(res.device)
    except RuntimeError:
        pass

    # torch.isclose allocates an auxiliary tensor roughly the size of the inputs,
    # so ensure we have enough headroom; otherwise compare on CPU.
    HUGE_TENSOR_BYTES = 1 << 30  # 1 GiB
    if (free_mem is not None and required >= free_mem) or (
        required >= HUGE_TENSOR_BYTES
    ):
        return res.cpu(), ref.cpu()
    return res, ref


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4):
    if dtype is None:
        dtype = torch.float32
    assert res.dtype == dtype
    ref = ref.to(dtype)
    res, ref = _maybe_move_to_cpu(res, ref)
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(
        res, ref, atol=atol * reduce_dim, rtol=rtol, equal_nan=equal_nan
    )


def assert_equal(res, ref, equal_nan=False):
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)
