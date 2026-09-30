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

from . import test_utils as tu

# aten::is_pinned(Tensor self, Device? device=None) -> bool reports which host
# allocator backs the tensor storage: a Python bool with no elementwise result,
# no gradient and no broadcast form. Covered axes: the storage state (pinned
# host / plain host / backend / meta / empty), the view-versus-new-storage
# relation, the lazy conjugate view state and the optional device argument.
# Host fixtures pass device='cpu' explicitly, backend fixtures use
# flag_gems.device and meta fixtures stay on the meta device, so no ambient
# default device can move a fixture.
#
# Fixtures are payload-free torch.empty allocations: the operator reads no
# element, so the shared value-range framework -- which fills the whole buffer
# through torch.testing.make_tensor -- adds no coverage here, and no
# uninitialized content is ever compared. tu.to_reference is not usable for
# these operands: it rebuilds them through torch.empty(0).set_(storage.clone(),
# ...) and that rebuilt tensor reports is_pinned() == False even when the cloned
# storage is pinned.

# Static capability flags read at import: no tensor is allocated at collection
# time and no test probes support at run time.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]
_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# Location of the active non-CPU backend; None on a CPU backend, where the
# backend-location rows cannot be expressed.
_REMOTE_DEVICE = None if flag_gems.device == "cpu" else flag_gems.device


def _assert_matches_native(res_out, ref_out):
    # aten::is_pinned returns a Python bool, not a tensor, so the shared tensor
    # assertions do not apply; compare the exact type and value directly.
    assert type(res_out) is bool, type(res_out)
    assert res_out is ref_out


def _storage_tensor(storage, shape, dtype):
    if storage == "pinned-host":
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    if storage == "plain-host":
        return torch.empty(shape, dtype=dtype, device="cpu")
    if storage == "backend":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device)
    raise ValueError(storage)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_pinned_host_storage(shape, dtype):
    inp = _storage_tensor("pinned-host", shape, dtype)
    ref_inp = _storage_tensor("pinned-host", shape, dtype)

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is True  # no spec shape is empty; empty storage is separate
    _assert_matches_native(res_out, ref_out)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_plain_host_storage(shape, dtype):
    inp = _storage_tensor("plain-host", shape, dtype)
    ref_inp = _storage_tensor("plain-host", shape, dtype)

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is False
    _assert_matches_native(res_out, ref_out)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_backend_storage(shape, dtype):
    inp = _storage_tensor("backend", shape, dtype)
    ref_inp = _storage_tensor("backend", shape, dtype)

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is False  # device-resident storage is never pinned
    _assert_matches_native(res_out, ref_out)


# Every cheap call/storage form of the optional device argument: default
# omission, explicit None, a device object, a device string, positional and
# keyword passing, and both the True and the False outcome. All rows stay in
# quick mode.
_DEVICE_MATRIX_ROWS = [
    pytest.param("pinned-host", None, "omitted", True, id="pinned-host/omitted"),
    pytest.param(
        "pinned-host", None, "positional", True, id="pinned-host/none-positional"
    ),
    pytest.param("pinned-host", None, "keyword", True, id="pinned-host/none-keyword"),
    pytest.param(
        "pinned-host",
        torch.device("cpu"),
        "positional",
        False,
        id="pinned-host/cpu-device-positional",
    ),
    pytest.param(
        "pinned-host",
        torch.device("cpu"),
        "keyword",
        False,
        id="pinned-host/cpu-device-keyword",
    ),
    pytest.param(
        "pinned-host", "cpu", "positional", False, id="pinned-host/cpu-str-positional"
    ),
    pytest.param(
        "pinned-host", "cpu", "keyword", False, id="pinned-host/cpu-str-keyword"
    ),
    pytest.param("plain-host", None, "omitted", False, id="plain-host/omitted"),
    pytest.param(
        "plain-host",
        torch.device("cpu"),
        "positional",
        False,
        id="plain-host/cpu-device-positional",
    ),
    pytest.param("backend", None, "omitted", False, id="backend/omitted"),
    pytest.param(
        "backend",
        torch.device("cpu"),
        "positional",
        False,
        id="backend/cpu-device-positional",
    ),
]
if _REMOTE_DEVICE is not None:
    _DEVICE_MATRIX_ROWS += [
        pytest.param(
            "pinned-host",
            torch.device(_REMOTE_DEVICE),
            "positional",
            True,
            id="pinned-host/backend-device-positional",
        ),
        pytest.param(
            "pinned-host",
            torch.device(_REMOTE_DEVICE),
            "keyword",
            True,
            id="pinned-host/backend-device-keyword",
        ),
        pytest.param(
            "pinned-host",
            _REMOTE_DEVICE,
            "positional",
            True,
            id="pinned-host/backend-str-positional",
        ),
        pytest.param(
            "pinned-host",
            _REMOTE_DEVICE,
            "keyword",
            True,
            id="pinned-host/backend-str-keyword",
        ),
        pytest.param(
            "plain-host",
            torch.device(_REMOTE_DEVICE),
            "positional",
            False,
            id="plain-host/backend-device-positional",
        ),
        pytest.param(
            "backend",
            torch.device(_REMOTE_DEVICE),
            "positional",
            False,
            id="backend/backend-device-positional",
        ),
    ]


@pytest.mark.is_pinned
@pytest.mark.parametrize("storage,device_arg,call_form,expected", _DEVICE_MATRIX_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_device_argument(storage, device_arg, call_form, expected, dtype):
    inp = _storage_tensor(storage, (64, 64), dtype)
    ref_inp = _storage_tensor(storage, (64, 64), dtype)

    if call_form == "omitted":
        ref_out = torch.ops.aten.is_pinned(ref_inp)
        res_out = flag_gems.is_pinned(inp)
    elif call_form == "keyword":
        ref_out = torch.ops.aten.is_pinned(ref_inp, device=device_arg)
        res_out = flag_gems.is_pinned(inp, device=device_arg)
    else:
        ref_out = torch.ops.aten.is_pinned(ref_inp, device_arg)
        res_out = flag_gems.is_pinned(inp, device_arg)

    assert res_out is expected
    _assert_matches_native(res_out, ref_out)


_VIEW_BASE_SHAPE = (4, 8)

_VIEW_ROWS = [
    pytest.param("transpose", id="transpose"),
    pytest.param("narrow", id="narrow"),
    pytest.param("step-slice", id="step-slice"),
    pytest.param("expand", id="expand"),
    pytest.param("as_strided-offset", id="as_strided-offset"),
    pytest.param("select", id="select"),
    pytest.param("diagonal", id="diagonal"),
    pytest.param("detach", id="detach"),
    pytest.param("unsqueeze", id="unsqueeze"),
]

_DERIVED_ROWS = [
    pytest.param("clone", id="clone"),
    pytest.param("contiguous-of-transpose", id="contiguous-of-transpose"),
    pytest.param("empty-like", id="empty-like"),
]
if _REMOTE_DEVICE is not None:
    _DERIVED_ROWS.append(pytest.param("to-backend", id="to-backend"))

_PIN_MEMORY_ROWS = [
    pytest.param("flat", True, id="flat"),
    pytest.param("non-contiguous-view", True, id="non-contiguous-view"),
    pytest.param("zero-dim", True, id="zero-dim"),
    pytest.param("empty", False, id="empty"),
]


def _view(base, transform):
    if transform == "transpose":
        return base.transpose(-1, -2)
    if transform == "narrow":
        return base.narrow(0, 1, 2)
    if transform == "step-slice":
        return base[:, ::2]
    if transform == "expand":
        return base[:1].expand(4, 8)
    if transform == "as_strided-offset":
        return base.as_strided((2, 2), (8, 1), 9)
    if transform == "select":
        return base.select(0, 1)
    if transform == "diagonal":
        return base.diagonal()
    if transform == "detach":
        return base.detach()
    if transform == "unsqueeze":
        return base.unsqueeze(0)
    raise ValueError(transform)


def _derived(base, derivation):
    if derivation == "clone":
        return base.clone()
    if derivation == "contiguous-of-transpose":
        return base.transpose(-1, -2).contiguous()
    if derivation == "empty-like":
        return torch.empty_like(base)
    if derivation == "to-backend":
        return base.to(flag_gems.device)
    raise ValueError(derivation)


@pytest.mark.is_pinned
@pytest.mark.parametrize("transform", _VIEW_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_views_keep_storage_pinning(transform, dtype):
    base = torch.empty(_VIEW_BASE_SHAPE, dtype=dtype, device="cpu", pin_memory=True)
    ref_base = torch.empty(_VIEW_BASE_SHAPE, dtype=dtype, device="cpu", pin_memory=True)
    view = _view(base, transform)
    ref_view = _view(ref_base, transform)

    ref_out = torch.ops.aten.is_pinned(ref_view)
    res_out = flag_gems.is_pinned(view)

    assert res_out is True  # a view reports the storage it was taken from
    _assert_matches_native(res_out, ref_out)
    # Post-query preservation: the operand still addresses its base storage.
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


@pytest.mark.is_pinned
@pytest.mark.parametrize("derivation", _DERIVED_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_new_storage_is_not_pinned(derivation, dtype):
    base = torch.empty(_VIEW_BASE_SHAPE, dtype=dtype, device="cpu", pin_memory=True)
    ref_base = torch.empty(_VIEW_BASE_SHAPE, dtype=dtype, device="cpu", pin_memory=True)
    derived = _derived(base, derivation)
    ref_derived = _derived(ref_base, derivation)

    ref_out = torch.ops.aten.is_pinned(ref_derived)
    res_out = flag_gems.is_pinned(derived)

    assert res_out is False  # page locking is not inherited by new storage
    _assert_matches_native(res_out, ref_out)
    assert derived.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()


def _pin_memory_state(host, state):
    if state == "flat":
        return host.pin_memory()
    if state == "non-contiguous-view":
        return host[:, ::2].pin_memory()
    if state == "zero-dim":
        return host[0, 0].pin_memory()
    if state == "empty":
        return torch.empty(0, dtype=host.dtype, device="cpu", pin_memory=True)
    raise ValueError(state)


@pytest.mark.is_pinned
@pytest.mark.parametrize("state,expected", _PIN_MEMORY_ROWS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_pin_memory_states(state, expected, dtype):
    host = torch.empty((2, 16), dtype=dtype, device="cpu")
    ref_host = torch.empty((2, 16), dtype=dtype, device="cpu")
    inp = _pin_memory_state(host, state)
    ref_inp = _pin_memory_state(ref_host, state)

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is expected
    _assert_matches_native(res_out, ref_out)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", [(0,), (0, 4)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_empty_storage(shape, dtype):
    # Zero elements cannot own page-locked memory, and an empty payload is never
    # read, so no uninitialized content is compared.
    inp = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    ref_inp = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is False
    _assert_matches_native(res_out, ref_out)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_pinned_meta_storage(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device="meta")
    ref_inp = torch.empty(shape, dtype=dtype, device="meta")

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is False  # meta storage has no host allocator
    _assert_matches_native(res_out, ref_out)


@pytest.mark.is_pinned
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_is_pinned_conjugate_view_of_pinned_complex(shape):
    # The lazy conjugate bit is an extra view state. This build reports False for
    # the conjugated view of a pinned complex storage, so the measured native
    # answer is the expectation instead of a hard-coded literal.
    base = torch.empty(shape, dtype=torch.complex64, device="cpu", pin_memory=True)
    ref_base = torch.empty(shape, dtype=torch.complex64, device="cpu", pin_memory=True)
    conj = base.conj()
    ref_conj = ref_base.conj()

    ref_out = torch.ops.aten.is_pinned(ref_conj)
    res_out = flag_gems.is_pinned(conj)

    _assert_matches_native(res_out, ref_out)
    # Post-query preservation of the lazy bit and the shared storage.
    assert conj.is_conj()
    assert conj.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


# Default-only: quick carries no positive special-value case.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.is_pinned
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_is_pinned_special_payload(dtype, scenario):
    # The shared helper drops the inf scenarios the dtype cannot represent.
    inp = tu.make_special_input(dtype, scenario).cpu().pin_memory()
    ref_inp = tu.make_special_input(dtype, scenario).cpu().pin_memory()

    ref_out = torch.ops.aten.is_pinned(ref_inp)
    res_out = flag_gems.is_pinned(inp)

    assert res_out is True
    _assert_matches_native(res_out, ref_out)


_INVALID_SELF_ROWS = [
    pytest.param([1, 2], id="list"),
    pytest.param(7, id="int"),
    pytest.param(None, id="none"),
    pytest.param("not a tensor", id="str"),
]


@pytest.mark.is_pinned
@pytest.mark.parametrize("invalid_self", _INVALID_SELF_ROWS)
def test_is_pinned_invalid_self(invalid_self):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_pinned(invalid_self)


@pytest.mark.is_pinned
def test_is_pinned_missing_self():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_pinned()


_INVALID_DEVICE_ROWS = [
    pytest.param(0, id="int"),
    pytest.param(0.5, id="float"),
    pytest.param("no-such-device", id="str"),
]


@pytest.mark.is_pinned
@pytest.mark.parametrize("invalid_device", _INVALID_DEVICE_ROWS)
def test_is_pinned_invalid_device(invalid_device):
    host = torch.empty((8,), dtype=torch.float32, device="cpu")

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_pinned(host, invalid_device)
