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

"""Host-only checks for backend dispatch policy, without importing device code."""

import ast
import sys
import types
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registrar(monkeypatch):
    # Match the host-only registrar tests in test_dynamic_override.py: load the
    # real class with device/library stubs instead of importing a GPU runtime.
    tree = ast.parse((ROOT / "src/flag_gems/runtime/op_registrar.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GeneralOpRegistrar"
    )
    module = types.ModuleType("flag_gems")
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    device = types.SimpleNamespace(
        vendor_name="ascend", dispatch_key="PrivateUse1", vendor="ascend"
    )

    def fail(error):
        raise error

    scope = {
        "warnings": warnings,
        "DeviceDetector": lambda: device,
        "backend": types.SimpleNamespace(),
        "common": types.SimpleNamespace(
            vendors=types.SimpleNamespace(CAMBRICON="cambricon")
        ),
        "error": types.SimpleNamespace(register_error=fail),
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<registrar-policy>", "exec"),
        scope,
    )
    return scope["GeneralOpRegistrar"], module


@pytest.mark.parametrize("extra", [(), ("Autograd",), ("Autograd", object())])
def test_unannotated_functions_keep_shared_config(registrar, extra):
    cls, _ = registrar

    def candidate(x):
        return x

    item = ("clone", candidate, None, extra)
    assert cls._extra_dispatch_keys(item) == extra
    assert item[3] is extra
    assert cls._extra_dispatch_keys(("clone", candidate)) == ()
    candidate._gems_extra_dispatch_keys = None
    assert cls._extra_dispatch_keys(item) == extra


@pytest.mark.parametrize("override", [(), [], ("AutogradPrivateUse1",)])
def test_backend_policy_replaces_only_extra_keys(registrar, override):
    cls, _ = registrar

    def candidate(x):
        return x

    legacy = ("Autograd", object())
    candidate._gems_extra_dispatch_keys = override
    item = ("clone", candidate, None, legacy)
    assert cls._extra_dispatch_keys(item) == tuple(override)
    assert item[3] is legacy


@pytest.mark.parametrize("override", ["Autograd", False, 1])
def test_invalid_backend_policy_is_rejected(registrar, override):
    cls, _ = registrar

    def candidate(x):
        return x

    candidate._gems_extra_dispatch_keys = override
    with pytest.raises(TypeError, match="tuple or list"):
        cls._extra_dispatch_keys(("clone", candidate))


def test_live_candidate_retains_backend_registration_policy(registrar):
    cls, module = registrar

    def clone(x):
        return x

    def replacement(x):
        return x

    clone._gems_extra_dispatch_keys = ()
    item = ("clone", clone, None, ("Autograd", object()))
    module._FULL_CONFIG = (item,)
    module.clone = replacement
    instance = cls.__new__(cls)
    assert instance._normalized_config(item) == ("clone", replacement, ())


def test_backend_only_policy_still_registers_device_kernel(registrar):
    cls, module = registrar

    def clone(x):
        return x

    clone._gems_extra_dispatch_keys = ()
    item = ("clone", clone, None, ("Autograd", object()))
    module._FULL_CONFIG = (item,)
    module.clone = clone
    calls = []
    library = types.SimpleNamespace(impl=lambda *args: calls.append(args))
    instance = cls((item,), user_include_ops=["clone"], lib=library)
    assert calls == [("clone", clone, "PrivateUse1")]
    assert instance.get_all_keys() == ["clone"]
