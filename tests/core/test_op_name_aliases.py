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

"""Host-only registrar tests; run pytest with --confcutdir=tests/core."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def registrar(monkeypatch):
    package = "_alias_test_runtime"
    runtime = types.ModuleType(package)
    runtime.__path__ = []
    runtime.backend = types.SimpleNamespace(get_unused_ops=lambda vendor: [])
    runtime.common = types.SimpleNamespace(
        vendors=types.SimpleNamespace(CAMBRICON="cambricon")
    )
    runtime.error = types.SimpleNamespace(
        register_error=lambda exc: pytest.fail(str(exc))
    )
    backend = types.ModuleType(package + ".backend")
    backend.__path__ = []
    finder = types.ModuleType(package + ".backend.device_finder")
    finder.DeviceDetector = lambda: types.SimpleNamespace(
        dispatch_key="PrivateUse1", vendor="mthreads", vendor_name="mthreads"
    )
    for name, module in (
        (package, runtime),
        (package + ".backend", backend),
        (package + ".backend.device_finder", finder),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "src/flag_gems/runtime/op_registrar.py"
    spec = importlib.util.spec_from_file_location(package + ".op_registrar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneralOpRegistrar, runtime.backend


def _unique2():
    pass


def unique_dim():
    pass


def unique_consecutive():
    pass


CONFIG = [(fn.__name__, fn) for fn in (_unique2, unique_dim, unique_consecutive)]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("names", [["unique"], ["_unique2"], ["unique", "_unique2"]])
def test_include_registers_canonical_key_once(registrar, cached, names):
    cls, _ = registrar
    lib = Mock()
    mapping = {item[0]: [item] for item in CONFIG} if cached else None
    result = cls(CONFIG, user_include_ops=names, lib=lib, full_config_by_func=mapping)
    lib.impl.assert_called_once_with("_unique2", _unique2, "PrivateUse1")
    assert result.get_all_keys() == ["_unique2"]


@pytest.mark.parametrize("vendor", [False, True])
def test_exclude_alias_preserves_other_unique_variants(registrar, vendor):
    cls, backend = registrar
    if vendor:
        backend.get_unused_ops = lambda name: ["unique"]
    result = cls(CONFIG, user_exclude_ops=[] if vendor else ["unique"], lib=Mock())
    assert result.get_all_keys() == ["unique_dim", "unique_consecutive"]


def test_alias_respects_cpp_patched_ops(registrar):
    cls, _ = registrar
    with pytest.warns(UserWarning, match="No op to register"):
        result = cls(
            CONFIG,
            user_include_ops=["unique"],
            cpp_patched_ops=["_unique2"],
            lib=Mock(),
        )
    assert result.get_all_keys() == []
