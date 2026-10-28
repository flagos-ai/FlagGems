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
import threading
from types import SimpleNamespace

import pytest

lib = importlib.import_module("flag_gems.utils.libentry")
settings = importlib.import_module("flag_gems.runtime.flagtune")


def entry_with(apply):
    entry = object.__new__(lib.LibEntry)
    entry.fn = SimpleNamespace(apply_flagtune=apply, fn=None)
    entry.lock = threading.Lock()
    entry._flagtune_state = None
    entry._flagtune_model_ids = ()
    entry.kernel_cache = ({"old": 1}, {"old": 1})
    entry._cpu_cache = {"old": 1}
    return entry


def test_stable_state_skips_mode_resolution_and_device_changes_recheck():
    calls = []
    entry = entry_with(lambda: calls.append(1) or False)
    for _ in range(20):
        entry._apply_flagtune(0)
    assert len(calls) == 1
    entry._apply_flagtune(1)
    assert len(calls) == 2
    entry._apply_flagtune("cpu")
    assert len(calls) == 3
    assert entry.kernel_cache[0] == {"old": 1}


@pytest.mark.parametrize(
    "name",
    ["USE_FLAGTUNE", "USE_FLAGTUNE_COST_MODEL", "FLAGTUNE_INCLUDE"],
)
def test_environment_changes_recheck_and_removal_restores_state(monkeypatch, name):
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv(name, raising=False)
    calls = []
    entry = entry_with(lambda: calls.append(1) or False)
    entry._apply_flagtune(0)
    monkeypatch.setenv(name, "changed")
    entry._apply_flagtune(0)
    entry._apply_flagtune(0)
    monkeypatch.delenv(name)
    entry._apply_flagtune(0)
    assert len(calls) == 3


def test_changed_mode_clears_compiled_entries():
    entry = entry_with(lambda: True)
    entry._apply_flagtune(0)
    assert not any(entry.kernel_cache)
    assert not entry._cpu_cache


def test_failed_resolution_is_not_cached():
    calls = []

    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("invalid setting")
        return False

    entry = entry_with(fail_once)
    with pytest.raises(ValueError, match="invalid setting"):
        entry._apply_flagtune(0)
    assert entry._flagtune_state is None
    entry._apply_flagtune(0)
    entry._apply_flagtune(0)
    assert len(calls) == 2


def test_api_include_state_and_registry_changes_invalidate(monkeypatch):
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    calls = []
    entry = entry_with(lambda: calls.append(1) or False)
    entry._apply_flagtune(0)
    monkeypatch.setattr(settings, "_include_ops", frozenset(["mm"]))
    entry._apply_flagtune(0)
    monkeypatch.setattr(settings, "_registry_revision", settings._registry_revision + 1)
    entry._apply_flagtune(0)
    assert len(calls) == 3


def test_live_mode_switches_preserve_default_cost_and_expanded(monkeypatch):
    monkeypatch.setattr(settings, "_include_ops", None)
    monkeypatch.setenv("FLAGTUNE_INCLUDE", "mm")
    modes = []

    def apply():
        modes.append(settings.resolve_tuning_mode("mm", supports_cost_model=True))
        return True

    entry = entry_with(apply)
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    entry._apply_flagtune(0)
    monkeypatch.delenv("USE_FLAGTUNE")
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")
    entry._apply_flagtune(0)
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    entry._apply_flagtune(0)
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    entry._apply_flagtune(0)
    assert modes == [
        settings.TuningMode.DEFAULT,
        settings.TuningMode.COST_MODEL,
        settings.TuningMode.EXPANDED,
        settings.TuningMode.DEFAULT,
    ]


def test_invalid_live_setting_raises_until_fixed(monkeypatch):
    def apply():
        settings.resolve_tuning_mode("mm", supports_cost_model=False)
        return False

    entry = entry_with(apply)
    monkeypatch.setenv("USE_FLAGTUNE", "invalid")
    for _ in range(2):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            entry._apply_flagtune(0)
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    entry._apply_flagtune(0)


def test_explicit_default_ignores_irrelevant_model_and_include_changes(monkeypatch):
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    calls = []
    entry = entry_with(lambda: calls.append(1) or False)
    entry._apply_flagtune(0)
    for name in ["FLAGTUNE_INCLUDE", "FLAGTUNE_MODEL_DIR", "FLAGTUNE_MODEL_VERSION"]:
        monkeypatch.setenv(name, "changed")
        entry._apply_flagtune(0)
    assert len(calls) == 1
    monkeypatch.setenv("USE_FLAGTUNE", "1")
    entry._apply_flagtune(0)
    assert len(calls) == 2


def test_model_source_is_not_a_mode_selection_dependency(monkeypatch):
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    calls = []
    entry = entry_with(lambda: calls.append(1) or False)
    entry._apply_flagtune(0)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", "changed")
    entry._apply_flagtune(0)
    assert len(calls) == 1
    monkeypatch.setenv("FLAGTUNE_INCLUDE", "mm")
    entry._apply_flagtune(0)
    assert len(calls) == 2


def test_auto_fuse_invalidates_all_entries_for_the_same_model(monkeypatch):
    key = ("flaggems/mm", "gemv_ppu")
    disabled = set()
    monkeypatch.setattr(
        lib.cost_model, "is_auto_disabled", lambda *identity: identity in disabled
    )
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")
    calls = []
    first = entry_with(lambda: calls.append("first") or True)
    second = entry_with(lambda: calls.append("second") or True)
    for entry in (first, second):
        entry._flagtune_model_ids = (key,)
        entry._apply_flagtune(0)
        entry._apply_flagtune(0)
    assert calls == ["first", "second"]
    disabled.add(("flaggems/mm", "unrelated"))
    first._apply_flagtune(0)
    assert len(calls) == 2
    disabled.add(key)
    for entry in (first, second):
        entry.kernel_cache[0]["stale"] = 1
        entry._cpu_cache["stale"] = 1
        entry._apply_flagtune(0)
        assert not any(entry.kernel_cache)
        assert not entry._cpu_cache
        entry._apply_flagtune(0)
    assert calls == ["first", "second", "first", "second"]
