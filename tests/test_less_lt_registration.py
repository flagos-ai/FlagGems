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

from pathlib import Path

import pytest
import torch
import yaml

import flag_gems
import flag_gems.ops as ops

from . import accuracy_utils as utils


@pytest.mark.parametrize("overload", ["Tensor", "Scalar"])
def test_less_lt_metadata(overload):
    path = Path(__file__).resolve().parents[1] / "conf/operators.yaml"
    metadata = yaml.safe_load(path.read_text())["ops"]
    assert not any(item["id"] == "lt" for item in metadata)
    assert not hasattr(flag_gems, "lt")
    # Python exposes the implementation submodule, not a public callable.
    assert not callable(getattr(ops, "lt", None))
    assert "lt" not in ops.__all__
    for name in ("less", "lt"):
        op_id, key = f"{name}_{overload.lower()}", f"{name}.{overload}"
        assert sum(item["id"] == op_id for item in metadata) == 1
        owners = [item for item in metadata if key in (item.get("for") or [])]
        assert len(owners) == 1
        assert owners[0]["id"] == op_id
        assert owners[0]["for"] == [key]
        assert owners[0]["stages"] == [{"stable": "2.0"}]
        assert owners[0]["labels"] == ["aten", "pointwise"]
        assert owners[0]["kind"] == ["Math"]
        entries = [entry for k, entry, *rest in flag_gems._FULL_CONFIG if k == key]
        assert entries == [getattr(flag_gems, op_id)]
        assert entries[0].__name__ == op_id
        assert getattr(ops, op_id).__name__ == op_id
        assert op_id in ops.__all__
    assert getattr(flag_gems, f"less_{overload.lower()}") is not getattr(
        flag_gems, f"lt_{overload.lower()}"
    )


@pytest.mark.parametrize("name", ["less", "lt"])
@pytest.mark.parametrize("overload", ["Tensor", "Scalar"])
def test_less_lt_registration_and_direct_call(monkeypatch, name, overload):
    key, op_id = f"{name}.{overload}", f"{name}_{overload.lower()}"
    op = getattr(getattr(torch.ops.aten, name), overload)
    entry = getattr(flag_gems, op_id)
    a = torch.tensor([-float("inf"), -1, 0, 1, float("inf"), float("nan")])
    b = torch.tensor([0, -1, 1, 0, float("inf"), 0]) if overload == "Tensor" else 0.5
    expected = op(a, b)
    a = a.to(flag_gems.device)
    if overload == "Tensor":
        b = b.to(flag_gems.device)
    registered = []
    original_impl = torch.library.Library.impl

    def intercept(library, overload, fn, *args, **kwargs):
        if overload == key:
            assert fn is entry
            registered.append(overload)
        return original_impl(library, overload, fn, *args, **kwargs)

    monkeypatch.setattr(torch.library.Library, "impl", intercept)
    previous = flag_gems.current_work_registrar
    library = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(
            lib=library, include=[op_id], registrar=flag_gems.GeneralOpRegistrar
        )
        assert flag_gems.all_registered_keys() == [key]
        assert flag_gems.all_registered_ops() == [op_id]
    finally:
        try:
            library._destroy()
        finally:
            flag_gems.current_work_registrar = previous
    assert registered == [key]
    actual = entry(a, b)
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.less__scalar
@pytest.mark.less__tensor
@pytest.mark.lt__scalar
@pytest.mark.lt__tensor
@pytest.mark.parametrize("name", ["less", "lt"])
@pytest.mark.parametrize("overload", ["Tensor", "Scalar"])
def test_less_lt_inplace_metadata(name, overload):
    path = Path(__file__).resolve().parents[1] / "conf/operators.yaml"
    metadata = yaml.safe_load(path.read_text())["ops"]
    op_id, key = f"{name}__{overload.lower()}", f"{name}_.{overload}"
    old_id = f"{name}_" if overload == "Tensor" else f"{name}_scalar_"
    assert not any(item["id"] == old_id for item in metadata)
    assert sum(item["id"] == op_id for item in metadata) == 1
    owners = [item for item in metadata if key in (item.get("for") or [])]
    assert len(owners) == 1
    assert owners[0]["id"] == op_id
    assert owners[0]["for"] == [key]
    assert owners[0]["stages"] == [{"alpha": "5.4"}]
    entries = [entry for k, entry, *rest in flag_gems._FULL_CONFIG if k == key]
    assert entries == [getattr(flag_gems, op_id)]
    assert entries[0].__name__ == op_id
    assert getattr(ops, op_id).__name__ == op_id
    assert op_id in ops.__all__
    aliases_path = path.with_name("ci_test_aliases.yaml")
    aliases = yaml.safe_load(aliases_path.read_text())
    assert aliases[op_id] == f"test_{name}_"
    assert old_id not in aliases
    for module in (flag_gems, ops):
        # Retained implementation modules may still be package attributes.
        assert not callable(getattr(module, old_id, None))
        assert old_id not in module.__all__
    assert not hasattr(flag_gems, old_id)


@pytest.mark.less__scalar
@pytest.mark.less__tensor
@pytest.mark.lt__scalar
@pytest.mark.lt__tensor
@pytest.mark.parametrize("name", ["less", "lt"])
@pytest.mark.parametrize("overload", ["Tensor", "Scalar"])
def test_less_lt_inplace_registration_and_direct_call(monkeypatch, name, overload):
    key, op_id = f"{name}_.{overload}", f"{name}__{overload.lower()}"
    op = getattr(getattr(torch.ops.aten, f"{name}_"), overload)
    entry = getattr(flag_gems, op_id)
    a = torch.tensor([-float("inf"), -1, 0, 1, float("inf"), float("nan")])
    b = torch.tensor([0, -1, 1, 0, float("inf"), 0]) if overload == "Tensor" else 0.5
    expected = op(a.clone(), b)
    a = a.to(flag_gems.device)
    if overload == "Tensor":
        b = b.to(flag_gems.device)
    registered = []
    original_impl = torch.library.Library.impl

    def intercept(library, overload, fn, *args, **kwargs):
        if overload == key:
            assert fn is entry
            registered.append(overload)
        return original_impl(library, overload, fn, *args, **kwargs)

    monkeypatch.setattr(torch.library.Library, "impl", intercept)
    previous = flag_gems.current_work_registrar
    library = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(
            lib=library, include=[op_id], registrar=flag_gems.GeneralOpRegistrar
        )
        assert flag_gems.all_registered_keys() == [key]
        assert flag_gems.all_registered_ops() == [op_id]
        dispatched_input = a.clone()
        dispatched = op(dispatched_input, b)
        assert dispatched is dispatched_input
        utils.gems_assert_equal(dispatched.cpu(), expected)
    finally:
        try:
            library._destroy()
        finally:
            flag_gems.current_work_registrar = previous
    assert registered == [key]
    actual = entry(a, b)
    assert actual is a
    utils.gems_assert_equal(actual.cpu(), expected)
    utils.gems_assert_equal(a.cpu(), expected)
