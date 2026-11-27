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

"""Dispatch tests for the Ascend ``index_put`` native fallback.

The Ascend backend overrides ``index_put`` / ``index_put_`` to redispatch to the
native ATen operator by default, working around the Ascend MLIR compiler crashes
tracked in issues #4280 and #4020. ``FLAGGEMS_ASCEND_INDEX_PUT_TRITON`` opts
back into the Triton implementation.

These tests exercise only the dispatch decision, not kernel correctness: the
Ascend module is loaded with ``torch``, the Triton implementation and the input
tensor stubbed out, so no torch / triton / NPU is required. Kernel accuracy on
real hardware is covered by the upstream CI suite (``tests/test_index_put*.py``).
"""

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASCEND_INDEX_PUT = ROOT / "src/flag_gems/runtime/backend/_ascend/ops/index_put.py"
ASCEND_OPS_INIT = ROOT / "src/flag_gems/runtime/backend/_ascend/ops/__init__.py"
ENV = "FLAGGEMS_ASCEND_INDEX_PUT_TRITON"


class FakeTensor:
    """Stand-in tensor exposing just the ``device`` attribute we branch on."""

    def __init__(self, device="npu:0"):
        self.device = device


class FakeRedispatch:
    """Records native ``aten`` redispatch calls keyed by op name."""

    def __init__(self, calls):
        self._calls = calls
        self.default = self

    def redispatch(self, keyset, *args):
        self._calls.append((self._op, keyset, args))
        return f"native-{self._op}"


@pytest.fixture
def ascend_index_put(monkeypatch):
    """Load ``_ascend/ops/index_put.py`` with torch and the Triton op stubbed.

    Returns ``(module, triton_calls, native_calls)``.
    """
    triton_calls = []
    native_calls = []

    def _triton_index_put(inp, indices, values, accumulate=False):
        triton_calls.append(("index_put", inp, indices, values, accumulate))
        return "triton-index_put"

    def _triton_index_put_(inp, indices, values, accumulate=False):
        triton_calls.append(("index_put_", inp, indices, values, accumulate))
        return "triton-index_put_"

    base = types.ModuleType("flag_gems.ops.index_put")
    base.index_put = _triton_index_put
    base.index_put_ = _triton_index_put_
    ops = types.ModuleType("flag_gems.ops")
    ops.index_put = base
    gems = types.ModuleType("flag_gems")
    gems.ops = ops
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    monkeypatch.setitem(sys.modules, "flag_gems.ops", ops)
    monkeypatch.setitem(sys.modules, "flag_gems.ops.index_put", base)

    # Minimal torch surface used at import time and in the fallback path.
    index_put_native = FakeRedispatch(native_calls)
    index_put_native._op = "index_put"
    index_put_inplace_native = FakeRedispatch(native_calls)
    index_put_inplace_native._op = "index_put_"
    aten = types.ModuleType("aten")
    aten.index_put = index_put_native
    aten.index_put_ = index_put_inplace_native
    torch_ops = types.ModuleType("torch.ops")
    torch_ops.aten = aten
    dispatch_key = types.SimpleNamespace(CompositeExplicitAutograd="CEA")
    torch_c = types.ModuleType("torch._C")
    torch_c.DispatchKey = dispatch_key
    torch_c.DispatchKeySet = lambda key: ("keyset", key)
    torch_module = types.ModuleType("torch")
    torch_module._C = torch_c
    torch_module.ops = torch_ops
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch._C", torch_c)
    monkeypatch.setitem(sys.modules, "torch.ops", torch_ops)

    spec = importlib.util.spec_from_file_location(
        "_ascend_index_put_under_test", ASCEND_INDEX_PUT
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, triton_calls, native_calls


def test_default_redispatches_index_put_to_native(ascend_index_put, monkeypatch):
    module, triton_calls, native_calls = ascend_index_put
    monkeypatch.delenv(ENV, raising=False)
    inp = FakeTensor()
    values = FakeTensor()

    result = module.index_put(inp, ["idx"], values, accumulate=True)

    assert result == "native-index_put"
    assert triton_calls == []
    assert native_calls == [
        ("index_put", ("keyset", "CEA"), (inp, ["idx"], values, True))
    ]


def test_default_redispatches_index_put_inplace_to_native(
    ascend_index_put, monkeypatch
):
    module, triton_calls, native_calls = ascend_index_put
    monkeypatch.delenv(ENV, raising=False)
    inp = FakeTensor()
    values = FakeTensor()

    result = module.index_put_(inp, ["idx"], values)

    assert result == "native-index_put_"
    assert triton_calls == []
    assert native_calls[0][0] == "index_put_"
    assert native_calls[0][2] == (inp, ["idx"], values, False)


@pytest.mark.parametrize("value", ["1", "true", "on", "TRUE", "On"])
def test_switch_forces_triton_index_put(ascend_index_put, monkeypatch, value):
    module, triton_calls, native_calls = ascend_index_put
    monkeypatch.setenv(ENV, value)
    inp = FakeTensor()

    result = module.index_put(inp, ["i"], "v", accumulate=False)

    assert result == "triton-index_put"
    assert native_calls == []
    assert triton_calls == [("index_put", inp, ["i"], "v", False)]


@pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
def test_switch_off_keeps_native(ascend_index_put, monkeypatch, value):
    module, triton_calls, native_calls = ascend_index_put
    monkeypatch.setenv(ENV, value)
    inp = FakeTensor()
    values = FakeTensor()

    module.index_put_(inp, ["i"], values)

    assert triton_calls == []
    assert native_calls[0][0] == "index_put_"


def test_switch_forces_triton_index_put_inplace(ascend_index_put, monkeypatch):
    module, triton_calls, native_calls = ascend_index_put
    monkeypatch.setenv(ENV, "1")
    inp = FakeTensor()

    result = module.index_put_(inp, ["i"], "v", accumulate=True)

    assert result == "triton-index_put_"
    assert native_calls == []
    assert triton_calls == [("index_put_", inp, ["i"], "v", True)]


def test_native_path_passes_index_list_and_moves_values(ascend_index_put, monkeypatch):
    """ATen takes ``Tensor?[]``: indices go in as a list, values on-device."""
    module, _, native_calls = ascend_index_put
    monkeypatch.delenv(ENV, raising=False)
    inp = FakeTensor(device="npu:0")
    off_device_values = FakeTensor(device="cpu")
    moved = FakeTensor(device="npu:0")

    def _to(device):
        assert device == "npu:0"
        return moved

    off_device_values.to = _to

    module.index_put(inp, ("a", "b"), off_device_values)

    _, _, args = native_calls[0]
    assert args[1] == ["a", "b"]
    assert args[2] is moved


def test_native_fallback_warns_only_once(ascend_index_put, monkeypatch, caplog):
    """The fallback warning fires once per process, not on every call."""
    module, _, _ = ascend_index_put
    monkeypatch.delenv(ENV, raising=False)
    inp = FakeTensor()
    values = FakeTensor()

    with caplog.at_level(logging.WARNING):
        module.index_put(inp, ["i"], values)
        module.index_put_(inp, ["i"], values)
        module.index_put(inp, ["i"], values)

    warnings = [
        record
        for record in caplog.records
        if "falling back to native ATen" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_ascend_ops_init_exports_index_put():
    """The override only takes effect if ``_ascend/ops/__init__`` re-exports it."""
    source = ASCEND_OPS_INIT.read_text()
    assert "from .index_put import index_put, index_put_" in source

    tree = ast.parse(source)
    exported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exported = {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    }
    assert {"index_put", "index_put_"} <= exported
