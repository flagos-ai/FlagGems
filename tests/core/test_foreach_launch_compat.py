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

"""Host-only launch routing tests using real Torch/Triton, without device work.

Run: pytest --confcutdir=tests/core tests/core/test_foreach_launch_compat.py
The isolated import avoids unrelated eager operator imports in flag_gems.
"""

import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import triton  # noqa: F401


def test_package_import_without_knobs():
    source = Path(__file__).resolve().parents[2] / "src"
    script = f"""
import builtins
import sys
import triton
sys.path.insert(0, {str(source)!r})
original_import = builtins.__import__
def without_knobs(name, globals=None, locals=None, fromlist=(), level=0):
    if (name == "triton" and "knobs" in (fromlist or ())
            and (globals or {{}}).get("__name__", "").startswith("flag_gems")):
        raise ImportError("cannot import name 'knobs' from 'triton'")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = without_knobs
import flag_gems
from flag_gems.utils import foreach
assert flag_gems.__file__.startswith({str(source)!r})
assert foreach.knobs is None
assert not foreach._can_replay_launch()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def executor(monkeypatch):
    original_import = builtins.__import__

    def without_knobs(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton" and "knobs" in (fromlist or ()):
            raise ImportError("cannot import name 'knobs' from 'triton'")
        return original_import(name, globals, locals, fromlist, level)

    runtime = ModuleType("flag_gems.runtime")
    runtime.device = SimpleNamespace(vendor_name="nvidia")
    path = Path(__file__).resolve().parents[2] / "src/flag_gems/utils/foreach.py"
    spec = importlib.util.spec_from_file_location("_foreach_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", without_knobs)
        scoped.setitem(sys.modules, "flag_gems.runtime", runtime)
        scoped.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    assert module.knobs is None
    module._triton_driver = SimpleNamespace(
        active=SimpleNamespace(get_current_stream=lambda _: 0)
    )
    reduction_path = path.parents[1] / "ops/_foreach_reduction.py"
    reduction_spec = importlib.util.spec_from_file_location(
        "_foreach_reduction_compat_test", reduction_path
    )
    reduction = importlib.util.module_from_spec(reduction_spec)
    zero = ModuleType("flag_gems.ops.zero")
    zero._launch_zero_kernel = None
    utils = ModuleType("flag_gems.utils")
    utils.tl_extra_shim = SimpleNamespace()
    runtime.torch_device_fn = SimpleNamespace()
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", without_knobs)
        scoped.setitem(sys.modules, "flag_gems.runtime", runtime)
        scoped.setitem(sys.modules, "flag_gems.utils", utils)
        scoped.setitem(sys.modules, "flag_gems.utils.foreach", module)
        scoped.setitem(sys.modules, "flag_gems.ops.zero", zero)
        scoped.setitem(sys.modules, reduction_spec.name, reduction)
        reduction_spec.loader.exec_module(reduction)
    reduction._triton_driver = module._triton_driver
    module.reduction = reduction
    return module


def make_launch(module, kind, events, hooks):
    class Launcher:
        launch = None  # Select the public launcher branch for unary replay.

        def __call__(self, *args):
            events.append("replay")

    class Kernel:
        def __getitem__(self, grid):
            def dispatch(*args):
                events.append("dispatch")
                # Standard dispatch owns hooks, including on legacy Triton.
                for hook in hooks:
                    hook()
                return SimpleNamespace(
                    run=Launcher(), function=object(), packed_metadata=object()
                )

            return dispatch

    kernel = Kernel()
    meta = torch.empty(4, dtype=torch.int64)
    scalar_buf = torch.empty(1)
    fn = object()
    dt = object()
    if kind == "reduction":
        return lambda: module.reduction._launch(
            kernel, (1, 1), (), (meta,), (meta,), torch.device("cuda:0")
        )
    if kind == "unary":
        return lambda: module._launch_executor_kernel(
            kernel, (1, 1), 1, meta, fn, dt, dt, 1024
        )
    if kind == "binary":
        return lambda: module._launch_binary_kernel(
            meta, (1, 1), 1, fn, dt, dt, dt, 1.0, False, 1024, kernel
        )
    if kind == "scalar":
        return lambda: module._launch_scalar_kernel(
            meta, scalar_buf, 1.0, (1, 1), 1, fn, dt, dt, dt, 1.0, True, 1024, kernel
        )
    return lambda: module._launch_ternary_kernel(
        (1, 1), 1, meta, scalar_buf, 1.0, fn, dt, dt, dt, True, 1024, False, kernel
    )


@pytest.mark.parametrize("kind", ["unary", "binary", "scalar", "ternary", "reduction"])
@pytest.mark.parametrize(
    "api",
    [
        "missing",
        "missing_idle",
        "missing_runtime",
        "missing_hooks",
        "legacy_none",
        "legacy_callable",
        "empty",
        "enter",
        "exit",
    ],
)
def test_import_and_launch_routing(executor, kind, api):
    events = []
    hooks = []
    callback = lambda: events.append("hook")
    if api == "missing_runtime":
        executor.knobs = SimpleNamespace()
    elif api not in ("missing", "missing_idle"):
        runtime = SimpleNamespace()
        executor.knobs = SimpleNamespace(runtime=runtime)
        if api == "legacy_none":
            runtime.launch_enter_hook = runtime.launch_exit_hook = None
        elif api == "legacy_callable":
            runtime.launch_enter_hook = callback
            runtime.launch_exit_hook = None
        elif api != "missing_hooks":
            runtime.launch_enter_hook = SimpleNamespace(calls=[])
            runtime.launch_exit_hook = SimpleNamespace(calls=[])
            if api in ("enter", "exit"):
                getattr(runtime, f"launch_{api}_hook").calls.append(callback)
    # Even with no knobs, a legacy standard launcher may have active hooks.
    if api in ("missing", "legacy_callable", "enter", "exit"):
        hooks.append(callback)
    launch = make_launch(executor, kind, events, hooks)
    for _ in range(3):
        launch()
    if api == "empty":
        assert events == ["dispatch", "replay", "replay"]
    else:
        assert events == (["dispatch", "hook"] if hooks else ["dispatch"]) * 3


@pytest.mark.parametrize("kind", ["unary", "binary", "scalar", "ternary", "reduction"])
@pytest.mark.parametrize("hook_name", ["launch_enter_hook", "launch_exit_hook"])
def test_warm_cache_rechecks_hooks(executor, kind, hook_name):
    runtime = SimpleNamespace(
        launch_enter_hook=SimpleNamespace(calls=[]),
        launch_exit_hook=SimpleNamespace(calls=[]),
    )
    executor.knobs = SimpleNamespace(runtime=runtime)
    events = []
    hooks = getattr(runtime, hook_name).calls
    launch = make_launch(executor, kind, events, hooks)
    for _ in range(3):
        launch()  # Dispatch, dict cache, then last-call cache where available.
    hooks.append(lambda: events.append("hook"))
    launch()
    hooks.clear()
    launch()
    assert events == ["dispatch", "replay", "replay", "dispatch", "hook", "replay"]
    # Unknown APIs must also bypass already-warm dict and last-call caches.
    executor.knobs = None
    launch()
    assert events[-1] == "dispatch"
