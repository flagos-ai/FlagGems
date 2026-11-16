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
import triton

lib = importlib.import_module("flag_gems.utils.libentry")
cm = lib.cost_model


@pytest.fixture
def models(monkeypatch):
    identity = pytest.importorskip("triton.flagtune.contract.identity")
    errors = pytest.importorskip("triton.flagtune.runtime.errors")
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setattr(
        importlib.import_module("flag_gems.runtime.flagtune"), "_include_ops", None
    )
    monkeypatch.setattr(lib, "_HAS_FLAGTREE_FLAGTUNE", True)
    for name in ("_COST_MODEL_IDENTITIES", "_COST_MODEL_RETRY_AFTER"):
        monkeypatch.setattr(cm, name, {})
    monkeypatch.setattr(cm, "_COST_MODEL_DISABLED_OPS", set())
    monkeypatch.setattr(cm, "available", lambda: (True, None))
    now = [0.0]
    monkeypatch.setattr(cm.time, "monotonic", lambda: now[0])
    platform = ["thead-zw810e"]
    monkeypatch.setattr(
        identity, "discover_gpu_metadata", lambda: {"platform_key": platform[0]}
    )
    configs = [triton.Config({"BLOCK": 1}), triton.Config({"BLOCK": 2})]
    monkeypatch.setattr(cm, "runtime_candidates", lambda *a: configs)
    attempts, broken = [], set()

    def ensure(model):
        key = (model.platform_key, model.op_id, model.variant, model.dtype_key)
        attempts.append(key)
        if key in broken:
            raise errors.ModelUnavailableError("missing model")
        return object(), object()

    monkeypatch.setattr(cm, "ensure_proposer", ensure)
    monkeypatch.setattr(cm, "run_proposer", lambda *a: (configs[1], {}))

    def tuner(device="cuda:0", dtype="bfloat16", output="bfloat16"):
        current = SimpleNamespace(
            _flagtune_op_name="mm",
            _flagtune_op_id="flaggems/mm",
            _flagtune_variant="gemv_ppu",
            arg_names=["A", "B", "C", "scratch"],
            nargs={
                name: SimpleNamespace(device=device, dtype=value)
                for name, value in zip(
                    ("A", "B", "C", "scratch"),
                    (dtype, dtype, output, "float32"),
                )
            },
            _flagtune_dtype_resolver=lambda args: tuple(
                args[name].dtype for name in ("A", "B", "C")
            ),
        )
        return current

    def run(current):
        return cm.run_policy(
            current,
            None,
            configs,
            (),
            {},
            supports_cost_model=True,
            infer_tensor_dtypes=lib._infer_tensor_dtypes,
            default_policy=lambda *a: (configs[0], {}),
            legacy_fallback=lambda *a: (configs[0], {}),
        )[0]

    return SimpleNamespace(
        tuner=tuner,
        run=run,
        now=now,
        platform=platform,
        broken=broken,
        attempts=attempts,
        configs=configs,
        error=errors.ModelUnavailableError,
    )


def test_failure_isolates_platform_dtype_and_recovery(models, caplog, monkeypatch):
    key = ("thead-zw810e", "flaggems/mm", "gemv_ppu", "bf16-bf16-bf16")
    other = ("nvidia-h20", *key[1:])
    models.broken.update((key, other))
    first = models.tuner()
    assert models.run(first) is models.configs[0]
    assert cm._COST_MODEL_DISABLED_OPS == {key}

    # A new ordinal of the same platform shares the canonical cooldown.
    same = models.tuner(device="cuda:1")
    assert models.run(same) is models.configs[0]
    assert models.attempts == [key]
    # Temporary/output tensors do not change the contract's ordered identity.
    same.nargs["scratch"].dtype = "float16"
    assert models.run(same) is models.configs[0]
    assert models.attempts == [key]

    fp16 = models.tuner(dtype="float16", output="float16")
    assert models.run(fp16) is models.configs[1]
    mixed = models.tuner(output="float32")
    assert models.run(mixed) is models.configs[1]
    assert models.attempts[-2:] == [
        (*key[:3], "f16-f16-f16"),
        (*key[:3], "bf16-bf16-f32"),
    ]
    assert cm._COST_MODEL_DISABLED_OPS == {key}

    models.platform[0] = other[0]
    second = models.tuner(device="cuda:2")
    assert models.run(second) is models.configs[0]
    assert models.attempts[-1] == other
    assert cm._COST_MODEL_DISABLED_OPS == {key, other}
    assert len([r for r in caplog.records if "model unavailable" in r.message]) == 2

    # REQUIRED bypasses only the cooldown check, preserving AUTO state on failure.
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")
    with pytest.raises(models.error):
        models.run(first)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL")
    assert cm._COST_MODEL_DISABLED_OPS == {key, other}

    models.broken.remove(key)
    models.now[0] = cm._AUTO_RETRY_SECONDS
    assert models.run(first) is models.configs[1]
    assert cm._COST_MODEL_DISABLED_OPS == {other}
    assert key not in cm._COST_MODEL_RETRY_AFTER
    assert cm._COST_MODEL_RETRY_AFTER[other] == cm._AUTO_RETRY_SECONDS


def test_identity_discovery_failure_recovers_without_stale_state(models, monkeypatch):
    identity = importlib.import_module("triton.flagtune.contract.identity")
    discover = identity.discover_gpu_metadata

    def unavailable():
        raise models.error("device metadata unavailable")

    monkeypatch.setattr(identity, "discover_gpu_metadata", unavailable)
    first = models.tuner()
    assert models.run(first) is models.configs[0]
    provisional = cm.failure_key(first, first.nargs)
    assert cm._COST_MODEL_DISABLED_OPS == {provisional}
    assert models.attempts == []
    monkeypatch.setattr(identity, "discover_gpu_metadata", discover)
    assert models.run(first) is models.configs[0]
    models.now[0] = cm._AUTO_RETRY_SECONDS
    assert models.run(first) is models.configs[1]
    assert not cm._COST_MODEL_DISABLED_OPS
    assert not cm._COST_MODEL_RETRY_AFTER
    assert cm.failure_key(first, first.nargs) != provisional


def test_direct_tuner_retry_bypasses_best_config_cache(models, monkeypatch):
    current = models.tuner()
    key = ("thead-zw810e", "flaggems/mm", "gemv_ppu", "bf16-bf16-bf16")
    models.broken.add(key)
    models.run(current)
    models.now[0] = cm._AUTO_RETRY_SECONDS

    class StaleBestCache:
        def __contains__(self, key):
            raise AssertionError("expired AUTO failure reused the best-config cache")

    current.cache = StaleBestCache()
    current.configs = models.configs
    current._flagtune_mode = lib.runtime.TuningMode.COST_MODEL
    current._flagtune_failure_key = key
    current.apply_flagtune = lambda arguments: False
    current.get_key = lambda arguments: (32,)
    current.get_benchmark_key = lambda arguments: (32,)
    current.benchmark_table_name = "identity_retry"
    current.pre_hook = lambda *a, **kw: None
    current.fn = SimpleNamespace(run=lambda *a, **kw: kw["BLOCK"])
    attempts = []

    def policy(*args):
        attempts.append(1)
        return models.configs[1], {}

    current.policy = policy
    monkeypatch.setattr(lib, "libcache", {("identity_retry", (32,)): {}})
    result = lib.LibTuner.run(current, **current.nargs)
    assert result == 2
    assert attempts == [1]


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("observe_cooldown", [False, True])
def test_dispatch_switches_only_for_current_model(
    models, monkeypatch, expanded, observe_cooldown
):
    if expanded:
        monkeypatch.setenv("FLAGTUNE_INCLUDE", "mm")
    current = models.tuner()
    modes = lib.runtime.TuningMode
    current._flagtune_configs_for_mode = lambda name, mode: ([mode], "default")

    def select(configs, strategy, *, mode):
        current._flagtune_mode = mode

    current._set_configs_and_strategy = select
    current.apply_flagtune = lambda arguments: lib.LibTuner.apply_flagtune(
        current, arguments
    )
    current.fn = None
    entry = object.__new__(lib.LibEntry)
    entry.fn = current
    entry.arg_names = current.arg_names
    entry._flagtune_tuners = (current,)
    entry._flagtune_state = None
    entry.lock = threading.Lock()
    entry.kernel_cache = ({},)
    entry._cpu_cache = {}
    entry._launch_plans = {}

    def dispatch(tuner):
        entry._apply_flagtune(0, (), tuner.nargs)

    # Start with healthy FP16 dispatch, then fail BF16 in another caller.
    healthy = models.tuner(dtype="float16", output="float16")
    models.run(healthy)
    dispatch(healthy)
    entry.kernel_cache[0]["healthy"] = object()
    key = ("thead-zw810e", "flaggems/mm", "gemv_ppu", "bf16-bf16-bf16")
    models.broken.add(key)
    models.run(current)
    dispatch(healthy)
    assert current._flagtune_mode is modes.COST_MODEL
    assert "healthy" in entry.kernel_cache[0]

    if not observe_cooldown:
        models.now[0] = cm._AUTO_RETRY_SECONDS
        dispatch(current)
        assert current._flagtune_mode is modes.COST_MODEL
        assert not entry.kernel_cache[0]
        return

    dispatch(current)
    assert current._flagtune_mode is (modes.EXPANDED if expanded else modes.DEFAULT)
    assert not entry.kernel_cache[0]
    dispatch(healthy)
    assert current._flagtune_mode is modes.COST_MODEL
    dispatch(current)
    entry.kernel_cache[0]["fallback"] = object()
    entry._launch_plans["fallback"] = object()
    models.now[0] = cm._AUTO_RETRY_SECONDS
    dispatch(current)
    assert current._flagtune_mode is modes.COST_MODEL
    assert not entry.kernel_cache[0]
    assert not entry._launch_plans
