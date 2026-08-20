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
import importlib.util
import multiprocessing
import os
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
from triton import language as tl

import flag_gems
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.code_cache import config_cache_dir
from flag_gems.utils.libentry import (
    LibTuner,
    LibTunerRunMode,
    libcache,
    major_version,
    minor_version,
)

libentry_mod = importlib.import_module("flag_gems.utils.libentry")
flagtune_runtime_mod = importlib.import_module("flag_gems.runtime.flagtune")
HAS_FLAGTREE_FLAGTUNE = importlib.util.find_spec("triton.flagtune") is not None
requires_flagtree_flagtune = pytest.mark.skipif(
    not HAS_FLAGTREE_FLAGTUNE,
    reason="test requires the optional FlagTree FlagTune package",
)


# not_raises is copied from https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69
@contextmanager
def not_raises(ExpectedException):
    try:
        yield

    except ExpectedException as error:
        raise AssertionError(f"Raised exception {error} when it should not!")

    except Exception as error:
        raise AssertionError(f"An unexpected exception {error} raised.")


def test_flagtune_environment_controls_operator_selection(monkeypatch):
    """Apply global and per-operator selection using each operator's capability."""
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setenv("USE_FLAGTUNE", "0")

    assert flagtune_runtime_mod.flagtune_enabled("mm") is False

    monkeypatch.setenv("USE_FLAGTUNE", "1")

    assert flagtune_runtime_mod.flagtune_enabled("mm") is True

    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    flag_gems.flagtune(include=["mm"])
    assert (
        flagtune_runtime_mod.resolve_tuning_mode("mm", supports_cost_model=False)
        is flagtune_runtime_mod.TuningMode.EXPANDED
    )
    assert (
        flagtune_runtime_mod.resolve_tuning_mode("mm", supports_cost_model=True)
        is flagtune_runtime_mod.TuningMode.COST_MODEL
    )

    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    assert (
        flagtune_runtime_mod.resolve_tuning_mode("mm", supports_cost_model=True)
        is flagtune_runtime_mod.TuningMode.EXPANDED
    )


@pytest.mark.parametrize(
    ("supports_cost_model", "use_flagtune", "use_cost_model", "expected"),
    [
        (False, None, None, "default"),
        (False, None, "0", "default"),
        (False, None, "1", "default"),
        (False, "0", None, "default"),
        (False, "0", "0", "default"),
        (False, "0", "1", "default"),
        (False, "1", None, "expanded"),
        (False, "1", "0", "expanded"),
        (False, "1", "1", "expanded"),
        (True, None, None, "cost_model"),
        (True, None, "0", "expanded"),
        (True, None, "1", "cost_model"),
        (True, "0", None, "default"),
        (True, "0", "0", "default"),
        (True, "0", "1", "default"),
        (True, "1", None, "cost_model"),
        (True, "1", "0", "expanded"),
        (True, "1", "1", "cost_model"),
    ],
)
def test_tuning_mode_environment_matrix(
    monkeypatch,
    supports_cost_model,
    use_flagtune,
    use_cost_model,
    expected,
):
    """Cover every switch combination for adapted and unadapted operators."""
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    if use_flagtune is None:
        monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    else:
        monkeypatch.setenv("USE_FLAGTUNE", use_flagtune)
    if use_cost_model is None:
        monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    else:
        monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", use_cost_model)

    mode = flagtune_runtime_mod.resolve_tuning_mode(
        "mm", supports_cost_model=supports_cost_model
    )

    assert mode.value == expected


def test_adapted_tuning_mode_prefers_cost_model_when_both_enabled(monkeypatch):
    """Treat both enabled switches as an explicit Cost Model request."""
    monkeypatch.setenv("USE_FLAGTUNE", "1")
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "1")

    assert (
        flagtune_runtime_mod.resolve_tuning_mode("mm", supports_cost_model=True)
        is flagtune_runtime_mod.TuningMode.COST_MODEL
    )


@pytest.mark.parametrize("name", ["USE_FLAGTUNE", "USE_FLAGTUNE_COST_MODEL"])
def test_tuning_mode_rejects_non_binary_environment_values(monkeypatch, name):
    """Fail fast instead of silently treating malformed switches as disabled."""
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.setenv(name, "true")

    with pytest.raises(ValueError, match=f"{name} must be 0 or 1"):
        flagtune_runtime_mod.resolve_tuning_mode("mm", supports_cost_model=True)


def test_adapted_libtuner_switches_default_expanded_and_cost_model(monkeypatch):
    """Apply all three paths through the same environment-mode resolver."""
    default_configs = [object()]
    expanded_configs = [object(), object()]

    class FakeTuner:
        __name__ = "mm"
        _flagtune_op_name = "mm"
        _flagtune_expand_op_name = "mm_general_tma"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"
        _flagtune_yaml_path = None
        _flagtune_pre_hook = None
        _flagtune_default_configs = default_configs
        _flagtune_default_strategy = "default_strategy"
        _flagtune_mode = flagtune_runtime_mod.TuningMode.DEFAULT
        _flagtune_warned = False

        def _set_configs_and_strategy(self, configs, strategy, *, mode=None):
            self.configs = configs
            self.strategy = strategy
            self._flagtune_mode = flagtune_runtime_mod.TuningMode(mode)

    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_HAS_FLAGTREE_FLAGTUNE", True)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.setattr(
        libentry_mod.runtime,
        "get_expand_config",
        lambda *_args, **_kwargs: {"strategy": "expanded_strategy"},
    )
    monkeypatch.setattr(
        libentry_mod.runtime,
        "ops_get_configs",
        lambda *_args, **_kwargs: expanded_configs,
    )
    tuner = FakeTuner()

    assert LibTuner.apply_flagtune(tuner) is True
    assert tuner._flagtune_mode is flagtune_runtime_mod.TuningMode.COST_MODEL
    assert tuner.configs is default_configs

    monkeypatch.setenv("USE_FLAGTUNE", "1")
    assert LibTuner.apply_flagtune(tuner) is False
    assert tuner._flagtune_mode is flagtune_runtime_mod.TuningMode.COST_MODEL
    assert tuner.configs is default_configs

    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    assert LibTuner.apply_flagtune(tuner) is True
    assert tuner._flagtune_mode is flagtune_runtime_mod.TuningMode.EXPANDED
    assert tuner.configs is expanded_configs

    monkeypatch.setenv("USE_FLAGTUNE", "0")
    assert LibTuner.apply_flagtune(tuner) is True
    assert tuner._flagtune_mode is flagtune_runtime_mod.TuningMode.DEFAULT
    assert tuner.configs is default_configs


@pytest.mark.parametrize(
    ("use_flagtune", "use_cost_model", "expected_mode", "use_expanded_configs"),
    [
        (None, None, "default", False),
        (None, "0", "default", False),
        (None, "1", "default", False),
        ("0", "0", "default", False),
        ("1", None, "expanded", True),
        ("1", "0", "expanded", True),
        ("1", "1", "expanded", True),
    ],
)
def test_official_triton_libtuner_uses_unadapted_routes(
    monkeypatch,
    use_flagtune,
    use_cost_model,
    expected_mode,
    use_expanded_configs,
):
    """Ignore Cost Model annotations when the FlagTree runtime is unavailable."""
    default_configs = [object()]
    expanded_configs = [object(), object()]

    class FakeTuner:
        __name__ = "mm"
        _flagtune_op_name = "mm"
        _flagtune_expand_op_name = "mm_general_tma"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"
        _flagtune_yaml_path = None
        _flagtune_pre_hook = None
        _flagtune_default_configs = default_configs
        _flagtune_default_strategy = "default_strategy"
        _flagtune_mode = flagtune_runtime_mod.TuningMode.DEFAULT
        _flagtune_warned = False
        configs = default_configs

        def _set_configs_and_strategy(self, configs, strategy, *, mode=None):
            self.configs = configs
            self.strategy = strategy
            self._flagtune_mode = flagtune_runtime_mod.TuningMode(mode)

    monkeypatch.setattr(libentry_mod, "_HAS_FLAGTREE_FLAGTUNE", False)
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    if use_flagtune is None:
        monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    else:
        monkeypatch.setenv("USE_FLAGTUNE", use_flagtune)
    if use_cost_model is None:
        monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    else:
        monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", use_cost_model)
    monkeypatch.setattr(
        libentry_mod.runtime,
        "get_expand_config",
        lambda *_args, **_kwargs: {"strategy": "expanded_strategy"},
    )
    monkeypatch.setattr(
        libentry_mod.runtime,
        "ops_get_configs",
        lambda *_args, **_kwargs: expanded_configs,
    )
    tuner = FakeTuner()

    changed = LibTuner.apply_flagtune(tuner)

    assert changed is use_expanded_configs
    assert tuner._flagtune_mode.value == expected_mode
    expected_configs = expanded_configs if use_expanded_configs else default_configs
    assert tuner.configs is expected_configs


def softmax_inner_decorator_cascade(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(inp, dtype=dtype)

    with torch_device_fn.device(out.device):
        grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
        softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            DUMMY=60,
        )
    return out


def softmax_inner_pass_kernel_arg_via_kw(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
    softmax_kernel_inner[grid](
        out,
        inp,
        M,
        N=N,
        DUMMY=60,
    )
    return out


def softmax_inner_kernel_arg_apply_default(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
    softmax_kernel_inner[grid](
        out,
        inp,
        M,
        N,
    )
    return out


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 32}),
        triton.Config({"TILE_N": 64}),
        triton.Config({"TILE_N": 128}),
        triton.Config({"TILE_N": 256}),
        triton.Config({"TILE_N": 512}),
        triton.Config({"TILE_N": 1024}),
    ],
    key=["N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: 1024 // args["TILE_N"],
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    DUMMY=42,
):
    _ = DUMMY
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 1)
        e = tl.exp(inp - m[:, None])
        z = tl.sum(e, 1)
        out = e / z[:, None]
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_M], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_M], value=0.0, dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, tl.max(inp, 1))
            alpha = m - m_new
            z = z * tl.exp(alpha) + tl.sum(tl.exp(inp - m_new[:, None]), axis=1)
            m = m_new
            n_offsets += TILE_N
            offset += TILE_N

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[:, None]) / z[:, None]
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, o, mask=mask)
            n_offsets += TILE_N
            offset += TILE_N


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="Issue #2825")
def test_decorator_cascade():
    # to test inner decorator can use arguments supplied by outer decorator
    # and grid function can use arguments supplied by all the decorator
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_decorator_cascade(x, dim=2)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="Issue #2825")
def test_pass_kernel_arg_via_kw():
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_pass_kernel_arg_via_kw(x, dim=2)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="Issue #2825")
def test_kernel_arg_apply_default():
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_kernel_arg_apply_default(x, dim=2)


class TaskThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        return self.func(*self.args)


def run_two_threads():
    devices = [0, 0]
    fs = []

    def task_fn(dev):
        x = torch.randn((128, 128, 128), device=dev)
        return softmax_inner_decorator_cascade(x, 1)

    for dev in devices:
        work = TaskThread(task_fn, (dev,))
        work.start()
        fs.append(work)

    for i in range(len(fs)):
        fs[i].join()


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="Issue #2825")
def test_threadsafety():
    for i in range(100):
        with not_raises(Exception):
            run_two_threads()


def test_hash_generation():
    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_a(x, y):
        return x + y + 1

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_b(x, y):
        return x + y

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_a_copy(x, y):
        return x + y + 1

    assert kernel_a.kernel_hash != kernel_a_copy.kernel_hash
    assert kernel_a.kernel_hash != kernel_b.kernel_hash


def test_hash_changes_when_dependency_modified():
    @triton.jit
    def sub_func(x, y):
        return x + y

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    original_hash = main_kernel.kernel_hash

    @triton.jit
    def sub_func(x, y):  # noqa:F811
        return x + y + 1

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    modified_hash = main_kernel.kernel_hash

    assert original_hash != modified_hash, (
        f"Expected different hashes when sub-function changes, "
        f"but got same hash: {original_hash}"
    )
    original_hash = modified_hash

    @triton.jit
    def sub_func(x, y, z=0):  # noqa:F811
        return x + y + z

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    modified_hash = main_kernel.kernel_hash
    assert original_hash != modified_hash, (
        f"Expected different hashes when sub-function changes, "
        f"but got same hash: {original_hash}"
    )


def test_flagtree_policy_is_bypassed_when_expanded_flagtune_is_enabled(monkeypatch):
    """Use exhaustive legacy tuning when expanded FlagTune is enabled."""
    configs = [
        triton.Config({"BLOCK": 4}),
        triton.Config({"BLOCK": 2}),
    ]
    called = False

    class FakeTuner:
        """Expose only the routing metadata required by the FlagTune policy."""

        _flagtune_expand_op_name = "mm_general_tma"
        _flagtune_op_name = "mm"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"
        _flagtune_pre_hook = None

    def fail_if_called(_op_id, _variant):
        """Record and reject any unexpected model-backed proposer lookup."""
        nonlocal called
        called = True
        raise AssertionError("FlagTree proposer should not be used")

    monkeypatch.setenv("USE_FLAGTUNE", "1")
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    monkeypatch.setattr(libentry_mod, "_ensure_flagtune_proposer", fail_if_called)

    best_config, timings = LibTuner.get("flagtune").policy(
        FakeTuner(),
        lambda cfg: [cfg.kwargs["BLOCK"]],
        configs,
        (),
        {},
    )

    assert best_config.kwargs["BLOCK"] == 2
    assert len(timings) == 2
    assert called is False


def test_official_triton_treats_model_annotation_as_unadapted(monkeypatch):
    """Keep default tuning usable when official Triton has no Cost Model."""

    class FakeTuner:
        _flagtune_expand_op_name = "mm_general_tma"
        _flagtune_op_name = "mm"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"

    observed = {}

    def resolve_mode(op_name, *, supports_cost_model):
        observed.update(
            op_name=op_name,
            supports_cost_model=supports_cost_model,
        )
        return flagtune_runtime_mod.TuningMode.DEFAULT

    def fail_if_called():
        raise AssertionError("official Triton must not load Cost Model modules")

    monkeypatch.setattr(libentry_mod, "_HAS_FLAGTREE_FLAGTUNE", False)
    monkeypatch.setattr(libentry_mod.runtime, "resolve_tuning_mode", resolve_mode)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", fail_if_called)

    best_config, timings = LibTuner.get("flagtune").policy(
        FakeTuner(),
        lambda cfg: [cfg.kwargs["BLOCK"]],
        [triton.Config({"BLOCK": 8})],
        (),
        {},
    )

    assert observed == {"op_name": "mm", "supports_cost_model": False}
    assert best_config.kwargs["BLOCK"] == 8
    assert list(timings.values()) == [[8]]


@requires_flagtree_flagtune
def test_flagtree_policy_uses_cost_model_by_default_for_adapted_operator(monkeypatch):
    """Use the model-backed proposer by default for an adapted operator."""

    class FakeVariantInfo:
        """Convert one synthetic feature/config schema for proposer testing."""

        param_names = ["BLOCK"]

        @staticmethod
        def normalize_inputs(_nargs):
            """Return the stable shape consumed by the fake proposer."""
            return {"M": 16, "N": 16, "K": 16}

        @staticmethod
        def to_config(config_dict):
            """Convert a proposed dictionary into a Triton Config."""
            return triton.Config({"BLOCK": int(config_dict["BLOCK"])})

    class FakeTuner:
        """Provide routing metadata and normalized arguments to the policy."""

        _flagtune_expand_op_name = "mm_general_tma"
        _flagtune_op_name = "mm"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"
        _flagtune_pre_hook = None
        _flagtune_dtype_resolver = staticmethod(
            lambda _arguments: (
                "bfloat16",
                "bfloat16",
                "bfloat16",
            )
        )
        arg_names = ["M", "N", "K"]
        nargs = {"M": 16, "N": 16, "K": 16}

    proposer_called = False

    def fake_proposer(_bench, _shape, _initial, _meta):
        """Record invocation and return one lower-latency synthetic config."""
        nonlocal proposer_called
        proposer_called = True
        return [{"BLOCK": 1}]

    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", lambda: (True, None))
    observed_identity = {}

    def fake_ensure(identity):
        observed_identity["value"] = identity
        return fake_proposer, FakeVariantInfo()

    monkeypatch.setattr(libentry_mod, "_ensure_flagtune_proposer", fake_ensure)

    best_config, timings = LibTuner.get("flagtune").policy(
        FakeTuner(),
        lambda cfg: [cfg.kwargs["BLOCK"]],
        [triton.Config({"BLOCK": 8})],
        (),
        {},
    )

    assert proposer_called is True
    assert observed_identity["value"].platform_key == "nvidia-h20"
    assert best_config.kwargs["BLOCK"] == 1
    assert list(timings.values()) == [1.0]


@pytest.mark.parametrize(
    "failure_stage, message",
    [
        ("identity", "identity contract failed"),
        ("init", "package contract failed"),
        ("inputs", "input contract failed"),
        ("proposer", "proposer contract failed"),
        ("benchmark", "candidate benchmark failed"),
    ],
)
@requires_flagtree_flagtune
def test_enabled_flagtree_policy_propagates_contract_failures(
    monkeypatch, failure_stage, message
):
    """Only explicit disablement may select the exhaustive default policy."""

    class FakeVariantInfo:
        param_names = ["BLOCK"]

        @staticmethod
        def normalize_inputs(_nargs):
            if failure_stage == "inputs":
                raise RuntimeError(message)
            return {"M": 16}

        @staticmethod
        def to_config(config_dict):
            return triton.Config({"BLOCK": int(config_dict["BLOCK"])})

    class FakeTuner:
        _flagtune_op_name = None
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "gemv"
        _flagtune_pre_hook = None
        arg_names = ["M"]
        nargs = {"M": 16}

        @staticmethod
        def _flagtune_dtype_resolver(_arguments):
            if failure_stage == "identity":
                raise RuntimeError(message)
            return ("bfloat16", "bfloat16", "bfloat16")

    def proposer(_bench, _shape, _initial, _meta):
        if failure_stage == "proposer":
            raise RuntimeError(message)
        return [{"BLOCK": 1}]

    def ensure(_identity):
        if failure_stage == "init":
            raise RuntimeError(message)
        return proposer, FakeVariantInfo()

    def bench(config):
        if failure_stage == "benchmark" and config.kwargs["BLOCK"] == 1:
            raise RuntimeError(message)
        return [float(config.kwargs["BLOCK"])]

    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", lambda: (True, None))
    monkeypatch.setattr(libentry_mod, "_ensure_flagtune_proposer", ensure)
    monkeypatch.setattr(
        "triton.flagtune.contract.identity.discover_gpu_metadata",
        lambda: {"platform_key": "nvidia-h20"},
    )

    with pytest.raises(RuntimeError, match=message):
        LibTuner.get("flagtune").policy(
            FakeTuner(),
            bench,
            [triton.Config({"BLOCK": 8})],
            (),
            {},
        )


@requires_flagtree_flagtune
def test_flagtree_proposer_cache_tracks_resolved_model_version(monkeypatch):
    """Refresh local proposer/variant pairs when the shared manager resolves a new version."""
    from triton.flagtune.runtime import proposer as proposer_mod

    class Identity:
        platform_key = "nvidia-h20"
        op_id = "flaggems/mm"
        variant = "general_tma"
        dtype_key = "bf16-bf16-bf16"

    identity = Identity()
    variants = [object(), object()]
    loaded = iter(
        (
            SimpleNamespace(model_version="1.0.0", variant=variants[0]),
            SimpleNamespace(model_version="1.1.0", variant=variants[1]),
        )
    )
    calls = []

    def fake_load(op_id, variant, **kwargs):
        calls.append(("load", op_id, variant, kwargs))
        return next(loaded)

    def fake_make(op_id, variant, **kwargs):
        calls.append(("make", op_id, variant, kwargs))
        return object()

    monkeypatch.setattr(libentry_mod, "_FLAGTUNE_PROPOSER_POOL", {})
    monkeypatch.setattr(libentry_mod, "_FLAGTUNE_VARIANT_INFO_POOL", {})
    monkeypatch.setattr(proposer_mod, "load_model_bundle", fake_load)
    monkeypatch.setattr(proposer_mod, "make_config_proposer", fake_make)

    first = libentry_mod._ensure_flagtune_proposer(identity)
    second = libentry_mod._ensure_flagtune_proposer(identity)

    assert first[0] is not second[0]
    assert first[1] is variants[0]
    assert second[1] is variants[1]
    assert [call[0] for call in calls] == ["load", "make", "load", "make"]
    assert all(
        call[3]
        == {
            "platform_key": "nvidia-h20",
            "dtype_key": "bf16-bf16-bf16",
        }
        for call in calls
    )


def test_flagtree_policy_is_bypassed_when_cost_model_is_disabled(monkeypatch):
    """Avoid loading pair models when the Cost Model switch selects Expanded."""
    called = False

    class FakeTuner:
        _flagtune_op_name = "mm"
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"

    def fail_if_called(_op_id, _variant):
        nonlocal called
        called = True
        raise AssertionError("disabled FlagTree proposer should not be loaded")

    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", lambda: (True, None))
    monkeypatch.setattr(libentry_mod, "_ensure_flagtune_proposer", fail_if_called)

    best_config, timings = LibTuner.get("flagtune").policy(
        FakeTuner(),
        lambda cfg: [cfg.kwargs["BLOCK"]],
        [triton.Config({"BLOCK": 8})],
        (),
        {},
    )

    assert best_config.kwargs["BLOCK"] == 8
    assert list(timings.values()) == [[8]]
    assert called is False


def test_benchmark_success_count_tracks_finite_uncached_benchmarks(monkeypatch):
    """Separate fresh finite measurements, latency hits, and best-cache hits.

    The fake tuner exercises all explicit LibTuner run modes without compiling
    a GPU kernel. It also verifies that both cache-isolated modes never read or
    write the shape-to-best-config cache.
    """
    configs = [
        triton.Config({"BLOCK": 8}),
        triton.Config({"BLOCK": 16}),
        triton.Config({"BLOCK": 32}),
    ]

    class FakeConfigCache:
        """Track best-config values and every cache protocol operation."""

        def __init__(self):
            """Initialize empty values and zero access counters."""
            self.values = {}
            self.contains_count = 0
            self.getitem_count = 0
            self.setitem_count = 0

        def reset_access_counts(self):
            """Reset protocol counters without changing stored best configs."""
            self.contains_count = 0
            self.getitem_count = 0
            self.setitem_count = 0

        def __contains__(self, key):
            """Record and perform a best-config membership query."""
            self.contains_count += 1
            return key in self.values

        def __getitem__(self, key):
            """Record and return one stored best config."""
            self.getitem_count += 1
            return self.values[key]

        def __setitem__(self, key, value):
            """Record and persist one best config in memory."""
            self.setitem_count += 1
            self.values[key] = value

    class FakeBenchmarkCache:
        """Store per-config latency tuples for one synthetic shape."""

        def __init__(self):
            """Initialize an empty config-to-latency mapping."""
            self.values = {}

        def get(self, config):
            """Return a cached latency tuple when present."""
            return self.values.get(config)

        def __setitem__(self, config, value):
            """Persist a newly measured latency tuple."""
            self.values[config] = value

    benchmark_cache = FakeBenchmarkCache()

    class FakeLibCache:
        """Expose the single BenchmarkCache expected by the fake tuner."""

        def __getitem__(self, key):
            """Validate the benchmark table/key pair and return its cache."""
            assert key == (
                "fake_benchmark",
                (32, "triton_do_bench", 5, 20),
            )
            return benchmark_cache

    class FakeFn:
        """Capture the final config chosen for the synthetic kernel launch."""

        @staticmethod
        def run(*args, **kwargs):
            """Return launch arguments instead of executing a GPU kernel."""
            return args, kwargs

    class FakeTuner:
        """Implement the minimal protocol consumed by ``LibTuner.run``."""

        arg_names = ["M"]
        benchmark_table_name = "fake_benchmark"
        fn = FakeFn()

        @staticmethod
        def get_key(_args):
            """Return one stable synthetic shape key."""
            return (32,)

        @staticmethod
        def get_benchmark_key(args):
            """Return the exact shape plus benchmark protocol identity."""
            return (args["M"], "triton_do_bench", 5, 20)

        def prune_configs(self, _kwargs):
            """Yield every active config without pruning."""
            return iter(self.configs)

        def policy(self, bench, candidates, _args, _kwargs):
            """Track normal-policy calls, benchmark candidates, and minimize p50."""
            self.policy_call_count = getattr(self, "policy_call_count", 0) + 1
            timings = {config: bench(config)[1] for config in candidates}
            return min(timings, key=timings.get), timings

        def _bench(self, *args, config, **kwargs):
            """Return finite samples except for the largest synthetic block."""
            block = float(config.kwargs["BLOCK"])
            if block == 32:
                return [float("inf")] * 3
            return [block - 1.0, block, block + 1.0]

        @staticmethod
        def pre_hook(_kwargs, reset_only=False):
            """Accept LibTuner's reset hook without external side effects."""
            return None

    monkeypatch.delenv("TRITON_PRINT_AUTOTUNING", raising=False)
    monkeypatch.setattr(libentry_mod, "libcache", FakeLibCache())
    tuner = FakeTuner()
    tuner.configs = configs
    config_cache = FakeConfigCache()
    tuner.cache = config_cache

    LibTuner.run(tuner, 32)
    assert tuner.benchmark_success_count == 2
    assert tuner.benchmark_cache_hit_count == 0
    assert tuner.policy_call_count == 1

    # Force best-config selection to run again while keeping per-config timings.
    tuner.cache.values.clear()
    LibTuner.run(tuner, 32)
    assert tuner.benchmark_success_count == 0
    assert tuner.benchmark_cache_hit_count == 3
    assert tuner.policy_call_count == 2

    # A best-config cache hit must also reset the count from the previous run.
    tuner.benchmark_success_count = 99
    tuner.benchmark_cache_hit_count = 99
    LibTuner.run(tuner, 32)
    assert tuner.benchmark_success_count == 0
    assert tuner.benchmark_cache_hit_count == 0

    # Single-config kernels bypass autotuning and therefore benchmark no configs.
    tuner.configs = [configs[0]]
    tuner.benchmark_success_count = 99
    tuner.benchmark_cache_hit_count = 99
    LibTuner.run(tuner, 32)
    assert tuner.benchmark_success_count == 0
    assert tuner.benchmark_cache_hit_count == 0

    # Exhaustive collection bypasses a populated best-config cache, reuses one
    # latency, and measures only the two missing configs (one finite, one inf).
    tuner.configs = configs
    config_cache.values = {(32,): configs[1]}
    config_cache.reset_access_counts()
    benchmark_cache.values = {configs[0]: (7.0, 8.0, 9.0)}
    with LibTuner.use_run_mode(tuner, LibTunerRunMode.EXHAUSTIVE_COLLECTION):
        LibTuner.run(tuner, 32)
    assert tuner.benchmark_success_count == 1
    assert tuner.benchmark_cache_hit_count == 1
    assert len(tuner.configs_timings) == 3
    assert tuner.best_config is configs[0]
    assert config_cache.values == {(32,): configs[1]}
    assert tuner.policy_call_count == 2
    assert (
        config_cache.contains_count,
        config_cache.getitem_count,
        config_cache.setitem_count,
    ) == (0, 0, 0)

    # Force-policy mode also bypasses ConfigCache, but preserves the tuner's
    # learned/custom policy instead of forcing the exhaustive default policy.
    config_cache.reset_access_counts()
    policy_calls = tuner.policy_call_count
    with LibTuner.use_run_mode(tuner, LibTunerRunMode.FORCE_POLICY):
        LibTuner.run(tuner, 32)
        assert tuner.policy_call_count == policy_calls + 1
        assert tuner.benchmark_success_count == 0
        assert tuner.benchmark_cache_hit_count == 3
        assert len(tuner.configs_timings) == 3
        assert (
            config_cache.contains_count,
            config_cache.getitem_count,
            config_cache.setitem_count,
        ) == (0, 0, 0)

        # The next cache-isolated pass reconstructs timings entirely from
        # latency entries and still never touches the best-config cache.
        config_cache.reset_access_counts()
        LibTuner.run(tuner, 32)
        assert tuner.benchmark_success_count == 0
        assert tuner.benchmark_cache_hit_count == 3
        assert len(tuner.configs_timings) == 3
        assert (
            config_cache.contains_count,
            config_cache.getitem_count,
            config_cache.setitem_count,
        ) == (0, 0, 0)

        tuner.configs = [configs[0]]
        LibTuner.run(tuner, 32)
        assert tuner.benchmark_success_count == 0
        assert tuner.benchmark_cache_hit_count == 1
        assert len(tuner.configs_timings) == 1

    assert tuner._last_benchmark_args == (32,)
    assert tuner._last_benchmark_meta == {}
    assert tuner._run_mode is LibTunerRunMode.NORMAL


def test_benchmark_key_preserves_raw_shape_and_scopes_timing_protocol(monkeypatch):
    """Keep ConfigCache bucketing while separating exact benchmark labels."""

    class FakeTuner:
        """Provide the key/protocol state consumed by LibTuner helpers."""

        keys = ["M"]
        strategy = [libentry_mod.align32_strategy]
        _benchmark_protocol = ("triton_do_bench", 5, 20)
        benchmark_protocol = None
        _benchmark_retries = 10
        do_bench = staticmethod(lambda _call, _quantiles: None)
        config_table_name = "fake_config"
        cache = object()

        @staticmethod
        def _make_config_table_name():
            return "fake_scoped_config"

    tuner = FakeTuner()
    scoped_protocol = libentry_mod.BenchmarkProtocol(
        requested_mode=libentry_mod.BenchmarkMode.REPLAY,
        resolved_mode=libentry_mod.BenchmarkMode.REPLAY,
        implementation="fake_replay_v1",
        cache_policy="warm_l2",
        warmup_ms=25,
        measurement_ms=100,
        n_retries=10,
        per_replay_ms=10.0,
    )
    monkeypatch.setattr(
        libentry_mod,
        "resolve_benchmarker",
        lambda *args, **kwargs: SimpleNamespace(
            protocol=scoped_protocol,
            benchmark=lambda _call, _quantiles: None,
        ),
    )

    class FakeScopedLibCache:
        def __getitem__(self, _key):
            return object()

    monkeypatch.setattr(libentry_mod, "libcache", FakeScopedLibCache())
    assert LibTuner.get_key(tuner, {"M": 33}) == (64,)
    assert LibTuner.get_key(tuner, {"M": 63}) == (64,)
    assert LibTuner.get_benchmark_key(tuner, {"M": 33}) == (
        33,
        "triton_do_bench",
        5,
        20,
    )
    assert LibTuner.get_benchmark_key(tuner, {"M": 63}) == (
        63,
        "triton_do_bench",
        5,
        20,
    )

    with LibTuner.use_benchmark_protocol(tuner, "replay", 25, 100, 10):
        assert LibTuner.get_benchmark_key(tuner, {"M": 33}) == (
            33,
            "fake_replay_v1",
            25,
            100,
            10,
            10.0,
        )
    assert tuner._benchmark_protocol == ("triton_do_bench", 5, 20)
    assert tuner.benchmark_protocol is None
    assert tuner._benchmark_retries == 10
    assert tuner.config_table_name == "fake_config"


@pytest.mark.parametrize(
    ("use_cuda_graph", "expected"),
    [
        (True, libentry_mod.BenchmarkMode.REPLAY),
        (False, libentry_mod.BenchmarkMode.EVENT),
    ],
)
def test_deprecated_cuda_graph_alias_maps_to_benchmark_mode(use_cuda_graph, expected):
    """Keep the legacy boolean API while making replay the implicit default."""
    with pytest.warns(DeprecationWarning, match="use_cuda_graph is deprecated"):
        assert libentry_mod._select_benchmark_mode(None, use_cuda_graph) is expected

    assert (
        libentry_mod._select_benchmark_mode(None, None)
        is libentry_mod.BenchmarkMode.REPLAY
    )
    with pytest.raises(ValueError, match="cannot be supplied together"):
        libentry_mod._select_benchmark_mode("event", False)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_benchmark_retries_rejects_invalid_legacy_protocol_values(value):
    """Avoid legacy replay division-by-zero before Triton resolves a benchmarker."""
    with pytest.raises(ValueError, match="positive integer"):
        libentry_mod._validate_benchmark_retries(value)


def test_replay_fallback_reuses_event_config_cache_namespace(monkeypatch):
    """Use the historical event cache when replay resolution falls back to event."""

    event_protocol = libentry_mod.BenchmarkProtocol(
        requested_mode=libentry_mod.BenchmarkMode.REPLAY,
        resolved_mode=libentry_mod.BenchmarkMode.EVENT,
        implementation="triton_do_bench",
        cache_policy="cold_l2",
        warmup_ms=25,
        measurement_ms=100,
        n_retries=1,
        per_replay_ms=None,
        fallback_reason="fake backend has no replay benchmarker",
    )
    monkeypatch.setattr(
        libentry_mod,
        "resolve_benchmarker",
        lambda *args, **kwargs: SimpleNamespace(
            protocol=event_protocol, benchmark=lambda _call, _quantiles: None
        ),
    )

    class FakeCache:
        def __getitem__(self, _key):
            return object()

    class FakeTuner:
        kernel_hash = "kernel"
        _benchmark_protocol = ("triton_do_bench", 25, 100)
        benchmark_protocol = None
        _benchmark_retries = 10
        do_bench = staticmethod(lambda _call, _quantiles: None)
        config_table_name = "mm_kernel"
        cache = object()
        _make_config_table_name = LibTuner._make_config_table_name

    monkeypatch.setattr(libentry_mod, "libcache", FakeCache())
    tuner = FakeTuner()
    tuner.__name__ = "mm"
    with LibTuner.use_benchmark_protocol(tuner, "replay", 25, 100, 10) as protocol:
        assert protocol is event_protocol
        assert tuner.config_table_name == "mm_kernel"
        assert tuner.benchmark_protocol is event_protocol
    assert tuner.config_table_name == "mm_kernel"
    assert tuner.benchmark_protocol is None


def test_config_cache_namespace_separates_replay_protocols():
    """Preserve the legacy event table while isolating replay timing choices."""

    class FakeTuner:
        pass

    tuner = FakeTuner()
    tuner.__name__ = "mm"
    tuner.kernel_hash = "kernel"
    tuner._benchmark_protocol = ("triton_do_bench", 25, 100)
    assert LibTuner._make_config_table_name(tuner) == "mm_kernel"

    tuner._benchmark_protocol = (
        "triton_cuda_graph_replay_v1",
        25,
        100,
        10,
        10.0,
    )
    replay_ten = LibTuner._make_config_table_name(tuner)
    tuner._benchmark_protocol = (
        "triton_cuda_graph_replay_v1",
        25,
        100,
        5,
        20.0,
    )
    replay_five = LibTuner._make_config_table_name(tuner)

    assert replay_ten.startswith("mm_kernel_benchmark_")
    assert replay_five.startswith("mm_kernel_benchmark_")
    assert replay_ten != replay_five


def test_adapted_config_cache_namespace_separates_tuning_modes():
    """Prevent Default and Cost Model paths from sharing best-config rows."""

    class FakeTuner:
        __name__ = "mm"
        kernel_hash = "kernel"
        _benchmark_protocol = ("triton_do_bench", 25, 100)
        _flagtune_op_id = "flaggems/mm"
        _flagtune_variant = "general_tma"

    tuner = FakeTuner()
    names = {}
    for mode in flagtune_runtime_mod.TuningMode:
        tuner._flagtune_mode = mode
        names[mode] = LibTuner._make_config_table_name(tuner)

    assert len(set(names.values())) == 3
    assert names[flagtune_runtime_mod.TuningMode.DEFAULT].endswith("_flagtune_default")
    assert names[flagtune_runtime_mod.TuningMode.EXPANDED].endswith(
        "_flagtune_expanded"
    )
    assert names[flagtune_runtime_mod.TuningMode.COST_MODEL].endswith(
        "_flagtune_cost_model"
    )


def test_benchmark_config_reuses_kernel_context_and_bypasses_caches(monkeypatch):
    """Benchmark one fixed config with explicit durations and no policy/cache call."""
    config = triton.Config({"BLOCK": 16})
    observed = {}

    protocol = libentry_mod.BenchmarkProtocol(
        requested_mode=libentry_mod.BenchmarkMode.EVENT,
        resolved_mode=libentry_mod.BenchmarkMode.EVENT,
        implementation="triton_do_bench",
        cache_policy="cold_l2",
        warmup_ms=200,
        measurement_ms=500,
        n_retries=1,
        per_replay_ms=None,
    )

    def fake_resolve(mode, *, warmup_ms, measurement_ms, n_retries):
        observed["mode"] = str(mode)
        observed["warmup"] = warmup_ms
        observed["rep"] = measurement_ms
        observed["retries"] = n_retries

        def benchmark(kernel_call, quantiles):
            observed["quantiles"] = quantiles
            observed["launch"] = kernel_call()
            return [1.0, 0.8, 1.2]

        return SimpleNamespace(protocol=protocol, benchmark=benchmark)

    monkeypatch.setattr(libentry_mod, "resolve_benchmarker", fake_resolve)

    class FakeScopedLibCache:
        def __getitem__(self, _key):
            return object()

    monkeypatch.setattr(libentry_mod, "libcache", FakeScopedLibCache())

    class FakeTuner:
        """Provide the retained context and _bench protocol used by the API."""

        _last_benchmark_args = ("descriptor",)
        _last_benchmark_meta = {"M": 32}
        arg_names = ["descriptor_arg"]
        nargs = None
        seen_tuned_metas = {"stale": [9.0, 9.0, 9.0]}
        benchmark_protocol = protocol
        _benchmark_protocol = protocol.cache_key()
        _benchmark_retries = 10
        config_table_name = "fake_config"
        cache = object()
        use_benchmark_protocol = LibTuner.use_benchmark_protocol

        @staticmethod
        def _make_config_table_name():
            return "fake_scoped_config"

        def __init__(self):
            """Install a sentinel benchmarker that must be restored."""
            self.do_bench = lambda _call, _quantiles: [99.0, 99.0, 99.0]
            self.original_do_bench = self.do_bench

        def _bench(self, *args, config, **meta):
            """Assert context/reset behavior, then use the installed benchmarker."""
            observed["args"] = args
            observed["config"] = config
            observed["meta"] = meta
            observed["nargs"] = dict(self.nargs)
            observed["seen_tuned_metas"] = dict(self.seen_tuned_metas)
            return self.do_bench(
                lambda: "kernel-launched",
                quantiles=(0.5, 0.2, 0.8),
            )

    tuner = FakeTuner()
    result = LibTuner.benchmark_config(
        tuner,
        config,
        warmup=200,
        rep=500,
        benchmark_mode="event",
        quantiles=(0.2, 0.5, 0.8),
    )

    assert result == [1.0, 0.8, 1.2]
    assert observed == {
        "args": ("descriptor",),
        "config": config,
        "meta": {"M": 32},
        "nargs": {"descriptor_arg": "descriptor"},
        "seen_tuned_metas": {},
        "mode": "BenchmarkMode.EVENT",
        "warmup": 200,
        "rep": 500,
        "retries": 10,
        "quantiles": (0.2, 0.5, 0.8),
        "launch": "kernel-launched",
    }
    assert tuner.do_bench is tuner.original_do_bench
    assert tuner.nargs is None


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not HAS_FLAGTREE_FLAGTUNE,
    reason="The config requires NVIDIA Hopper kernels and the optional FlagTree FlagTune package.",
)
def test_hopper_mm_config_compiles_without_runtime_registration():
    """Compile all training variants and verify canonical kernel pair bindings."""
    mm_ops = importlib.import_module("flag_gems.runtime.backend._nvidia.hopper.ops.mm")
    from flag_gems.flagtune.contracts import operator as operator_config_mod

    spec = operator_config_mod.load_operator_benchmark_spec(
        os.path.join(
            os.path.dirname(operator_config_mod.__file__),
            "configs",
            "mm_flagtune_configs.yaml",
        )
    )
    operator = spec.operator_info
    expected = {
        "general_tma": ({"M": 4096, "N": 4096, "K": 4096}, 3360, 54),
        "gemv": ({"M": 1024, "N": 1, "K": 4096}, 168, 46),
        "splitk": ({"M": 1024, "N": 1024, "K": 4096}, 672, 53),
    }
    assert set(operator.variants) == set(expected)
    assert spec.dispatch_order == ("gemv", "splitk", "general_tma")
    assert spec.shape.identity == ("B", "M", "N", "K")

    for name, (shape, config_count, feature_count) in expected.items():
        variant = operator.get_variant(name)
        assert variant.matches(shape)
        assert sum(1 for _ in variant.iter_configs()) == config_count
        assert len(variant.feature_names) == feature_count

    assert operator.op_id == "flaggems/mm"
    public_operator = operator_config_mod.resolve_public_operator(
        flag_gems, operator.op_id
    )
    bound_kernel_names = {
        "general_tma": "mm_kernel_general_host_tma",
        "gemv": "gemv_kernel",
        "splitk": "mm_kernel_splitk",
    }
    for variant_name, expected_kernel_name in bound_kernel_names.items():
        _, resolved_tuner = libentry_mod.find_flagtune_benchmark_target(
            public_operator, operator.op_id, variant_name
        )
        assert resolved_tuner.fn.__name__ == expected_kernel_name
    assert (
        mm_ops.mm_kernel_general_host_tma.fn._flagtune_op_id,
        mm_ops.mm_kernel_general_host_tma.fn._flagtune_variant,
    ) == ("flaggems/mm", "general_tma")
    assert (
        mm_ops.gemv_kernel.fn._flagtune_op_id,
        mm_ops.gemv_kernel.fn._flagtune_variant,
    ) == ("flaggems/mm", "gemv")
    assert (
        mm_ops.mm_kernel_splitk.fn._flagtune_op_id,
        mm_ops.mm_kernel_splitk.fn._flagtune_variant,
    ) == ("flaggems/mm", "splitk")


@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="Issue #2826: Cannot re-initialize MUSA in forked subprocess",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="Issue #2827: It's not stable in full test though it's passed by single test",
)
def test_libcache_vllm_signal_scenario():
    cache_ready = multiprocessing.Event()

    def child_process():
        cache = libcache["test_vllm_operator"]
        cache[(128, 256, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 64}, num_warps=4
        )
        cache[(256, 512, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 128}, num_warps=8
        )
        cache_ready.set()
        while True:
            time.sleep(0.1)

    assert libcache.db_url.startswith("sqlite:///")
    cache_path = Path(libcache.db_url.removeprefix("sqlite:///"))
    # Start child process
    process = multiprocessing.Process(target=child_process)
    process.start()
    try:
        assert cache_ready.wait(timeout=10), (
            "child process did not persist LibCache entries before timeout: "
            f"exitcode={process.exitcode}"
        )
        os.kill(process.pid, signal.SIGINT)
        process.join(timeout=5)

        cache_saved = False
        if cache_path.exists():
            cache = libcache["test_vllm_operator"]
            if (128, 256, "torch.float32") in cache and (
                256,
                512,
                "torch.float32",
            ) in cache:
                cache_saved = True

        if flag_gems.vendor_name != "cambricon":
            # TODO: (cambricon) Sqlite DO NOT approve that data can be written into
            # db file correctly, expecially in multiprocessing circumstances.
            assert (
                cache_saved
            ), f"Test documented current behavior: cache_saved={cache_saved}"
    finally:
        if process.is_alive():
            os.kill(process.pid, signal.SIGKILL)
            process.join()


@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads"
    or True,  # TODO: skip currently due to libcache table rename
    reason="Issue #2826: Cannot re-initialize MUSA in forked subprocess",
)
def test_libcache_concurrent_write_on_signal():
    """
    Tests that LibCache can handle concurrent writes from multiple processes
    when they are all terminated by a signal. This simulates a scenario where
    multiple vLLM workers are terminated at once.
    """
    NUM_PROCESSES = 10
    TABLE_NAME = "test_concurrent_signal_operator"

    def child_process_main(process_id):
        cache = libcache[TABLE_NAME]
        cache[(f"key_from_proc_{process_id}",)] = triton.Config(
            {}, num_warps=process_id + 1
        )
        while True:
            time.sleep(0.1)

    cache_file_name = (
        f"TunedConfig_{torch.cuda.get_device_name().replace(' ', '_')}_triton_{major_version}_{minor_version}.db"
        if device.vendor_name == "nvidia"
        else f"TunedConfig_{device.vendor_name}_triton_{major_version}_{minor_version}.db"
    )
    cache_path = config_cache_dir() / cache_file_name
    if cache_path.exists():
        try:
            with sqlite3.connect(cache_path, timeout=10.0) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        except sqlite3.Error:
            pass

    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=child_process_main, args=(i,)) for i in range(NUM_PROCESSES)
    ]
    for p in processes:
        p.start()

    try:
        time.sleep(2)
        for p in processes:
            os.kill(p.pid, signal.SIGTERM)

        for p in processes:
            p.join(timeout=10)

        total_entries = 0
        if cache_path.exists():
            with sqlite3.connect(cache_path) as conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                    total_entries = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    pass  # Table might not exist if saving failed

        assert total_entries == NUM_PROCESSES, (
            f"Expected {NUM_PROCESSES} entries from concurrent processes, "
            f"but found {total_entries}."
        )

    finally:
        for p in processes:
            if p.is_alive():
                p.kill()
        if cache_path.exists():
            try:
                cache_path.unlink()
            except sqlite3.Error:
                pass
