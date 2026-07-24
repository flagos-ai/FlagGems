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
import multiprocessing
import os
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

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


# not_raises is copied from https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69
@contextmanager
def not_raises(ExpectedException):
    try:
        yield

    except ExpectedException as error:
        raise AssertionError(f"Raised exception {error} when it should not!")

    except Exception as error:
        raise AssertionError(f"An unexpected exception {error} raised.")


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


def test_flagtree_policy_is_bypassed_when_use_flagtune_is_enabled(monkeypatch):
    """Use exhaustive legacy tuning and avoid the proposer when USE_FLAGTUNE is set."""
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


def test_flagtree_policy_is_default_when_use_flagtune_is_disabled(monkeypatch):
    """Use the model-backed proposer only when its independent switch is on."""

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
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.setenv("TRITON_USE_FLAGTUNE", "1")
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", lambda: (True, None))
    monkeypatch.setattr(libentry_mod, "_flagtune_enabled", lambda: True)
    monkeypatch.setattr(
        libentry_mod,
        "_ensure_flagtune_proposer",
        lambda _identity: (fake_proposer, FakeVariantInfo()),
    )

    best_config, timings = LibTuner.get("flagtune").policy(
        FakeTuner(),
        lambda cfg: [cfg.kwargs["BLOCK"]],
        [triton.Config({"BLOCK": 8})],
        (),
        {},
    )

    assert proposer_called is True
    assert best_config.kwargs["BLOCK"] == 1
    assert list(timings.values()) == [1.0]


def test_flagtree_policy_is_bypassed_when_triton_flagtune_is_disabled(monkeypatch):
    """Avoid loading pair models unless the new FlagTree switch is enabled."""
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
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    monkeypatch.delenv("TRITON_USE_FLAGTUNE", raising=False)
    monkeypatch.setattr(flagtune_runtime_mod, "_include_ops", None)
    monkeypatch.setattr(libentry_mod, "_flagtune_available", lambda: (True, None))
    monkeypatch.setattr(libentry_mod, "_flagtune_enabled", lambda: False)
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
    with LibTuner.use_run_mode(
        tuner, LibTunerRunMode.EXHAUSTIVE_COLLECTION
    ):
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


def test_benchmark_key_preserves_raw_shape_and_scopes_timing_protocol():
    """Keep ConfigCache bucketing while separating exact benchmark labels."""

    class FakeTuner:
        """Provide the key/protocol state consumed by LibTuner helpers."""

        keys = ["M"]
        strategy = [libentry_mod.align32_strategy]
        _benchmark_protocol = ("triton_do_bench", 5, 20)

    tuner = FakeTuner()
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

    with LibTuner.use_benchmark_protocol(tuner, 25, 100):
        assert LibTuner.get_benchmark_key(tuner, {"M": 33}) == (
            33,
            "triton_do_bench",
            25,
            100,
        )
    assert tuner._benchmark_protocol == ("triton_do_bench", 5, 20)


def test_benchmark_config_reuses_kernel_context_and_bypasses_caches(monkeypatch):
    """Benchmark one fixed config with explicit durations and no policy/cache call."""
    config = triton.Config({"BLOCK": 16})
    observed = {}

    def fake_triton_do_bench(kernel_call, *, warmup, rep, quantiles):
        """Capture the authoritative timing options and execute the fake kernel."""
        observed["warmup"] = warmup
        observed["rep"] = rep
        observed["quantiles"] = quantiles
        observed["launch"] = kernel_call()
        return [1.0, 0.8, 1.2]

    monkeypatch.setattr(triton.testing, "do_bench", fake_triton_do_bench)

    class FakeTuner:
        """Provide the retained context and _bench protocol used by the API."""

        _last_benchmark_args = ("descriptor",)
        _last_benchmark_meta = {"M": 32}
        arg_names = ["descriptor_arg"]
        nargs = None
        seen_tuned_metas = {"stale": [9.0, 9.0, 9.0]}

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
        quantiles=(0.2, 0.5, 0.8),
    )

    assert result == [1.0, 0.8, 1.2]
    assert observed == {
        "args": ("descriptor",),
        "config": config,
        "meta": {"M": 32},
        "nargs": {"descriptor_arg": "descriptor"},
        "seen_tuned_metas": {},
        "warmup": 200,
        "rep": 500,
        "quantiles": (0.2, 0.5, 0.8),
        "launch": "kernel-launched",
    }
    assert tuner.do_bench is tuner.original_do_bench
    assert tuner.nargs is None


@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="The config covers NVIDIA Hopper mm kernels.",
)
def test_hopper_mm_config_compiles_without_runtime_registration():
    """Compile all training variants and verify canonical kernel pair bindings."""
    mm_ops = importlib.import_module("flag_gems.runtime.backend._nvidia.hopper.ops.mm")
    from flag_gems.utils.flagtune import operator_config as operator_config_mod

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
    def child_process():
        cache = libcache["test_vllm_operator"]
        cache[(128, 256, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 64}, num_warps=4
        )
        cache[(256, 512, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 128}, num_warps=8
        )
        while True:
            time.sleep(0.1)

    assert libcache.db_url.startswith("sqlite:///")
    cache_path = Path(libcache.db_url.removeprefix("sqlite:///"))
    # Start child process
    process = multiprocessing.Process(target=child_process)
    process.start()
    time.sleep(1)
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
