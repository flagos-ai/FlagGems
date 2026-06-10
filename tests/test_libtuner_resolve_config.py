"""Unit tests for LibTuner.resolve_config refactor (Phase 1 of libtriton_jit autotune V0).

All stubs — implement when fixture infra (mock LibTuner instance, fake JIT kernel) is in place.
See `.claude/progress.md` Phase 1 and `.claude/autotune_config_interface_design.md` §8.
"""

import pytest


def test_lookup_hit_returns_cached_config():
    """Hit path: when key is in libcache, resolve_config returns triton.Config without calling _bench."""
    pytest.skip("Phase 1 stub - implement after fixture infra")


def test_lookup_miss_triggers_bench_and_caches():
    """Miss path: invoke self.policy(bench, ...), write best_config back, second call hits cache."""
    pytest.skip("Phase 1 stub - needs real GPU + JIT kernel to run _bench")


def test_lookup_does_not_launch_kernel():
    """CRITICAL regression guard: resolve_config never calls self.fn.run().

    Implementation hint: monkey-patch self.fn.run to raise; confirm resolve_config
    completes without invoking it. This is the whole point of the refactor.
    """
    pytest.skip("Phase 1 stub")


def test_concurrent_same_key():
    """Concurrent resolve_config calls with same key don't corrupt cache.

    Note: real concurrency story handled by C++ stripe lock in Phase 4;
    this Python-side test verifies LibTuner doesn't break under GIL-bounded threads.
    """
    pytest.skip("Phase 1 stub")


def test_single_config_short_circuits():
    """len(self.configs) == 1: skip lookup entirely, return configs[0] without computing key."""
    pytest.skip("Phase 1 stub")


def test_pre_hook_reset_only_invoked_once():
    """pre_hook(full_nargs, reset_only=True) called exactly once on miss, zero on hit.

    Important for ops with non-empty pre_hook. ops/ scan (2026-05-10) confirmed
    only group_gemm has it, and that has no C++ wrapper, so V0 unaffected.
    """
    pytest.skip("Phase 1 stub")


def test_run_behavior_unchanged_after_refactor():
    """End-to-end regression guard: refactored run() produces identical output to pre-refactor.

    Already externally verified 2026-05-10 via `pytest tests/test_max.py -k test_max`:
    141 passed × 3 runs (cold-baseline / warm-after / cold-after-refactor). This
    unit test formalizes the same check.
    """
    pytest.skip("Phase 1 stub - external verification done, formalize when fixture ready")
