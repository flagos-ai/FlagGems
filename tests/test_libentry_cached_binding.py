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

import gc
import importlib
import inspect
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

lib = importlib.import_module("flag_gems.utils.libentry")


class CaptureKey(Exception):
    pass


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((grid, args))

        return launch


@pytest.fixture(params=[0, "cpu"], ids=["device-cache", "cpu-cache"])
def entry(monkeypatch, request):
    monkeypatch.setattr(lib.torch_device_fn, "current_device", lambda: request.param)
    monkeypatch.setattr(lib, "major_version", 3)
    monkeypatch.setattr(lib, "minor_version", 6)
    e = object.__new__(lib.LibEntry)
    e._cache_binding = True
    e.lock = threading.Lock()
    e.arg_names = ["A", "N", "alpha", "FLAG", "BLOCK", "HEUR"]
    kinds = (1, 1, 2, 0, 0, 0)
    defaults = (
        inspect._empty,
        inspect._empty,
        1.0,
        False,
        inspect._empty,
        inspect._empty,
    )
    e.signature = inspect.Signature(
        [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in e.arg_names
        ]
    )
    e.jit_function = SimpleNamespace(
        params=[
            SimpleNamespace(
                name=n, is_constexpr=k == 0, do_not_specialize=k == 2, default=d
            )
            for n, k, d in zip(e.arg_names, kinds, defaults)
        ]
    )
    e.specialize_indices = [0, 1]
    e.do_not_specialize_indices = [2]
    e.divisibility = 16
    e._has_flagtune_tuner = True
    e._apply_flagtune = lambda device: None
    e.kernel_cache = ({},)
    e._cpu_cache = {}
    e._launch_plans = {}
    e._binding_tensor_type = torch.Tensor
    e._binding_scalar_types = (int, bool, float, str, type(None))
    e._binding_param_names = tuple(e.arg_names)
    e._binding_param_kinds = kinds
    e._binding_param_defaults = defaults
    e._binding_dtype_names = {}
    return e


def run_generic(entry, *args, **kwargs):
    enabled = entry._cache_binding
    entry._cache_binding = False
    try:
        return entry.run(*args, **kwargs)
    finally:
        entry._cache_binding = enabled


def seed_from_generic(e, args, kwargs, hooks=(), block=8):
    original_key = e.key

    def capture(spec, dns, const):
        raise CaptureKey(original_key(spec, dns, const))

    e.key = capture
    try:
        with pytest.raises(CaptureKey) as caught:
            run_generic(e, *args, **kwargs)
    finally:
        e.key = original_key
    key = caught.value.args[0]
    kernel = Kernel()
    cache = (
        e._cpu_cache
        if lib.torch_device_fn.current_device() == "cpu"
        else e.kernel_cache[lib.torch_device_fn.current_device()]
    )
    cache[key] = (
        kernel,
        {"BLOCK": block, "HEUR": 3},
        {"BLOCK": block},
        {"HEUR": 3},
        hooks,
    )
    return key, kernel


def assert_same_call(left, right):
    assert left[0] == right[0]
    assert len(left[1]) == len(right[1])
    for a, b in zip(left[1], right[1]):
        if isinstance(a, torch.Tensor):
            assert a is b
        else:
            assert type(a) is type(b) and a == b


@pytest.mark.parametrize("positional", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generic_and_cached_binding_agree(entry, positional, dtype):
    values = (torch.arange(8, dtype=dtype), 8, 0.5, True)
    args = values[:positional]
    kwargs = dict(zip(entry.arg_names[positional:4], values[positional:]))
    kwargs["grid"] = lambda m: (m["N"] // m["BLOCK"] + 1,)
    _, kernel = seed_from_generic(entry, args, kwargs)
    run_generic(entry, *args, **kwargs)
    assert entry._run_cached(args, kwargs) is not None
    assert_same_call(kernel.calls[0], kernel.calls[1])


def test_current_tensors_and_unspecialized_values_are_not_frozen(entry):
    a = torch.ones(8)
    b = torch.zeros(8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, (a, 8, 0.25), kwargs)
    entry._run_cached((a, 8, 0.25), kwargs)
    entry._run_cached((b, 8, 0.75), kwargs)
    assert kernel.calls[-1][1][0] is b
    assert kernel.calls[-1][1][2] == 0.75


def test_defaults_and_explicit_keyword_values_share_correct_plan(entry):
    a = torch.ones(8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, (a, 8), kwargs)
    entry._run_cached((a, 8), kwargs)
    entry._run_cached((a, 8), dict(kwargs, alpha=0.3, FLAG=False))
    assert kernel.calls[-1][1][2:] == (0.3, False, 8, 3)


@pytest.mark.parametrize("alpha", [-(2**31) - 1, 2**31, 2**63, 2**64 - 1, 2**64])
def test_unspecialized_integer_width_matches_generic(entry, alpha):
    args = (torch.ones(8), 8, alpha)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    assert entry._run_cached(args, kwargs) is not None
    assert kernel.calls[-1][1][2] == alpha


def test_alignment_and_dtype_changes_do_not_reuse_wrong_entry(entry):
    a = torch.ones(8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, (a, 8), kwargs)
    assert entry._run_cached((torch.ones(9)[1:], 8), kwargs) is None
    assert entry._run_cached((a.to(torch.bfloat16), 8), kwargs) is None


def test_custom_constexpr_uses_generic_fallback(entry):
    marker = object()
    args = (torch.ones(8), 8, 1.0, marker)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    assert entry._run_cached(args, kwargs) is None
    entry.run(*args, **kwargs)
    assert kernel.calls[-1][1][3] is marker


def test_hooks_receive_current_arguments(entry):
    seen = []
    a = torch.ones(8)
    b = torch.zeros(8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, (a, 8), kwargs, hooks=((seen.append, {"hook_value": 4}),))
    entry._run_cached((b, 8), kwargs)
    assert seen[-1]["A"] is b and seen[-1]["hook_value"] == 4


def test_replaced_kernel_cache_entry_refreshes_launch_constants(entry):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, args, kwargs, block=8)
    entry._run_cached(args, kwargs)
    _, kernel = seed_from_generic(entry, args, kwargs, block=16)
    entry._run_cached(args, kwargs)
    assert kernel.calls[-1][1][-2] == 16


def test_launch_plan_does_not_retain_input_tensors(entry):
    def launch():
        a = torch.ones(8)
        ref = weakref.ref(a)
        args = (a, 8)
        kwargs = {"grid": (1,)}
        _, kernel = seed_from_generic(entry, args, kwargs)
        entry._run_cached(args, kwargs)
        kernel.calls.clear()
        return ref

    ref = launch()
    gc.collect()
    assert ref() is None


def test_offline_benchmark_clear_also_releases_launch_plans(entry):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, args, kwargs)
    entry._run_cached(args, kwargs)
    assert entry._launch_plans
    lib.clear_libentry_dispatch_cache(entry)
    assert not any(entry.kernel_cache)
    assert not entry._launch_plans
    assert entry._run_cached(args, kwargs) is None


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "cpu", "hygon", "mthreads"])
@pytest.mark.parametrize("opt_in", [None, False, True])
def test_cached_binding_requires_explicit_opt_in(monkeypatch, vendor, opt_in):
    monkeypatch.setattr(lib, "device", SimpleNamespace(vendor_name=vendor))
    monkeypatch.setattr(lib, "major_version", 3)
    monkeypatch.setattr(lib, "minor_version", 6)

    @lib.triton.jit
    def kernel(x):
        pass

    decorate = lib.libentry() if opt_in is None else lib.libentry(cache_binding=opt_in)
    result = decorate(kernel)
    assert type(result) is lib.LibEntry
    assert result._cache_binding is (opt_in is True)
    assert hasattr(result, "_launch_plans") is (opt_in is True)


@pytest.mark.parametrize("major,minor", [(2, 3), (3, 2), (3, 7), (4, 0)])
def test_unsupported_triton_uses_generic_binding(monkeypatch, major, minor):
    monkeypatch.setattr(lib, "major_version", major)
    monkeypatch.setattr(lib, "minor_version", minor)

    @lib.triton.jit
    def kernel(x):
        pass

    result = lib.libentry(cache_binding=True)(kernel)
    assert not result._cache_binding
    assert not hasattr(result, "_launch_plans")


def test_disabled_binding_does_not_enter_fast_path(entry, monkeypatch):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    entry._cache_binding = False

    def unexpected(*args):
        raise AssertionError("disabled binding entered the fast path")

    monkeypatch.setattr(entry, "_run_cached", unexpected)
    entry.run(*args, **kwargs)
    assert kernel.calls[-1][1][0] is args[0]
    assert not entry._launch_plans


def test_cold_launch_uses_generic_path(entry):
    def cold_run(*args, **kwargs):
        raise RuntimeError("generic cold launch reached")

    entry.fn = SimpleNamespace(run=cold_run)
    with pytest.raises(RuntimeError, match="generic cold launch reached"):
        entry.run(torch.ones(8), 8, grid=(1,))
    assert not entry._launch_plans


def test_cached_binding_preserves_custom_specialization_key(entry, monkeypatch):
    original_key = entry.key
    monkeypatch.setattr(
        entry, "key", lambda *args: original_key(*args) + ("backend-specialization",)
    )
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    assert entry._run_cached(args, kwargs) is not None
    assert kernel.calls[-1][1][0] is args[0]


def test_binding_plans_are_scoped_to_device(entry, monkeypatch):
    entry.kernel_cache = ({}, {})
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    kernels = []
    for index in (0, 1):
        monkeypatch.setattr(lib.torch_device_fn, "current_device", lambda: index)
        _, kernel = seed_from_generic(entry, args, kwargs, block=8 * (index + 1))
        kernels.append(kernel)
        entry.run(*args, **kwargs)
    monkeypatch.setattr(lib.torch_device_fn, "current_device", lambda: 0)
    entry.run(*args, **kwargs)
    assert [len(kernel.calls) for kernel in kernels] == [2, 1]
    assert kernels[0].calls[-1][1][-2] == 8
    assert kernels[1].calls[-1][1][-2] == 16
    assert len(entry._launch_plans) == 2


def test_mode_change_clears_cached_binding_plans(entry):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, args, kwargs)
    entry.run(*args, **kwargs)
    assert entry._launch_plans
    entry.fn = SimpleNamespace(apply_flagtune=lambda: True, fn=None)
    entry._flagtune_model_ids = ()
    entry._flagtune_state = None
    lib.LibEntry._apply_flagtune(entry)
    assert not entry._launch_plans
    assert not any(entry.kernel_cache)
    assert not entry._cpu_cache
