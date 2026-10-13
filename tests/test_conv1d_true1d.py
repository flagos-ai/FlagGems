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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

MTHREADS = flag_gems.vendor_name == "mthreads"


@pytest.fixture(autouse=True)
def _exact_fp32_reference():
    """Compare against the exact mudnn reference inside this module only.

    The fp32 mudnn reference runs TF32 by default, which disagrees with an
    exact-fp32 kernel at ~1e-3 relative (the same pre-existing mismatch
    behind the official test_conv1d fp32 failures).  TF32 is disabled and
    restored around each test so other modules keep their default state.
    """
    if not MTHREADS:
        yield
        return
    old = torch.backends.mudnn.allow_tf32
    torch.backends.mudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.mudnn.allow_tf32 = old


# The four structural families of the official conv1d workload matrix:
# K3/K5 dense, K7 grouped, K11 long-sequence.
TRUE1D_FAMILIES = [
    (32, 64, 512, 64, 3, 1, 1, 1),
    (64, 48, 1024, 128, 5, 2, 2, 1),
    (16, 24, 2048, 96, 7, 1, 3, 2),
    (8, 8, 8192, 16, 11, 1, 5, 1),
]

FLOAT_DTYPES = [torch.float32, torch.float16]


class _GemsModuleView:
    """View over the backend module namespace actually used by flag_gems.

    The mthreads backend is imported under a short module name, so the
    authoritative namespace is the globals of the registered conv1d
    function itself.
    """

    def __init__(self):
        self._ns = flag_gems.conv1d.__globals__

    def __getattr__(self, name):
        return self._ns[name]

    def __setattr__(self, name, value):
        if name == "_ns":
            super().__setattr__(name, value)
        else:
            self._ns[name] = value


def _gems_conv1d_module():
    return _GemsModuleView()


class _KernelSpy:
    """Records launches of the True-1D kernel without changing behavior."""

    def __init__(self, real_entry):
        self.real_entry = real_entry
        self.calls = 0

    def __getitem__(self, grid):
        inner = self.real_entry[grid]

        def launch(*args, **kwargs):
            self.calls += 1
            return inner(*args, **kwargs)

        return launch


def _run_true1d_spy(monkeypatch, fn):
    module = _gems_conv1d_module()
    spy = _KernelSpy(module._conv1d_fwd_kernel)
    monkeypatch.setattr(module, "_conv1d_fwd_kernel", spy)
    result = fn()
    monkeypatch.undo()
    return result, spy.calls


def _expect_true1d(CIg, COg, dtype):
    """Mirror the structural dispatch rule for test expectations."""

    if dtype == torch.float32 and CIg < 32 and COg < 32:
        return False
    return True


def _forward_case(shape, dtype, bias=False):
    N, CI, L, CO, K, S, P, G = shape
    inp = torch.randn(shape[:1] + (CI, L), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((CO, CI // G, K), dtype=dtype, device=flag_gems.device)
    bias_t = None
    if bias:
        bias_t = torch.randn((CO,), dtype=dtype, device=flag_gems.device)
    return inp, weight, bias_t, S, P, G


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("shape", TRUE1D_FAMILIES, ids=["K3", "K5", "K7", "K11"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_family_forward(monkeypatch, shape, dtype):
    N, CI, L, CO, K, S, P, G = shape
    inp, weight, bias, S, P, G = _forward_case(shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=S, padding=P, dilation=1, groups=G
    )
    result, calls = _run_true1d_spy(
        monkeypatch,
        lambda: flag_gems.conv1d(inp, weight, None, S, P, 1, G),
    )
    expected = _expect_true1d(CI // G, CO // G, dtype)
    assert (
        calls >= 1
    ) == expected, f"dispatch mismatch: true1d={calls >= 1}, expected={expected}"
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("shape", TRUE1D_FAMILIES, ids=["K3", "K5", "K7", "K11"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_family_backward(monkeypatch, shape, dtype):
    N, CI, L, CO, K, S, P, G = shape
    Lout = (L + 2 * P - (K - 1) - 1) // S + 1
    inp = torch.randn(
        (N, CI, L), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        (CO, CI // G, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias = torch.randn((CO,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_out = torch.randn((N, CO, Lout), dtype=dtype, device=flag_gems.device)

    out = flag_gems.conv1d(inp, weight, bias, S, P, 1, G)
    out.backward(grad_out)

    ref_inp = utils.to_reference(inp.detach(), True)
    ref_weight = utils.to_reference(weight.detach(), True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_gi, ref_gw, ref_gb = torch.ops.aten.convolution_backward(
        ref_grad_out,
        ref_inp,
        ref_weight,
        [CO],
        [S],
        [P],
        [1],
        False,
        [0],
        G,
        [True, True, True],
    )

    reduce_dw = max(1, int((N * Lout) ** 0.5))
    if dtype == torch.float16:
        reduce_dw = max(reduce_dw, 1024)
    utils.gems_assert_close(inp.grad, ref_gi, dtype, reduce_dim=K)
    utils.gems_assert_close(weight.grad, ref_gw, dtype, reduce_dim=reduce_dw)
    utils.gems_assert_close(bias.grad, ref_gb, dtype, reduce_dim=reduce_dw)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize(
    "CI,CO",
    [
        (15, 64),
        (16, 64),
        (17, 64),
        (31, 64),
        (32, 64),
        (33, 64),
        (48, 127),
        (48, 128),
        (48, 129),
        (12, 31),
        (12, 32),
        (12, 33),
        (8, 15),
        (8, 16),
        (8, 17),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_dispatch_boundary(monkeypatch, CI, CO, dtype):
    inp = torch.randn((4, CI, 128), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((CO, CI, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(ref_inp, ref_weight, None, 1, 1)
    result, calls = _run_true1d_spy(
        monkeypatch,
        lambda: flag_gems.conv1d(inp, weight, None, 1, 1),
    )
    expected = _expect_true1d(CI, CO, dtype)
    assert (calls >= 1) == expected
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize(
    "L,K,S,P",
    [
        (100, 3, 1, 1),
        (333, 7, 1, 3),
        (129, 5, 2, 2),
        (257, 11, 1, 5),
        (1022, 3, 2, 1),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_tail_blocks(monkeypatch, L, K, S, P, dtype):
    inp = torch.randn((4, 48, L), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, K), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(ref_inp, ref_weight, None, S, P)
    result, calls = _run_true1d_spy(
        monkeypatch,
        lambda: flag_gems.conv1d(inp, weight, None, S, P),
    )
    assert calls >= 1
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_non_contiguous_fallback(monkeypatch, dtype):
    base = torch.randn((8, 64, 2048), dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :]
    weight = torch.randn((64, 32, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.contiguous(), True)
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(ref_inp, ref_weight, None, 1, 1)
    result, calls = _run_true1d_spy(
        monkeypatch,
        lambda: flag_gems.conv1d(inp, weight, None, 1, 1),
    )
    assert calls == 0, "non-contiguous input must fall back to conv2d routing"
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_packed_weight_cache_inplace_update(monkeypatch, dtype):
    module = _gems_conv1d_module()
    module._PACKED_WEIGHT_CACHE.clear()
    inp = torch.randn((4, 48, 256), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)

    out1, _ = _run_true1d_spy(
        monkeypatch, lambda: flag_gems.conv1d(inp, weight, None, 1, 2)
    )
    packed1 = module._get_packed_weight(weight)

    # Repeat call: packed weight must be the identical cached object and
    # the output must be bit-identical (same kernel, same cached operand).
    out2, _ = _run_true1d_spy(
        monkeypatch, lambda: flag_gems.conv1d(inp, weight, None, 1, 2)
    )
    packed2 = module._get_packed_weight(weight)
    assert packed1 is packed2
    assert torch.equal(out1, out2)

    # In-place update bumps _version: cache must be rebuilt with new values.
    weight.add_(1.0)
    out3, _ = _run_true1d_spy(
        monkeypatch, lambda: flag_gems.conv1d(inp, weight, None, 1, 2)
    )
    packed3 = module._get_packed_weight(weight)
    assert packed3 is not packed1

    ref_weight = utils.to_reference(weight, True)
    ref_inp = utils.to_reference(inp, True)
    # fp16 forward results accumulate a 240-term dot per element; scale the
    # absolute tolerance with the reduction length as gems_assert_close
    # provides (this test targets cache semantics, not kernel precision).
    reduce_dim = 5 * 48 if dtype == torch.float16 else 1
    ref3 = torch.nn.functional.conv1d(ref_inp, ref_weight, None, 1, 2)
    utils.gems_assert_close(out3, ref3, dtype, reduce_dim=reduce_dim)
    ref1_weight = ref_weight - 1.0
    ref1 = torch.nn.functional.conv1d(ref_inp, ref1_weight, None, 1, 2)
    utils.gems_assert_close(out1, ref1, dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(out2, ref1, dtype, reduce_dim=reduce_dim)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_packed_weight_data_swap(monkeypatch, dtype):
    module = _gems_conv1d_module()
    module._PACKED_WEIGHT_CACHE.clear()
    inp = torch.randn((4, 48, 256), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)
    flag_gems.conv1d(inp, weight, None, 1, 2)
    packed_old = module._get_packed_weight(weight)

    # Swapping .data keeps _version; data_ptr invalidation must catch it.
    weight.data = torch.randn_like(weight)
    out = flag_gems.conv1d(inp, weight, None, 1, 2)
    packed_new = module._get_packed_weight(weight)
    assert packed_new is not packed_old

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(ref_inp, ref_weight, None, 1, 2)
    utils.gems_assert_close(
        out, ref, dtype, reduce_dim=5 * 48 if dtype == torch.float16 else 1
    )


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_groups_and_bias(monkeypatch, dtype):
    inp = torch.randn((4, 24, 512), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((96, 8, 7), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((96,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=ref_bias, stride=1, padding=3, dilation=1, groups=3
    )
    result, calls = _run_true1d_spy(
        monkeypatch,
        lambda: flag_gems.conv1d(inp, weight, bias, 1, 3, 1, 3),
    )
    assert calls >= 1
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_weight_data_copy_contract(monkeypatch, dtype):
    """``weight.data.copy_`` is outside the pack-cache contract.

    Writes through ``.data`` views do not bump the tensor version
    counter -- PyTorch's autograd engine is blind to exactly the same
    operations (they also corrupt gradients in stock PyTorch).  This
    test pins the documented boundary and verifies that the supported
    rewrite (``weight.copy_`` under no_grad) invalidates the cache and
    produces correct results.
    """
    module = _gems_conv1d_module()
    module._PACKED_WEIGHT_CACHE.clear()
    inp = torch.randn((4, 48, 256), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    flag_gems.conv1d(inp, weight, None, 1, 2)  # warm the pack cache
    packed_before = module._get_packed_weight(weight)

    # Supported in-place rewrite: must invalidate and stay correct.
    with torch.no_grad():
        weight.copy_(torch.randn_like(weight))
    out = flag_gems.conv1d(inp, weight, None, 1, 2)
    packed_after = module._get_packed_weight(weight)
    assert packed_after is not packed_before
    ref_weight = utils.to_reference(weight, True)
    ref = torch.nn.functional.conv1d(ref_inp, ref_weight, None, 1, 2)
    utils.gems_assert_close(
        out, ref, dtype, reduce_dim=5 * 48 if dtype == torch.float16 else 1
    )

    # Unsupported write path: pin the documented stale-pack behavior so
    # any future change that alters this contract must update the cache
    # documentation in the operator module.
    weight.data.copy_(torch.full_like(weight, 3.0))
    flag_gems.conv1d(inp, weight, None, 1, 2)
    # Empirical pin: cache entry was NOT rebuilt by the .data write.
    assert module._get_packed_weight(weight) is packed_after


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_shared_storage_alias(monkeypatch, dtype):
    """Version-counter-sharing aliases invalidate the pack cache.

    ``as_strided`` aliases share the weight's version counter, so
    in-place writes through them are framework-visible and must be
    honored.  Raw storage-level writes (``set_`` from an unrelated
    tensor) are outside the tensor API and outside the contract.
    """
    module = _gems_conv1d_module()
    module._PACKED_WEIGHT_CACHE.clear()
    inp = torch.randn((4, 48, 256), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    flag_gems.conv1d(inp, weight, None, 1, 2)
    packed_before = module._get_packed_weight(weight)

    alias = torch.as_strided(weight, (64, 48, 5), (240, 5, 1))
    alias.mul_(2.0)
    out = flag_gems.conv1d(inp, weight, None, 1, 2)
    packed_after = module._get_packed_weight(weight)
    assert packed_after is not packed_before
    ref = torch.nn.functional.conv1d(
        ref_inp, utils.to_reference(weight, True), None, 1, 2
    )
    utils.gems_assert_close(
        out, ref, dtype, reduce_dim=5 * 48 if dtype == torch.float16 else 1
    )


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_gc_and_id_reuse(monkeypatch, dtype):
    """A dead weakref must never serve a cache entry, even when the
    replacement tensor at the same id matches all metadata fields."""

    import gc
    import weakref

    module = _gems_conv1d_module()
    module._PACKED_WEIGHT_CACHE.clear()
    inp = torch.randn((4, 48, 256), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)

    flag_gems.conv1d(inp, weight, None, 1, 2)
    packed = module._get_packed_weight(weight)

    # Forge the stale-entry hazard: dead weakref + matching metadata.
    dead = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)
    dead_ref = weakref.ref(dead)
    del dead
    gc.collect()
    if dead_ref() is not None:
        pytest.skip("allocator kept the forged dead tensor alive")
    key = id(weight)
    prev = module._PACKED_WEIGHT_CACHE.get(key)
    module._PACKED_WEIGHT_CACHE[key] = (
        dead_ref,
        int(weight._version),
        weight.data_ptr(),
        tuple(weight.shape),
        packed,
    )
    try:
        rebuilt = module._get_packed_weight(weight)
        assert rebuilt is not packed
        out = flag_gems.conv1d(inp, weight, None, 1, 2)
        ref = torch.nn.functional.conv1d(
            utils.to_reference(inp, True), utils.to_reference(weight, True), None, 1, 2
        )
        utils.gems_assert_close(
            out, ref, dtype, reduce_dim=5 * 48 if dtype == torch.float16 else 1
        )
    finally:
        if prev is None:
            module._PACKED_WEIGHT_CACHE.pop(key, None)
        else:
            module._PACKED_WEIGHT_CACHE[key] = prev

    # Real GC: entry is dropped with the weight.
    deadweight = torch.randn((64, 48, 5), dtype=dtype, device=flag_gems.device)
    dead_id = id(deadweight)
    flag_gems.conv1d(inp, deadweight, None, 1, 2)
    del deadweight
    gc.collect()
    entry = module._PACKED_WEIGHT_CACHE.get(dead_id)
    assert entry is None or entry[0]() is None


@pytest.mark.conv1d
@pytest.mark.skipif(not MTHREADS, reason="MTHREADS-specific True-1D kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv1d_true1d_requires_grad_training_step(monkeypatch, dtype):
    """A Parameter-style training step: forward, backward, optimizer
    update (in-place), then forward+backward again -- all with
    requires_grad=True weights, validated against references."""
    N, CI, L, CO, K, S, P = 4, 48, 256, 64, 5, 2, 2
    Lout = (L + 2 * P - (K - 1) - 1) // S + 1
    inp = torch.randn((N, CI, L), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (CO, CI, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad_out = torch.randn((N, CO, Lout), dtype=dtype, device=flag_gems.device)

    outs = []

    def step():
        w = weight.detach().clone().requires_grad_(True)
        out = flag_gems.conv1d(inp, w, None, S, P, 1, 1)
        outs.append(out.detach().clone())
        out.backward(grad_out)
        return w

    w1 = step()
    with torch.no_grad():
        weight.add_(0.1)  # optimizer-style update: version-visible
    w2 = step()

    ref_gi, ref_gw, _ = torch.ops.aten.convolution_backward(
        utils.to_reference(grad_out, True),
        utils.to_reference(inp, True),
        utils.to_reference(w1, True),
        [CO],
        [S],
        [P],
        [1],
        False,
        [0],
        1,
        [True, True, True],
    )
    utils.gems_assert_close(w1.grad, ref_gw, dtype, reduce_dim=4096)
    ref_gi2, ref_gw2, _ = torch.ops.aten.convolution_backward(
        utils.to_reference(grad_out, True),
        utils.to_reference(inp, True),
        utils.to_reference(w2, True),
        [CO],
        [S],
        [P],
        [1],
        False,
        [0],
        1,
        [True, True, True],
    )
    utils.gems_assert_close(w2.grad, ref_gw2, dtype, reduce_dim=4096)
    # dw does not depend on weight values (only inp and grad_out), so
    # equality of grads is expected; the *outputs* must reflect the
    # optimizer update instead.
    assert not torch.equal(outs[0], outs[1])
    ref_out2 = torch.nn.functional.conv1d(
        utils.to_reference(inp, True), utils.to_reference(w2, True), None, S, P
    )
    utils.gems_assert_close(
        outs[1], ref_out2, dtype, reduce_dim=5 * 48 if dtype == torch.float16 else 1
    )
