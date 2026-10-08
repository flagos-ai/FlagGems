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

import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not utils.bf16_is_supported, reason="Device does not support bfloat16"
        ),
    ),
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            not utils.fp64_is_supported
            or flag_gems.vendor_name in ("ascend", "iluvatar", "mthreads"),
            reason="Backend dtype policy excludes float64",
        ),
    ),
]
SHAPES = [(), (0,), (2, 0, 3), (1,), (17,), (4097,), (96000,)]
KEEP_PROBABILITIES = [0.0, 0.2, 0.5, 0.8, 1.0]


@contextmanager
def _registered_dropout():
    library = torch.library.Library("aten", "IMPL")
    previous = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=library,
            include=["_fused_dropout", "_masked_scale"],
            registrar=flag_gems.GeneralOpRegistrar,
        )
        yield
    finally:
        library._destroy()
        flag_gems.current_work_registrar = previous


def _generator(inp, seed=12345):
    return torch.Generator(device=inp.device).manual_seed(seed)


def _assert_result(inp, output, mask, p):
    assert output.shape == mask.shape == inp.shape
    assert output.device == mask.device == inp.device
    assert output.dtype == inp.dtype
    assert mask.dtype == torch.uint8
    if inp.numel() and torch.ops.aten.is_non_overlapping_and_dense(inp):
        assert output.stride() == mask.stride() == inp.stride()
    else:
        assert output.stride() == torch.empty_like(inp).stride()
        assert mask.stride() == torch.empty_like(inp, dtype=torch.uint8).stride()
    assert output is not inp
    if inp.numel():
        assert output.data_ptr() != inp.data_ptr()
    values = inp.detach().cpu()
    actual = output.detach().cpu()
    bits = mask.cpu()
    assert ((bits == 0) | (bits == 1)).all()
    if p == 0:
        assert (bits == 0).all()
        assert torch.isnan(actual).all()
    elif p == 1:
        assert (bits == 1).all()
        torch.testing.assert_close(actual, values, rtol=0, atol=0)
    else:
        expected = (values.double() * bits.double() / p).to(inp.dtype)
        torch.testing.assert_close(actual, expected)


@pytest.mark.fused_dropout
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("p", KEEP_PROBABILITIES)
def test_fused_dropout(shape, dtype, p):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original = inp.cpu().clone()
    generator = _generator(inp)
    before = generator.get_state().clone()
    output, mask = flag_gems._fused_dropout(inp, p, generator=generator)
    _assert_result(inp, output, mask, p)
    torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)
    after = generator.get_state()
    assert torch.equal(before, after) == (inp.numel() == 0)
    if inp.numel() >= 96000 and 0 < p < 1:
        observed = mask.cpu().double().mean().item()
        deviation = 8 * math.sqrt(p * (1 - p) / inp.numel())
        assert abs(observed - p) < deviation


def _layout_input(layout, dtype):
    if layout == "channels_last":
        return torch.randn(
            (2, 17, 19, 3), dtype=dtype, device=flag_gems.device
        ).permute(0, 3, 1, 2)
    base = torch.randn((33, 66), dtype=dtype, device=flag_gems.device)
    return {
        "transpose": base.T,
        "slice": base[:, 1::2],
        "offset": base[1:, :],
        "expanded": base[:1, :].expand(33, 66),
        "empty_transpose": base[:0, :].T,
    }[layout]


@pytest.mark.fused_dropout
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "layout",
    ["transpose", "channels_last", "slice", "offset", "expanded", "empty_transpose"],
)
def test_fused_dropout_layout(layout, dtype):
    inp = _layout_input(layout, dtype)
    original = inp.cpu().clone()
    generator = _generator(inp)
    before = generator.get_state().clone()
    output, mask = flag_gems._fused_dropout(inp, 0.5, generator=generator)
    _assert_result(inp, output, mask, 0.5)
    if layout in ("transpose", "channels_last"):
        assert output.stride() == mask.stride() == inp.stride()
    torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)
    generator.set_state(before)
    replay_output, replay_mask = flag_gems._fused_dropout(inp, 0.5, generator=generator)
    assert torch.equal(mask.cpu(), replay_mask.cpu())
    torch.testing.assert_close(output.cpu(), replay_output.cpu(), rtol=0, atol=0)


@pytest.mark.fused_dropout
def test_fused_dropout_generator_replay_and_independence():
    inp = torch.ones(4097, device=flag_gems.device)
    first = _generator(inp, seed=712)
    second = _generator(inp, seed=712)
    before = first.get_state().clone()
    second_before = second.get_state().clone()
    output, mask = flag_gems._fused_dropout(inp, 0.5, generator=first)
    first_after = first.get_state().clone()
    assert torch.equal(second.get_state(), second_before)
    independent_output, independent_mask = flag_gems._fused_dropout(
        inp, 0.5, generator=second
    )
    assert torch.equal(mask.cpu(), independent_mask.cpu())
    assert torch.equal(output.cpu(), independent_output.cpu())
    assert torch.equal(first_after, second.get_state())
    _, successive_mask = flag_gems._fused_dropout(inp, 0.5, generator=first)
    assert not torch.equal(mask.cpu(), successive_mask.cpu())
    first.set_state(before)
    replay_output, replay_mask = flag_gems._fused_dropout(inp, 0.5, generator=first)
    assert torch.equal(mask.cpu(), replay_mask.cpu())
    assert torch.equal(output.cpu(), replay_output.cpu())
    assert torch.equal(first.get_state(), first_after)


@pytest.mark.fused_dropout
@pytest.mark.parametrize("size", [1, 3, 4, 5, 4097, 96000])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_fused_dropout_generator_word_increment(size, p):
    inp = torch.ones(size, device=flag_gems.device)
    generator = _generator(inp)
    before = generator.get_offset()
    flag_gems._fused_dropout(inp, p, generator=generator)
    assert generator.get_offset() == before + ((size + 3) // 4) * 4


@pytest.mark.fused_dropout
@pytest.mark.parametrize("boundary", [2**32, 2**34])
def test_fused_dropout_generator_counter_carry(boundary):
    inp = torch.ones(4101, device=flag_gems.device)
    generator = _generator(inp)
    generator.set_offset(boundary - 4)
    before = generator.get_state().clone()
    _, whole_mask = flag_gems._fused_dropout(inp, 0.5, generator=generator)
    whole_state = generator.get_state().clone()
    generator.set_state(before)
    _, head_mask = flag_gems._fused_dropout(inp[:4], 0.5, generator=generator)
    assert generator.get_offset() == boundary
    _, tail_mask = flag_gems._fused_dropout(inp[4:], 0.5, generator=generator)
    assert torch.equal(generator.get_state(), whole_state)
    assert torch.equal(whole_mask[:4].cpu(), head_mask.cpu())
    assert torch.equal(whole_mask[4:].cpu(), tail_mask.cpu())
    generator.set_offset(0)
    _, zero_offset_mask = flag_gems._fused_dropout(inp[4:], 0.5, generator=generator)
    # Crossing the uint32 word/counter boundary must not wrap onto offset zero.
    assert not torch.equal(tail_mask.cpu(), zero_offset_mask.cpu())


@pytest.mark.fused_dropout
def test_fused_dropout_generator_offset_overflow_leaves_state_unchanged():
    inp = torch.ones(5, device=flag_gems.device)
    generator = _generator(inp)
    generator.set_offset(2**64 - 4)
    before = generator.get_state().clone()
    with pytest.raises(RuntimeError, match="overflow"):
        flag_gems._fused_dropout(inp, 0.5, generator=generator)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.fused_dropout
def test_fused_dropout_default_generator_replay():
    inp = torch.ones(4097, device=flag_gems.device)
    device_api = flag_gems.runtime.torch_device_fn
    saved = device_api.get_rng_state().clone()
    try:
        device_api.manual_seed(42)
        before = device_api.get_rng_state().clone()
        output, mask = flag_gems._fused_dropout(inp, 0.5)
        after = device_api.get_rng_state().clone()
        _, next_mask = flag_gems._fused_dropout(inp, 0.5)
        assert not torch.equal(mask.cpu(), next_mask.cpu())
        device_api.set_rng_state(before)
        replay_output, replay_mask = flag_gems._fused_dropout(inp, 0.5)
        assert torch.equal(output.cpu(), replay_output.cpu())
        assert torch.equal(mask.cpu(), replay_mask.cpu())
        assert torch.equal(device_api.get_rng_state(), after)
    finally:
        device_api.set_rng_state(saved)


@pytest.mark.fused_dropout
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_fused_dropout_empty_default_rng(p):
    inp = torch.empty((3, 0, 7), device=flag_gems.device)
    device_api = flag_gems.runtime.torch_device_fn
    before = device_api.get_rng_state().clone()
    output, mask = flag_gems._fused_dropout(inp, p)
    _assert_result(inp, output, mask, p)
    assert torch.equal(before, device_api.get_rng_state())


@pytest.mark.fused_dropout
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int32, torch.int64, torch.complex64]
)
def test_fused_dropout_rejects_nonfloating_dtype(dtype):
    inp = torch.empty(17, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, ValueError), match="float|dtype"):
        flag_gems._fused_dropout(inp, 0.5)


@pytest.mark.fused_dropout
def test_fused_dropout_rejects_wrong_generator_device():
    inp = torch.ones(17, device=flag_gems.device)
    generator = torch.Generator(device="cpu").manual_seed(12)
    with pytest.raises((RuntimeError, ValueError), match="[Gg]enerator|device"):
        flag_gems._fused_dropout(inp, 0.5, generator=generator)


@pytest.mark.fused_dropout
def test_fused_dropout_concurrent_generator_reservations():
    inp = torch.ones(4097, device=flag_gems.device)
    generator = _generator(inp, seed=481)
    # Warm up compilation before making competing reservations. The native
    # reservation helper releases the GIL while holding the generator mutex.
    flag_gems._fused_dropout(inp, 0.5, generator=generator)
    flag_gems.runtime.torch_device_fn.synchronize()
    initial = generator.get_state().clone()
    serial_masks = [
        flag_gems._fused_dropout(inp, 0.5, generator=generator)[1] for _ in range(8)
    ]
    serial_state = generator.get_state().clone()
    generator.set_state(initial)
    barrier = Barrier(4)

    def sample(_):
        barrier.wait(timeout=30)
        return flag_gems._fused_dropout(inp, 0.5, generator=generator)[1]

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_masks = list(pool.map(sample, range(8)))
    flag_gems.runtime.torch_device_fn.synchronize()
    assert torch.equal(generator.get_state(), serial_state)
    serial = sorted(bytes(mask.cpu().tolist()) for mask in serial_masks)
    concurrent = sorted(bytes(mask.cpu().tolist()) for mask in concurrent_masks)
    assert len(set(concurrent)) == len(concurrent)
    assert concurrent == serial


@pytest.mark.fused_dropout
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="Native CUDA generator mutex regression"
)
def test_fused_dropout_reservations_interoperate_with_native_rand():
    inp = torch.ones(4097, device=flag_gems.device)
    generator = _generator(inp, seed=812)

    def gems_sample():
        return flag_gems._fused_dropout(inp, 0.5, generator=generator)[1]

    def native_sample():
        return torch.rand(inp.shape, device=inp.device, generator=generator)

    gems_sample()
    native_sample()
    flag_gems.runtime.torch_device_fn.synchronize()
    initial = generator.get_state().clone()
    initial_offset = generator.get_offset()
    native_sample()
    native_increment = generator.get_offset() - initial_offset
    generator.set_state(initial)
    gems_sample()
    gems_increment = generator.get_offset() - initial_offset
    serial_results = []
    for native_first in (False, True):
        generator.set_state(initial)
        if native_first:
            native = native_sample()
            mask = gems_sample()
        else:
            mask = gems_sample()
            native = native_sample()
        serial_results.append((mask.cpu(), native.cpu(), generator.get_state().clone()))

    generator.set_state(initial)
    barrier = Barrier(2)

    def simultaneous(fn):
        barrier.wait(timeout=30)
        return fn()

    with ThreadPoolExecutor(max_workers=2) as pool:
        gems_future = pool.submit(simultaneous, gems_sample)
        native_future = pool.submit(simultaneous, native_sample)
        mask = gems_future.result().cpu()
        native = native_future.result().cpu()
    state = generator.get_state()
    assert generator.get_offset() == initial_offset + native_increment + gems_increment
    assert any(
        torch.equal(mask, expected_mask)
        and torch.equal(native, expected_native)
        and torch.equal(state, expected_state)
        for expected_mask, expected_native, expected_state in serial_results
    )


@pytest.mark.fused_dropout
@pytest.mark.parametrize("p", [0.2, 0.8, 1.0])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_dropout_aten_autograd_uses_gems(monkeypatch, p, dtype):
    calls = {"forward": 0, "backward": 0}
    forward = flag_gems._fused_dropout
    backward = flag_gems._masked_scale

    @functools.wraps(forward)
    def forward_spy(*args, **kwargs):
        calls["forward"] += 1
        return forward(*args, **kwargs)

    @functools.wraps(backward)
    def backward_spy(*args, **kwargs):
        calls["backward"] += 1
        return backward(*args, **kwargs)

    monkeypatch.setattr(flag_gems, "_fused_dropout", forward_spy)
    monkeypatch.setattr(flag_gems, "_masked_scale", backward_spy)
    inp = torch.randn(4097, device=flag_gems.device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(inp)
    with _registered_dropout():
        keys = flag_gems.all_registered_keys()
        assert "_fused_dropout" in keys and "_masked_scale" in keys
        output, mask = torch.ops.aten._fused_dropout.default(
            inp, p, generator=_generator(inp)
        )
        output.backward(grad)
    assert calls["forward"] == 1
    assert calls["backward"] == 1
    _assert_result(inp, output, mask, p)
    expected = torch.where(mask.cpu().bool(), grad.cpu().double() / p, 0.0).to(dtype)
    torch.testing.assert_close(inp.grad.cpu(), expected)


@pytest.mark.fused_dropout
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads"
    and not getattr(flag_gems.runtime.torch_device_fn, "_initialized", False),
    reason=(
        "torch_musa does not expose its initialized RNG state to torch checkpoint; "
        "native and Gems checkpoint replay both fail on this adapter"
    ),
)
def test_fused_dropout_checkpoint_replays_default_rng():
    inp = torch.randn(4097, device=flag_gems.device, requires_grad=True)
    grad = torch.randn_like(inp)
    device_api = flag_gems.runtime.torch_device_fn
    saved = device_api.get_rng_state().clone()
    masks = []

    def apply_dropout(value):
        output, mask = torch.ops.aten._fused_dropout.default(value, 0.5)
        masks.append(mask.detach().cpu().clone())
        return output

    try:
        device_api.manual_seed(315)
        with _registered_dropout():
            output = checkpoint(apply_dropout, inp, use_reentrant=True)
            after_forward = device_api.get_rng_state().clone()
            output.backward(grad)
        assert len(masks) == 2
        assert torch.equal(masks[0], masks[1])
        assert torch.equal(device_api.get_rng_state(), after_forward)
        expected = grad.cpu() * masks[0] * 2
        torch.testing.assert_close(inp.grad.cpu(), expected, rtol=0, atol=0)
    finally:
        device_api.set_rng_state(saved)


@pytest.mark.fused_dropout
@pytest.mark.skipif(
    os.environ.get("FLAGGEMS_FUSED_DROPOUT_LARGE") != "1",
    reason="Set FLAGGEMS_FUSED_DROPOUT_LARGE=1 to allocate more than 2**31 elements",
)
def test_fused_dropout_large_64bit_indexing():
    size = 2**31 + 4097
    device_api = flag_gems.runtime.torch_device_fn
    if not hasattr(device_api, "mem_get_info"):
        pytest.skip("Large-index test requires a device free-memory query")
    free, _ = device_api.mem_get_info()
    # Input/output are float16; the mask is uint8. Leave a GiB for runtime work.
    required = size * (2 * 2 + 1) + 2**30
    if free < required:
        pytest.skip(f"Large-index test needs {required} free bytes, found {free}")
    inp = torch.ones(size, dtype=torch.float16, device=flag_gems.device)
    indices = torch.tensor(
        [0, 4096, 2**31 - 1, 2**31, size - 1],
        dtype=torch.int64,
        device=inp.device,
    )
    sentinels = torch.tensor(
        [-3.0, -2.0, 2.0, 3.0, 4.0], dtype=inp.dtype, device=inp.device
    )
    inp[indices] = sentinels
    output, mask = flag_gems._fused_dropout(inp, 1.0, generator=_generator(inp))
    sample_mask = mask[indices].cpu()
    assert mask.dtype == torch.uint8 and mask.shape == inp.shape
    assert (sample_mask == 1).all()
    torch.testing.assert_close(output[indices].cpu(), sentinels.cpu(), rtol=0, atol=0)


@pytest.mark.fused_dropout
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="CUDA capture API regression"
)
def test_fused_dropout_rejects_graph_capture_without_consuming_rng():
    inp = torch.ones(4097, device=flag_gems.device)
    generator = _generator(inp)
    flag_gems._fused_dropout(inp, 0.5, generator=generator)
    torch.cuda.synchronize()
    before = generator.get_state().clone()
    with pytest.raises(RuntimeError, match="does not support graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            flag_gems._fused_dropout(inp, 0.5, generator=generator)
    assert torch.equal(before, generator.get_state())


@pytest.mark.fused_dropout
@pytest.mark.parametrize("p", [0.0, 0.2, 0.8, 1.0])
def test_fused_dropout_special_values(p):
    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    output, mask = flag_gems._fused_dropout(inp, p, generator=_generator(inp))
    probability = torch.tensor(p, dtype=torch.float32)
    expected = inp.cpu() * mask.cpu().float() * probability.reciprocal()
    torch.testing.assert_close(output.cpu(), expected, equal_nan=True)
