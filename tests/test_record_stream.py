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

"""Correctness tests for ``aten::record_stream``.

``record_stream(Tensor(a!) self, Stream s) -> ()`` is caching-allocator
bookkeeping: it marks ``self``'s storage as used by ``s`` so the allocator cannot
hand the block out again before ``s`` completes. It returns ``None`` and reads no
element, so the checked contract is the return value, the untouched allocation
metadata, the untouched payload and the observed allocator block state.

Coverage exemptions: ``record_stream`` takes one tensor and returns ``None``, so
it has no broadcast form, no scalar operand and no autograd formula; the
allocator-lifetime test below covers the operator-specific runtime effect.
"""

import contextlib
import gc

import pytest
import torch

import flag_gems

from . import test_utils as tu

_DEVICE_TYPE = torch.device(flag_gems.device).type
_DEVICE_MODULE = getattr(torch, _DEVICE_TYPE, None)
_HAS_BLOCK_STATES = hasattr(getattr(_DEVICE_MODULE, "memory", None), "memory_snapshot")

_ALLOC_SHAPE = (64, 64)
_VIEW_BASE = (32, 64)
# Every layout row is cheap, so all of them stay in the quick smoke subset.
_VIEWS = ["contiguous", "transposed", "sliced", "expanded", "scalar0d"]
# The stream call forms are cheap as well and stay in quick too.
_STREAM_KINDS = ["fresh", "current", "device"]
_MULTI_STREAM_SHAPES = [(64, 64), (20, 320, 15)]
_MULTI_STREAM_COUNTS = [2, 4, 8]
# Quick keeps every stream count on the small shape, so no cheap branch is lost.
_MULTI_STREAM_ROWS = tu.selected_cases(
    [
        (shape, count)
        for shape in _MULTI_STREAM_SHAPES
        for count in _MULTI_STREAM_COUNTS
    ],
    quick=[((64, 64), count) for count in _MULTI_STREAM_COUNTS],
)
# Positive special values are default-only.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(tu.REQUIRED_DTYPES), quick=[])

_NEGATIVE_ROWS = [
    ("int_self", 1),
    ("list_self", [1, 2, 3]),
    ("none_self", None),
    ("float_stream", 3.14),
    ("none_stream", None),
]


def _accelerator_available():
    """Static capability gate: metadata only, no tensor and no operator call."""
    accelerator = torch.accelerator.current_accelerator()
    return (
        accelerator is not None
        and accelerator.type == _DEVICE_TYPE
        and torch.accelerator.device_count() > 0
    )


pytestmark = pytest.mark.skipif(
    not _accelerator_available(),
    reason="no accelerator of the tensor device type is available for record_stream",
)


def _allocation_metadata(tensor):
    """Storage identity and layout a record must leave untouched."""
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.dtype,
    )


def _oracle_tensor(inp, ref_inp):
    """Operand the native oracle can take.

    ``tu.to_reference`` moves its copy to CPU under ``--ref cpu``, but the native
    op requires the operand to share the stream's accelerator, so that mode uses
    a same-device copy with the same values and dtype.
    """
    return ref_inp if ref_inp.device == inp.device else inp.clone()


def _view_tensor(dtype, view):
    """Return ``(recorded, owner)``: the view to record plus its storage owner."""
    base = tu.make_input(dtype, _VIEW_BASE, ["-1", "1"])
    if view == "contiguous":
        return base, base
    if view == "transposed":
        return base.t(), base
    if view == "sliced":
        return base[4:20, 8:40], base
    if view == "expanded":
        return base[0].expand(_VIEW_BASE[0], _VIEW_BASE[1]), base
    if view == "scalar0d":
        return base[3, 5], base
    raise AssertionError(view)


def _stream_for(kind):
    if kind == "fresh":
        return torch.Stream()
    if kind == "current":
        return torch.accelerator.current_stream()
    if kind == "device":
        return torch.Stream(device=flag_gems.device)
    raise AssertionError(kind)


@contextlib.contextmanager
def _on_stream(stream):
    """Run the block on ``stream`` and restore the previous current stream."""
    previous = torch.accelerator.current_stream()
    torch.accelerator.set_stream(stream)
    try:
        yield
    finally:
        torch.accelerator.set_stream(previous)


def _quiesce():
    """Collect garbage, wait for pending streams, then release cached blocks."""
    gc.collect()
    torch.accelerator.synchronize()
    _DEVICE_MODULE.empty_cache()


def _pending_blocks():
    """Blocks the allocator keeps for a stream that recorded a released storage.

    Only ``inactive`` and ``active_allocated`` are settled states, so any other
    state is a block held for an outstanding recording stream (this build spells
    it ``active_pending_free``; older documentation says
    ``active_awaiting_free``).
    """
    snapshot = _DEVICE_MODULE.memory.memory_snapshot()
    return sorted(
        (block["state"], block["size"])
        for segment in snapshot
        for block in segment.get("blocks", ())
        if block["state"] not in ("inactive", "active_allocated")
    )


def _allocator_signature(record, dtype):
    """Record on a completed side stream, release storage, inspect bookkeeping.

    The held entry is allocator bookkeeping rather than elapsed time: a storage
    recorded on another stream stays out of the free pool until the
    allocator processes that stream's event, so a pure no-op candidate reports
    nothing held.
    """
    _quiesce()
    side = torch.Stream()
    work = tu.make_input(dtype, _ALLOC_SHAPE, ["-1", "1"])
    side.wait_stream(torch.accelerator.current_stream())
    with _on_stream(side):
        work.fill_(1)
    side.synchronize()
    record(work, side)
    del work
    held = _pending_blocks()

    # Once that stream completes the allocator can hand the block out again.
    torch.accelerator.synchronize()
    reuse = tu.make_input(dtype, _ALLOC_SHAPE, ["-1", "1"])
    reuse.fill_(3)
    return held, float(reuse.reshape(-1)[0].item())


@pytest.mark.record_stream
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test_record_stream_grid(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    stream = torch.accelerator.current_stream()
    before = _allocation_metadata(inp)
    ref_inp = tu.to_reference(inp)

    torch.ops.aten.record_stream(_oracle_tensor(inp, ref_inp), stream)
    result = flag_gems.record_stream(inp, stream)

    assert result is None
    assert _allocation_metadata(inp) == before
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.record_stream
@pytest.mark.parametrize("view", _VIEWS)
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test_record_stream_view_storage(dtype, view):
    recorded, owner = _view_tensor(dtype, view)
    stream = torch.Stream()
    before = _allocation_metadata(recorded)
    ref_recorded = tu.to_reference(recorded)

    torch.ops.aten.record_stream(_oracle_tensor(recorded, ref_recorded), stream)
    result = flag_gems.record_stream(recorded, stream)

    assert result is None
    # Recording a view records its storage: no copy, no rebind, same offset,
    # same strides.
    assert _allocation_metadata(recorded) == before
    assert recorded.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
    tu.assert_result_equal(recorded, ref_recorded)


@pytest.mark.record_stream
@pytest.mark.parametrize("shape,count", _MULTI_STREAM_ROWS)
def test_record_stream_multiple_streams(shape, count):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    before = _allocation_metadata(inp)
    ref_inp = tu.to_reference(inp)
    streams = [torch.Stream() for _ in range(count - 1)]
    streams.append(torch.accelerator.current_stream())

    oracle_inp = _oracle_tensor(inp, ref_inp)
    for stream in streams:
        torch.ops.aten.record_stream(oracle_inp, stream)
    results = [flag_gems.record_stream(inp, stream) for stream in streams]

    assert all(result is None for result in results)
    assert _allocation_metadata(inp) == before
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.record_stream
@pytest.mark.parametrize("stream_kind", _STREAM_KINDS)
def test_record_stream_stream_forms(stream_kind):
    inp = tu.make_input(torch.int32, (2, 19, 7), ["-1", "1"])
    stream = _stream_for(stream_kind)
    state = (stream.device_type, stream.device_index, stream.stream_id)
    before = _allocation_metadata(inp)
    ref_inp = tu.to_reference(inp)

    torch.ops.aten.record_stream(_oracle_tensor(inp, ref_inp), stream)
    result = flag_gems.record_stream(inp, stream)

    assert result is None
    # Recording must not rebind or reconfigure the stream it was given.
    assert (stream.device_type, stream.device_index, stream.stream_id) == state
    assert _allocation_metadata(inp) == before
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.record_stream
@pytest.mark.skipif(
    not _HAS_BLOCK_STATES,
    reason="the active backend exposes no allocator block-state snapshot",
)
@pytest.mark.parametrize("dtype", tu.REQUIRED_DTYPES)
def test_record_stream_allocator_lifetime(dtype):
    control_held, _ = _allocator_signature(lambda tensor, stream: None, dtype)
    ref_held, ref_reused = _allocator_signature(torch.ops.aten.record_stream, dtype)
    res_held, res_reused = _allocator_signature(flag_gems.record_stream, dtype)

    # The no-op control shows the fixture discriminates: without a real record
    # the released block goes straight back to the free pool.
    assert control_held == []
    # The native free records an event whose allocator bookkeeping remains
    # pending until the next allocation; a no-op candidate holds nothing.
    assert ref_held, "the native sequence left no held block"
    assert res_held == ref_held
    assert res_reused == ref_reused == 3.0


@pytest.mark.record_stream
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_record_stream_special_values(dtype, scenario):
    # The shared generator returns a fixed 5-element payload tensor for the
    # dtype/scenario pair; the shape is not a dimension of this helper.
    inp = tu.make_special_input(dtype, scenario)
    stream = torch.accelerator.current_stream()
    ref_inp = tu.to_reference(inp)

    torch.ops.aten.record_stream(_oracle_tensor(inp, ref_inp), stream)
    result = flag_gems.record_stream(inp, stream)

    assert result is None
    # NaN/Inf payloads must survive the call unchanged.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.record_stream
@pytest.mark.parametrize("case,value", _NEGATIVE_ROWS)
def test_record_stream_invalid_args(case, value):
    if case.endswith("_self"):
        args = (value, torch.Stream())
        expected = (RuntimeError, TypeError)
    else:
        args = (torch.zeros(4, device=flag_gems.device), value)
        expected = (RuntimeError, TypeError)
    with pytest.raises(expected):
        flag_gems.record_stream(*args)
