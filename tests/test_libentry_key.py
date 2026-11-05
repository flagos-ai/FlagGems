# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Key construction must observe current arguments on every invocation."""

import importlib
from types import SimpleNamespace

import pytest
import torch

runtime = importlib.import_module("flag_gems.utils.libentry")


@pytest.mark.parametrize(
    "value,bucket",
    [
        (-(2**63) - 1, "i64"),
        (-(2**31) - 1, "i64"),
        (-(2**31), "i32"),
        (2**31 - 1, "i32"),
        (2**31, "i64"),
        (2**63 - 1, "i64"),
        (2**63, "u64"),
        (2**64 - 1, "u64"),
        (2**64, "i64"),
        (True, "i32"),
        (False, "i32"),
        (None, type(None)),
        (1.25, float),
        ("text", str),
    ],
)
def test_key_scalar_roles(value, bucket):
    entry = SimpleNamespace(
        divisibility=16, _tensor_spec_fn=runtime.LibEntry._resolve_tensor_spec_fn()
    )
    assert runtime.LibEntry.key(entry, [value], [value], [value]) == (
        (type(value), value),
        bucket,
        value,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_key_observes_tensor_storage_offset_and_dtype(monkeypatch, dtype):
    monkeypatch.setattr(runtime, "device", SimpleNamespace(vendor_name="mthreads"))
    entry = SimpleNamespace(
        divisibility=16, _tensor_spec_fn=runtime.LibEntry._resolve_tensor_spec_fn()
    )
    tensor = torch.empty(16, dtype=dtype, device="cpu")
    for view in [tensor, tensor[1:], tensor[8:]]:
        assert runtime.LibEntry.key(entry, [view], [view], []) == (
            (dtype, view.data_ptr() % 16 == 0),
            dtype,
        )


def test_key_rechecks_mutable_tensor_like(monkeypatch):
    monkeypatch.setattr(runtime, "device", SimpleNamespace(vendor_name="mthreads"))
    entry = SimpleNamespace(
        divisibility=16, _tensor_spec_fn=runtime.LibEntry._resolve_tensor_spec_fn()
    )
    tensor = SimpleNamespace(dtype="f32", pointer=32)
    tensor.data_ptr = lambda: tensor.pointer
    first = runtime.LibEntry.key(entry, [tensor], [tensor], [7])
    tensor.pointer = 33
    tensor.dtype = "f16"
    second = runtime.LibEntry.key(entry, [tensor], [tensor], [9])
    assert first == (("f32", True), "f32", 7)
    assert second == (("f16", False), "f16", 9)


def test_key_descriptor_mutation(monkeypatch):
    class Descriptor:
        shape = [32, 64]
        strides = [64, 1]
        block_shape = [16, 32]
        padding = "zero"

    # This test owns the resolved type and restores it, unlike the older fixture.
    monkeypatch.setattr(runtime, "_TENSOR_DESCRIPTOR_TYPE", Descriptor)
    entry = SimpleNamespace(
        divisibility=16, _tensor_spec_fn=runtime.LibEntry._resolve_tensor_spec_fn()
    )
    descriptor = Descriptor()
    for size in [32, 48]:
        descriptor.shape = [size, 64]
        normalized = (
            "TensorDescriptor",
            (size, 64),
            (64, 1),
            (16, 32),
            "zero",
        )
        assert runtime.LibEntry.key(
            entry, [descriptor], [descriptor], [descriptor]
        ) == ((tuple, normalized), tuple, normalized)


def test_key_empty_and_single_use_iterables():
    entry = SimpleNamespace(
        divisibility=16, _tensor_spec_fn=runtime.LibEntry._resolve_tensor_spec_fn()
    )
    assert runtime.LibEntry.key(entry, [], [], []) == ()
    assert runtime.LibEntry.key(
        entry, iter([None, 4]), iter([3, 1.5]), iter([8, "layout"])
    ) == ((type(None), None), (int, 4), "i32", float, 8, "layout")
