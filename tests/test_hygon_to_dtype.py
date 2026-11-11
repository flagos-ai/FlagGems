# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon dtype-conversion regressions, including exact spreadsheet strides."""

import importlib
import json
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.to_copy,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon cast coverage"),
]
CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/hygon-optimization/to_dtype/cases.json"
    ).read_text(encoding="utf-8")
)


def transfer_storage(x):
    storage = x.as_strided((x.untyped_storage().nbytes() // x.element_size(),), (1,), 0)
    return storage.to(flag_gems.device).as_strided(
        x.shape, x.stride(), x.storage_offset()
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["row"]))
@pytest.mark.parametrize("api", ["aten", "direct"])
def test_hygon_to_dtype_spreadsheet(case, api):
    meta, target = case["metadata"][0]
    dtype = getattr(torch, meta["dtype"].split(".")[-1])
    target = getattr(torch, target.split(".")[-1])
    size = (
        meta.get("storage_offset", 0)
        + 1
        + sum((n - 1) * s for n, s in zip(meta["shape"], meta["stride"]))
    )
    gen = torch.Generator().manual_seed(31997 + case["row"])
    if dtype == torch.bool:
        storage = torch.randint(0, 2, (size,), generator=gen).bool()
    elif dtype == torch.int64:
        storage = torch.randint(-(2**30), 2**30, (size,), generator=gen, dtype=dtype)
    else:
        storage = torch.randn(size, dtype=dtype, generator=gen)
    cpu = storage.as_strided(
        meta["shape"], meta["stride"], meta.get("storage_offset", 0)
    )
    x = transfer_storage(cpu)
    if api == "aten":
        with flag_gems.use_gems(include=["to_copy"]):
            out = torch.ops.aten.to.dtype(x, target)
        assert (out is x) == (dtype == target)
    else:
        out = flag_gems.to_copy(x, dtype=target)
        assert out is not x
    ref = cpu.to(target, copy=api == "direct")
    assert out.device == x.device and out.dtype == target
    assert out.stride() == ref.stride()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "view", ["contiguous", "transpose", "offset", "holes", "empty", "scalar"]
)
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
def test_hygon_to_dtype_layouts(view, memory_format):
    gen = torch.Generator().manual_seed(713)
    if view == "transpose":
        cpu = torch.randn((5, 17), generator=gen).t()
    elif view == "offset":
        cpu = torch.randn(86, generator=gen)[1:].view(5, 17)
    elif view == "holes":
        cpu = torch.randn((10, 34), generator=gen)[::2, ::2]
    elif view == "empty":
        cpu = torch.empty((0, 17))
    elif view == "scalar":
        cpu = torch.tensor(1.00390625)
    else:
        cpu = torch.randn((5, 17), generator=gen)
    x = transfer_storage(cpu)
    ref = cpu.to(torch.bfloat16, memory_format=memory_format)
    out = flag_gems.to_copy(x, dtype=torch.bfloat16, memory_format=memory_format)
    assert out.stride() == ref.stride() and out is not x
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


def test_hygon_to_dtype_bfloat16_rounding_and_special_values():
    bits = [
        0,
        0x80000000,
        1,
        0x00007FFF,
        0x00008000,
        0x00008001,
        0x007FFFFF,
        0x00800000,
        0x3F800001,
        0x3F808000,
        0x3F818000,
        0x7F7F8000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7F800001,
        0xFFC01234,
    ]
    cpu = torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    ref = cpu.to(torch.bfloat16)
    x = transfer_storage(cpu)
    with flag_gems.use_gems(include=["to_copy"]):
        out = x.to(torch.bfloat16).cpu()
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)
    finite_or_infinite = ~torch.isnan(ref)
    assert torch.equal(
        out.view(torch.int16)[finite_or_infinite],
        ref.view(torch.int16)[finite_or_infinite],
    )


@pytest.mark.parametrize("length", [34, 36, 2048, 2050, 2052, 4096, 4100])
@pytest.mark.parametrize("offset", [0, 1, 2, 4])
def test_hygon_to_dtype_packed_rounding_and_tail(monkeypatch, length, offset):
    module = importlib.import_module(flag_gems.to_copy.__module__)
    monkeypatch.setattr(module, "_PACKED_MIN_ELEMENTS", 0)
    bits = [
        0,
        0x80000000,
        1,
        0x7FFF,
        0x8000,
        0x8001,
        0x18000,
        0x007FFFFF,
        0x00800000,
        0x3F800001,
        0x3F808000,
        0x3F818000,
        0x7F7F8000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC01234,
    ]
    cpu = torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    cpu = cpu.repeat((length + offset + len(bits) - 1) // len(bits))[
        offset : offset + length
    ]
    ref = cpu.to(torch.bfloat16)
    with flag_gems.use_gems(include=["to_copy"]):
        result = transfer_storage(cpu).to(torch.bfloat16).cpu()
    torch.testing.assert_close(result, ref, atol=0, rtol=0, equal_nan=True)
    valid = ~torch.isnan(ref)
    assert torch.equal(result.view(torch.int16)[valid], ref.view(torch.int16)[valid])


@pytest.mark.parametrize("view", ["contiguous", "transpose", "offset"])
def test_hygon_to_dtype_bool_zero_comparison(view):
    cpu = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            1.0,
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            1e-38,
            -1e-38,
            1e-45,
            -1e-45,
        ]
    ).repeat(7)
    if view == "transpose":
        cpu = cpu.view(7, 13).t()
    elif view == "offset":
        cpu = cpu[1:]
    x = transfer_storage(cpu)
    with flag_gems.use_gems(include=["to_copy"]):
        out = torch.ops.aten._to_copy.default(x, dtype=torch.bool)
    ref = cpu.to(torch.bool)
    assert out.device == x.device and out.stride() == ref.stride()
    assert out.data_ptr() != x.data_ptr()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source,target", [(torch.bfloat16, torch.float32), (torch.int64, torch.float32)]
)
@pytest.mark.parametrize("view", ["contiguous", "transpose", "offset"])
def test_hygon_to_dtype_dense_pairs(source, target, view):
    if source == torch.int64:
        cpu = torch.tensor(
            [
                0,
                1,
                -1,
                2**24 - 1,
                2**24 + 1,
                -(2**24 + 1),
                2**40,
                -(2**40),
                2**63 - 1,
                -(2**63),
            ]
        ).repeat(7)
    else:
        cpu = torch.tensor(
            [
                0.0,
                -0.0,
                0.25,
                -0.25,
                float("inf"),
                float("-inf"),
                float("nan"),
                1e-38,
                -1e-38,
                1.00390625,
            ],
            dtype=source,
        ).repeat(7)
    if view == "transpose":
        cpu = cpu.view(7, 10).t()
    elif view == "offset":
        cpu = cpu[1:]
    x = transfer_storage(cpu)
    with flag_gems.use_gems(include=["to_copy"]):
        out = torch.ops.aten._to_copy.default(x, dtype=target)
    ref = cpu.to(target)
    assert out.device == x.device and out.stride() == ref.stride()
    assert out.data_ptr() != x.data_ptr()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0, equal_nan=True)


_EXTENDED_CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/hygon-optimization/to_dtype/extended-cases-20261110.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", _EXTENDED_CASES, ids=lambda c: str(c["case"]))
def test_hygon_to_dtype_extended_workbook_public(case):
    meta = case["metadata"][0][0]
    kwargs = case["metadata"][1]
    source = getattr(torch, meta["dtype"].split(".")[-1])
    target = getattr(torch, kwargs["dtype"].split(".")[-1])
    size = (
        meta.get("storage_offset", 0)
        + 1
        + sum((n - 1) * stride for n, stride in zip(meta["shape"], meta["stride"]))
    )
    gen = torch.Generator().manual_seed(20261110 + case["case"])
    if source == torch.int64:
        storage = torch.randint(-(2**30), 2**30, (size,), dtype=source, generator=gen)
    else:
        storage = torch.randn(size, dtype=source, generator=gen)
    cpu = storage.as_strided(
        meta["shape"], meta["stride"], meta.get("storage_offset", 0)
    )
    x = transfer_storage(cpu)
    x.requires_grad_(meta.get("requires_grad", False))
    kw = {"dtype": target}
    if "layout" in kwargs:
        kw["layout"] = torch.strided
    if "device" in kwargs:
        kw["device"] = x.device
    with torch.no_grad(), flag_gems.use_gems(include=["to_copy"]):
        out = torch.ops.aten._to_copy.default(x, **kw)
    ref = torch.ops.aten._to_copy.default(cpu, dtype=target)
    assert out.device == x.device and out.dtype == target
    assert out.stride() == ref.stride() and out.data_ptr() != x.data_ptr()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
