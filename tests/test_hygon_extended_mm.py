# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Metadata reconstruction regressions for the two extended workbook tabs."""

import importlib
import json
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon GEMM coverage"),
]
CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/hygon-optimization/mm/extended-cases-20261110.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["case"]))
def test_hygon_mm_extended_workbook_public(case):
    cpu_inputs, inputs = [], []
    for i, meta in enumerate(case["metadata"][0]):
        dtype = getattr(torch, meta["dtype"].split(".")[-1])
        offset = meta.get("storage_offset", 0)
        size = (
            offset
            + 1
            + sum((n - 1) * stride for n, stride in zip(meta["shape"], meta["stride"]))
        )
        gen = torch.Generator().manual_seed(case["seed"] + i)
        storage = torch.randn(size, dtype=dtype, generator=gen)
        cpu_inputs.append(storage.as_strided(meta["shape"], meta["stride"], offset))
        x = storage.to(flag_gems.device).as_strided(
            meta["shape"], meta["stride"], offset
        )
        x.requires_grad_(meta.get("requires_grad", False))
        inputs.append(x)
    ref = cpu_inputs[0].double() @ cpu_inputs[1].double()
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*inputs)
    assert output.device == inputs[0].device and output.dtype == inputs[0].dtype
    atol, rtol = (1e-10, 1e-10) if output.dtype == torch.float64 else (2e-4, 1e-4)
    torch.testing.assert_close(output.cpu().double(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("k", [17, 33, 65, 7888])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("seed", [20261109, 719])
def test_hygon_mm_swizzle_split_tail(monkeypatch, k, offset, seed):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    cpu_inputs, inputs = [], []
    for index, shape in enumerate([(17, k), (k, 19)]):
        gen = torch.Generator().manual_seed(seed + index)
        storage = torch.randn(
            shape[0] * shape[1] + offset, dtype=torch.float64, generator=gen
        )
        view = storage[offset:].view(shape)
        cpu_inputs.append(view)
        inputs.append(
            storage.to(flag_gems.device).as_strided(shape, view.stride(), offset)
        )
    key = (str(inputs[0].dtype), 17, 19, k, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(kind="swizzle_split", tm=1, tn=1, wm=1, warps=2, prefetch=4, split=2),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        result = torch.mm(*inputs)
    ref = cpu_inputs[0] @ cpu_inputs[1]
    torch.testing.assert_close(result.cpu(), ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("m,n", [(17, 19), (513, 64)])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_hygon_mm_outer_product_layouts(monkeypatch, dtype, m, n, offset, strided):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    cpu_inputs, inputs = [], []
    for i, shape in enumerate([(m, 1), (1, n)]):
        stride = (
            (3, 1) if i == 0 and strided else (2 * n, 2) if strided else (shape[1], 1)
        )
        size = offset + 1 + sum((d - 1) * s for d, s in zip(shape, stride))
        storage = torch.randn(
            size, dtype=dtype, generator=torch.Generator().manual_seed(719 + i)
        )
        cpu_inputs.append(storage.as_strided(shape, stride, offset))
        inputs.append(storage.to(flag_gems.device).as_strided(shape, stride, offset))
    key = (str(dtype), m, n, 1, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(kind="outer", rows=256, cols=64, warps=4, store=".cs"),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        out = torch.mm(*inputs)
    ref = cpu_inputs[0] @ cpu_inputs[1]
    assert out.dtype == dtype and out.device == inputs[0].device
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k", [(17, 19, 17), (33, 65, 32), (64, 32, 65)])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("banks", [(1, 16, 2, 8), (2, 8, 1, 16)])
def test_hygon_mm_banked_boundaries(
    monkeypatch, m, n, k, swap, transposed, offset, banks
):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    cpu_inputs, inputs = [], []
    for i, shape in enumerate([(m, k), (k, n)]):
        stride = (1, shape[0]) if transposed else (shape[1], 1)
        storage = torch.randn(
            shape[0] * shape[1] + offset,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(719 + i),
        )
        view = storage.as_strided(shape, stride, offset)
        cpu_inputs.append(view)
        inputs.append(storage.to(flag_gems.device).as_strided(shape, stride, offset))
    key = (str(inputs[0].dtype), m, n, k, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(
            kind="banked",
            tm=1,
            tn=1,
            wm=2,
            warps=4,
            prefetch=4,
            swap=swap,
            group=1,
            pipe=True,
            sa_vec=banks[0],
            sa_max=banks[1],
            sb_vec=banks[2],
            sb_max=banks[3],
        ),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*inputs)
    assert output.dtype == torch.float64 and output.device == inputs[0].device
    torch.testing.assert_close(
        output.cpu(), cpu_inputs[0] @ cpu_inputs[1], atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("k,split", [(17, 8), (65, 4), (129, 2), (2896, 4)])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_hygon_mm_fast_split_tail_and_empty_partitions(
    monkeypatch, k, split, swap, transposed, offset
):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    m, n = 17, 19
    cpu_inputs, inputs = [], []
    for i, shape in enumerate([(m, k), (k, n)]):
        stride = (1, shape[0]) if transposed else (shape[1], 1)
        storage = torch.randn(
            shape[0] * shape[1] + offset,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(197 + i),
        )
        cpu_inputs.append(storage.as_strided(shape, stride, offset))
        inputs.append(storage.to(flag_gems.device).as_strided(shape, stride, offset))
    key = (str(inputs[0].dtype), m, n, k, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(
            kind="fast_split",
            bm=64,
            bn=64,
            bk=32,
            warps=4,
            stages=2,
            split=split,
            swap=swap,
            group=8,
        ),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*inputs)
    assert output.dtype == torch.float32 and output.device == inputs[0].device
    torch.testing.assert_close(
        output.cpu().double(),
        cpu_inputs[0].double() @ cpu_inputs[1].double(),
        atol=2e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("k", [17, 171, 256])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_hygon_mm_cache_policy_public_boundaries(monkeypatch, k, swap, offset):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    m, n = 17, 19
    cpu_inputs, inputs = [], []
    for i, shape in enumerate([(m, k), (k, n)]):
        storage = torch.randn(
            shape[0] * shape[1] + offset,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(471 + i),
        )
        stride = (1, shape[0]) if i == 1 else (shape[1], 1)
        cpu_inputs.append(storage.as_strided(shape, stride, offset))
        inputs.append(storage.to(flag_gems.device).as_strided(shape, stride, offset))
    key = (str(inputs[0].dtype), m, n, k, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(
            kind="cache3",
            bm=64,
            bn=32,
            bk=16,
            warps=4,
            stages=2,
            swap=swap,
            tail=False,
            unroll=2,
            load="",
            store=".cg",
        ),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*inputs)
    torch.testing.assert_close(
        output.cpu().double(),
        cpu_inputs[0].double() @ cpu_inputs[1].double(),
        atol=2e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 15, 17, 18, 19, 20, 32])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_hygon_mm_short_fp64_tail_public_boundaries(monkeypatch, k, swap, offset):
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    m, n = 33, 37
    cpu_inputs, inputs = [], []
    for i, shape in enumerate([(m, k), (k, n)]):
        storage = torch.randn(
            shape[0] * shape[1] + offset,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(487 + i),
        )
        stride = (1, shape[0]) if i == 1 else (shape[1], 1)
        cpu_inputs.append(storage.as_strided(shape, stride, offset))
        inputs.append(storage.to(flag_gems.device).as_strided(shape, stride, offset))
    key = (str(inputs[0].dtype), m, n, k, *inputs[0].stride(), *inputs[1].stride())
    monkeypatch.setitem(
        configs.CONFIGS,
        key,
        dict(
            kind="bank_tail",
            tm=2,
            tn=2,
            wm=2,
            warps=8,
            prefetch=4,
            swap=swap,
            pipe=True,
            group=8,
            sa_vec=1,
            sa_max=16,
            sb_vec=2,
            sb_max=8,
            sa_per=1,
            sb_per=2,
        ),
    )
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*inputs)
    torch.testing.assert_close(
        output.cpu(), cpu_inputs[0] @ cpu_inputs[1], atol=1e-10, rtol=1e-10
    )
