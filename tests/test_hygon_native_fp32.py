# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public-route FP32 native pipeline boundaries and precision regressions."""

import importlib

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon FP32 pipeline"),
]


@pytest.mark.parametrize(
    "m,n,k",
    [(64, 128, 32), (65, 129, 39), (17, 19, 17), (67, 131, 32), (905, 1159, 1159)],
)
@pytest.mark.parametrize("offset", [0, 17])
@pytest.mark.parametrize("regime", ["random", "low_bits", "scaled", "basis"])
def test_hygon_mm_native_fp32_public(monkeypatch, m, n, k, offset, regime):
    gen = torch.Generator().manual_seed(20261110)
    a = torch.randn((m, k), generator=gen)
    b = torch.randn((k, n), generator=gen)
    if regime == "low_bits":
        a.zero_()
        a[:, 0] = 1.0 + 2.0**-22
        a[:, 1] = -1.0
        b.fill_(1.0)
    elif regime == "scaled":
        a *= 2.0**100
        b *= 2.0**-100
    elif regime == "basis":
        a.zero_()
        a[torch.arange(m), (torch.arange(m) * 13) % k] = 1.0
        b = ((torch.arange(k * n).reshape(k, n) % 37) - 18).float()
    inputs = []
    for cpu in [a, b]:
        storage = torch.empty(
            offset + cpu.numel(), device=flag_gems.device, dtype=cpu.dtype
        )
        view = storage.as_strided(cpu.shape, cpu.stride(), offset)
        view.copy_(cpu)
        inputs.append(view)
    module = importlib.import_module(flag_gems.mm.__module__)
    configs = importlib.import_module(module.__package__ + ".mm_config")
    key = (str(a.dtype), m, n, k, *a.stride(), *b.stride())
    monkeypatch.setitem(configs.CONFIGS, key, dict(kind="native_fp32", group=0))
    with torch.no_grad(), flag_gems.use_gems(include=["mm"]):
        result = torch.mm(*inputs)
    assert result.dtype == torch.float32 and result.device == inputs[0].device
    ref = a.double() @ b.double()
    atol, rtol = (0, 0) if regime in ("low_bits", "basis") else (2e-4, 1e-4)
    torch.testing.assert_close(result.cpu().double(), ref, atol=atol, rtol=rtol)
