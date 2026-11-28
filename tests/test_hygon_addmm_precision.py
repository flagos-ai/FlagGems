# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Long FP32 reductions use an independently checked high-precision oracle."""

import math

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon kernels"),
]


@pytest.mark.parametrize(
    "case,seed,specs",
    [
        (
            "R178",
            10178000,
            [((1,), (1,)), ((512, 8192), (8192, 1)), ((8192, 1), (1, 128))],
        ),
        (
            "R186",
            10186000,
            [((1,), (1,)), ((1, 3145728), (48, 1)), ((3145728, 1), (1, 48))],
        ),
    ],
)
@pytest.mark.addmm
def test_hygon_addmm_issue_precision(case, seed, specs):
    host, device = [], []
    for i, (shape, stride) in enumerate(specs, start=1):
        size = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        generator = torch.Generator().manual_seed(seed + i)
        base = torch.empty(size, dtype=torch.float32).normal_(
            0, 0.25, generator=generator
        )
        host.append(torch.as_strided(base, shape, stride))
        device.append(torch.as_strided(base.to(flag_gems.device), shape, stride))
    bias, a, b = host
    products = a.double() * b.double().t()
    exact = torch.tensor(
        [math.fsum((math.fsum(row), bias.item())) for row in products.tolist()],
        dtype=torch.float64,
    ).view(a.shape[0], b.shape[1])
    expected64 = torch.addmm(bias.double(), a.double(), b.double())
    torch.testing.assert_close(expected64, exact, rtol=1e-12, atol=1e-12)
    actual = flag_gems.addmm(*device).cpu()
    # Round an independently verified FP64 reference once, keeping the default
    # FP32 tolerances for the final comparison.
    torch.testing.assert_close(actual, expected64.float())


@pytest.mark.parametrize(
    "shape", [(1, 4096, 3), (3, 4097, 1), (1, 8193, 1), (5, 12289, 1)]
)
@pytest.mark.parametrize("alpha,beta", [(1, 1), (-0.5, 1.25), (1, 0), (0, 1)])
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_layouts(shape, alpha, beta, out_variant):
    m, k, n = shape
    generator = torch.Generator().manual_seed(18601)
    a_base = torch.randn(m, 2 * k + 1, generator=generator) * 0.25
    b_base = torch.randn(n, 2 * k + 1, generator=generator) * 0.25
    a, b = a_base[:, 1::2], b_base[:, 1::2].t()
    bias = torch.randn(m, n, generator=generator) * 0.25
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.addmm(
        bias.double(), a.double(), b.double(), alpha=alpha, beta=beta
    ).float()
    ag = a_base.to(flag_gems.device)[:, 1::2]
    bg = b_base.to(flag_gems.device)[:, 1::2].t()
    ig = bias.to(flag_gems.device)
    if out_variant:
        backing = torch.empty(m, 2 * n + 1, device=flag_gems.device)
        out = backing[:, 1::2]
        result = flag_gems.addmm_out(ig, ag, bg, alpha=alpha, beta=beta, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("k", [4096, 4097, 8193])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_cancellation(k):
    # Exact sum is small while FP32 accumulation loses the unit terms.
    a = torch.ones(1, k)
    a[:, 0] = 2**24
    a[:, -1] = -(2**24)
    b = torch.ones(k, 1)
    bias = torch.tensor([0.25])
    expected = torch.tensor([[k - 2 + 0.25]], dtype=torch.float32)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.0, 1.25])
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.addmm
def test_hygon_addmm_zero_alpha_ignores_nonfinite_inputs(beta, out_variant):
    a = torch.full((3, 4097), float("nan"), dtype=torch.float32).to(flag_gems.device)
    b = torch.full((4097, 1), float("nan"), dtype=torch.float32).to(flag_gems.device)
    bias_cpu = torch.tensor([[0.25], [-0.5], [1.0]], dtype=torch.float32)
    if beta == 0:
        bias_cpu.fill_(float("nan"))
        expected = torch.zeros(3, 1)
    else:
        expected = (bias_cpu.double() * beta).float()
    bias = bias_cpu.to(flag_gems.device)
    if out_variant:
        out = torch.empty(3, 1, device=flag_gems.device)
        actual = flag_gems.addmm_out(bias, a, b, alpha=0, beta=beta, out=out)
        assert actual.data_ptr() == out.data_ptr()
    else:
        actual = flag_gems.addmm(bias, a, b, alpha=0, beta=beta)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("alias", ["bias", "mat1", "mat2"])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_out_alias(alias):
    generator = torch.Generator().manual_seed(18603)
    a = torch.randn(1, 4097, generator=generator) * 0.25
    b = torch.randn(4097, 1, generator=generator) * 0.25
    bias = torch.tensor([[0.25]])
    expected = torch.addmm(bias.double(), a.double(), b.double()).float()
    ig, ag, bg = [t.to(flag_gems.device) for t in (bias, a, b)]
    out = ig if alias == "bias" else ag[:, :1] if alias == "mat1" else bg[:1, :]
    actual = flag_gems.addmm_out(ig, ag, bg, out=out)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("expanded", ["mat1", "mat2"])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_zero_strides(expanded):
    generator = torch.Generator().manual_seed(18604)
    a_base = (
        torch.randn(1 if expanded == "mat1" else 3, 4097, generator=generator) * 0.25
    )
    b_base = (
        torch.randn(1 if expanded == "mat2" else 4097, 1, generator=generator) * 0.25
    )
    a, b = a_base.expand(3, 4097), b_base.expand(4097, 1)
    ag = a_base.to(flag_gems.device).expand(3, 4097)
    bg = b_base.to(flag_gems.device).expand(4097, 1)
    bias = torch.tensor([0.125])
    expected = torch.addmm(bias.double(), a.double(), b.double()).float()
    actual = flag_gems.addmm(bias.to(flag_gems.device), ag, bg)
    torch.testing.assert_close(actual.cpu(), expected)
