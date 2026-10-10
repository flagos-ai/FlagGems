# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.swiglu_oai,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def reference(x, limit=7.0, alpha=1.702, beta=1.0):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return (gate * torch.sigmoid(alpha * gate)) * (up + beta)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(0, 768), (2, 0), (1, 770), (7, 3072), (2, 3, 768)])
@pytest.mark.parametrize("parameters", [(7.0, 1.702, 1.0), (3.0, 0.75, -0.5)])
def test_swiglu_oai(shape, dtype, parameters):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype, device="cuda") * 8
    actual = flag_gems.swiglu_oai(x, *parameters)
    expected = reference(x, *parameters)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.032 if dtype == torch.bfloat16 else 0.004,
        rtol=0.016 if dtype == torch.bfloat16 else 0.002,
    )
    assert actual.shape == (*shape[:-1], shape[-1] // 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_swiglu_oai_strides_graph(dtype):
    x = torch.randn((3, 1540), dtype=dtype, device="cuda")[:, ::2]
    flag_gems.swiglu_oai(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.swiglu_oai(x)
    for multiplier in (2.0, -3.0):
        x.copy_(torch.randn_like(x) * multiplier)
        graph.replay()
        torch.testing.assert_close(
            output,
            reference(x),
            atol=0.032 if dtype == torch.bfloat16 else 0.004,
            rtol=0.016 if dtype == torch.bfloat16 else 0.002,
        )


def test_swiglu_oai_boundaries():
    x = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 0.0, 1.0, -2.0, 3.0, 4.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    torch.testing.assert_close(
        flag_gems.swiglu_oai(x), reference(x), equal_nan=True, atol=0.032, rtol=0.016
    )
    with pytest.raises(ValueError):
        flag_gems.swiglu_oai(x[:, :7])
    with pytest.raises(NotImplementedError):
        flag_gems.swiglu_oai(x.float())
    with pytest.raises(ValueError):
        flag_gems.swiglu_oai(x, limit=-1)
    with pytest.raises(NotImplementedError):
        flag_gems.swiglu_oai(
            torch.randn((2, 3, 8), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        )
