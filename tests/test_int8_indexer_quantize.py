# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from .int8_indexer_exact_checks import (  # noqa: F401
    test_adversarial_fixture_range_and_signs,
    test_bitceil_against_exact_rational_inequalities,
    test_division_rn_not_rounded_reciprocal_multiply,
    test_exact_oracle_agrees_with_unchanged_math_reference,
    test_floor_order_zero_and_full_finite_range,
    test_integer_signed_halfway_parities,
    test_round_even_midpoints_across_exponents,
)
from .int8_indexer_reference import quantize_reference
from .int8_indexer_reference_checks import (  # noqa: F401
    test_fp32_logarithm_precision_boundary_is_discriminating,
    test_nearest_even_signed_range_and_zero_scale,
    test_quantization_empty_and_power_of_two_thresholds,
)

pytestmark = pytest.mark.int8_indexer_quantize


@pytest.mark.int8_indexer_quantize
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_quantize_nearest_even_zero_and_scale_boundaries(dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_quantize

    x = torch.zeros(4, 128, dtype=dtype)
    x[0, :8] = torch.tensor([127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=dtype)
    x[1, 0] = 128
    raw = torch.nextafter(torch.tensor(2.0**-20), torch.tensor(float("inf")))
    x[3, 0] = (raw * 127).to(dtype)
    expected_q, expected_scale = quantize_reference(x)
    actual_q, actual_scale = int8_indexer_quantize(x.cuda())
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual_q.cpu(), expected_q, atol=0, rtol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@torch.inference_mode()
def test_public_quantize_exact_integer_oracle_and_ordinary_graph(dtype_name):
    if not torch.cuda.is_available():
        pytest.skip("GPU component test")
    from flag_gems.fused.int8_sparse_indexer import int8_indexer_quantize

    from .int8_indexer_exact_oracle import adversarial_tensors

    x_cpu, expected_q, expected_scales = adversarial_tensors(dtype_name)
    x = x_cpu.cuda()
    actual_q, actual_scales = int8_indexer_quantize(x)
    torch.testing.assert_close(actual_q.cpu(), expected_q, atol=0, rtol=0)
    assert torch.equal(
        actual_scales.cpu().view(torch.int32), expected_scales.view(torch.int32)
    )
    dequantized = actual_q.cpu().double() * actual_scales.cpu().double()
    error = (dequantized - x_cpu.double()).abs()
    assert torch.all(error <= expected_scales.double() * 0.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        int8_indexer_quantize(x)
        with torch.cuda.graph(graph, stream=stream):
            graph_q, graph_scales = int8_indexer_quantize(x)
    torch.cuda.current_stream().wait_stream(stream)
    # Refresh rows with adjacent fixture values while retaining all owners.
    x.copy_(x_cpu.flip(0))
    graph_q.fill_(17)
    graph_scales.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_q.cpu(), expected_q.flip(0), atol=0, rtol=0)
    assert torch.equal(
        graph_scales.cpu().view(torch.int32), expected_scales.flip(0).view(torch.int32)
    )
