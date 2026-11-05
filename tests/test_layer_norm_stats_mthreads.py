# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Statistics shape, ownership, and fallback contract across public interfaces."""

import math

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_equal
from .test_layer_norm_mthreads import _check, _oracle

pytestmark = pytest.mark.native_layer_norm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,norm",
    [
        ((2, 3, 128), (128,)),
        ((2, 3, 128), (3, 128)),
        ((3, 128), (3, 128)),
        ((0, 2, 128), (128,)),
        ((2, 65537), (65537,)),
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_stats_shape_values_and_independent_storage(dtype, shape, norm, native):
    generator = torch.Generator().manual_seed(20261107)
    host = torch.randn(shape + (2,), generator=generator).to(dtype)
    x = host.to(flag_gems.device)[..., 0]
    fn = flag_gems.native_layer_norm if native else flag_gems.layer_norm
    first = fn(x, norm)
    reference = _oracle(x, norm, None, None, 1e-5)
    m = x.numel() // math.prod(norm)
    expected_shape = shape[: -len(norm)] + (1,) * len(norm) if native else (m,)
    assert first[1].shape == first[2].shape == expected_shape
    for actual, expected in zip(first, reference):
        _check(actual, expected.reshape(actual.shape), dtype)
    snapshots = [value.cpu().clone() for value in first]
    second = fn(x * 2, norm)
    for actual, saved in zip(first, snapshots):
        gems_assert_equal(actual.cpu(), saved)
    if m:
        assert first[1].data_ptr() != first[2].data_ptr()
        assert first[1].data_ptr() != second[1].data_ptr()
        assert first[2].data_ptr() != second[2].data_ptr()
    assert first[1].is_contiguous() and first[2].is_contiguous()
