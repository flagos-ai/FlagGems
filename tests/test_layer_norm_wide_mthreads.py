# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Wide-row mixed-affine contracts using the shared numerical reference."""

import pytest
import torch

import flag_gems

from .test_layer_norm_mthreads import _check, _oracle

pytestmark = pytest.mark.native_layer_norm


@pytest.mark.parametrize("norm", [(256, 256), (256, 257)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", ["both", "weight", "bias", "none"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_wide_mixed_affine_contract(norm, dtype, affine, strided, eps):
    gen = torch.Generator().manual_seed(20261127)
    shape = (2, 3) + norm
    x = torch.randn(shape + ((2,) if strided else ()), generator=gen)
    x = x.to(device=flag_gems.device, dtype=dtype)
    if strided:
        x = x[..., 0]

    def parameter():
        value = torch.randn(norm + ((2,) if strided else ()), generator=gen)
        value = value.to(device=x.device, dtype=torch.float32)
        return value[..., 0] if strided else value

    w = parameter() if affine in ("both", "weight") else None
    b = parameter() if affine in ("both", "bias") else None
    actual = flag_gems.native_layer_norm(x, norm, w, b, eps)
    for value, ref in zip(actual, _oracle(x, norm, w, b, eps)):
        _check(value, ref, dtype)
