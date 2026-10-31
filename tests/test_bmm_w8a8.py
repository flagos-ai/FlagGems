# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch

import flag_gems
from benchmark.test_bmm_w8a8 import make_channel_int8_bmm_inputs

from .accuracy_utils import gems_assert_close, gems_assert_equal

CONTRACT_SHAPES = [
    (1, 1, 1, 1),
    (2, 33, 65, 129),
    (4, 64, 128, 256),
    (2, 129, 17, 257),
    (1, 1, 1024, 4096),
    (1, 4, 1024, 4096),
    (1, 16, 1024, 4096),
    (1, 32, 1024, 4096),
    (1, 64, 1024, 4096),
    (1, 65, 1024, 4096),
    (1, 71, 1024, 4096),
    (1, 72, 1024, 4096),
    (1, 160, 1024, 4096),
    (1, 161, 1024, 4096),
    (1, 224, 1024, 4096),
    (1, 225, 1024, 4096),
    (1, 321, 1024, 4096),
    (1, 512, 1024, 4096),
    (1, 513, 1024, 4096),
]
BMM_CASES = list(
    product(
        CONTRACT_SHAPES,
        (torch.bfloat16, torch.float16, torch.float32),
        (2, 3),
        (False, True),
    )
)


requires_channel_int8_bmm = pytest.mark.skipif(
    "bmm_w8a8" not in vars(flag_gems),
    reason="backend does not export channel-wise INT8 BMM",
)


@pytest.mark.parametrize("shape,dtype,scale_rank,use_out", BMM_CASES)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm(shape, dtype, scale_rank, use_out):
    batch, rows, columns = shape[:3]
    activation, weight, activation_scale, weight_scale = make_channel_int8_bmm_inputs(
        shape, flag_gems.device
    )
    reference = torch.bmm(activation.cpu().long(), weight.cpu().long()).float()
    reference *= activation_scale.cpu()[:, :, None]
    reference *= weight_scale.cpu()[:, None, :]
    activation_scale = (
        activation_scale[:, :, None] if scale_rank == 3 else activation_scale
    )
    weight_scale = weight_scale[:, None, :] if scale_rank == 3 else weight_scale
    output = (
        torch.empty((batch, rows, columns), device=activation.device, dtype=dtype)
        if use_out
        else None
    )
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, out_dtype=dtype, out=output
    )
    gems_assert_close(actual.cpu(), reference.to(dtype), dtype, atol=0)


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 5, 7, 131072), (2, 0, 7, 32), (2, 5, 0, 32), (2, 5, 7, 0)]
)
@pytest.mark.bmm_w8a8
@requires_channel_int8_bmm
def test_channel_int8_bmm_empty(shape, use_out):
    batch, rows, columns, reduction = shape
    activation = torch.empty(
        shape[:2] + (reduction,), device=flag_gems.device, dtype=torch.int8
    )
    weight = torch.empty(
        (batch, columns, reduction), device=activation.device, dtype=torch.int8
    ).transpose(1, 2)
    activation_scale = torch.ones((batch, rows), device=activation.device)
    weight_scale = torch.ones((batch, columns), device=activation.device)
    output = (
        torch.empty(
            (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
        )
        if use_out
        else None
    )
    actual = flag_gems.bmm_w8a8(
        activation, weight, activation_scale, weight_scale, out=output
    )
    reference = torch.zeros(
        (batch, rows, columns), device=activation.device, dtype=torch.bfloat16
    )
    gems_assert_equal(actual.cpu(), reference.cpu())
