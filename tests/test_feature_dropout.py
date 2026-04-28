import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

FEATURE_DROPOUT_SHAPES = (
    [(2, 8, 4, 4)]
    if QUICK_MODE
    else [(2, 3), (4, 8, 16), (2, 16, 8, 8), (2, 32, 4, 4, 4)]
)


@pytest.mark.feature_dropout
@pytest.mark.parametrize("shape", FEATURE_DROPOUT_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_feature_dropout(shape, p, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.feature_dropout(inp, p, True)
    assert res_out.shape == inp.shape
    batch_size, num_channels = shape[0], shape[1]
    scale = 1.0 / (1.0 - p)
    inp_reshaped = inp.view(batch_size, num_channels, -1)
    out_reshaped = res_out.view(batch_size, num_channels, -1)
    for b in range(batch_size):
        for c in range(num_channels):
            channel_out = out_reshaped[b, c]
            channel_inp = inp_reshaped[b, c]
            if not torch.all(channel_out == 0).item():
                assert torch.allclose(
                    channel_out, channel_inp * scale, rtol=1e-4, atol=1e-5
                )
    out_by_channel = res_out.view(batch_size, num_channels, -1)
    dropped = sum(
        1
        for b in range(batch_size)
        for c in range(num_channels)
        if torch.all(out_by_channel[b, c] == 0)
    )
    total = batch_size * num_channels
    tolerance = max(0.3, 2.0 / (total**0.5)) if total < 50 else 0.2
    assert abs(dropped / total - p) < tolerance


@pytest.mark.feature_dropout
@pytest.mark.parametrize("shape", FEATURE_DROPOUT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_feature_dropout_no_train(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref = torch.feature_dropout(ref_inp, 0.5, False)
    with flag_gems.use_gems():
        res_out = torch.feature_dropout(inp, 0.5, False)
    gems_assert_close(res_out, ref, dtype)


@pytest.mark.feature_dropout
@pytest.mark.parametrize("shape", FEATURE_DROPOUT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_feature_dropout_p_zero(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref = torch.feature_dropout(ref_inp, 0.0, True)
    with flag_gems.use_gems():
        res_out = torch.feature_dropout(inp, 0.0, True)
    gems_assert_close(res_out, ref, dtype)


@pytest.mark.feature_dropout
@pytest.mark.parametrize("shape", FEATURE_DROPOUT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_feature_dropout_p_one(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.feature_dropout(inp, 1.0, True)
    gems_assert_equal(res_out, torch.zeros_like(inp))


@pytest.mark.feature_dropout_
@pytest.mark.parametrize("shape", FEATURE_DROPOUT_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.5])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_feature_dropout_inplace(shape, p, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_clone = inp.clone()
    with flag_gems.use_gems():
        res_out = torch.feature_dropout_(inp, p, True)
    assert res_out.data_ptr() == inp.data_ptr()
    batch_size, num_channels = shape[0], shape[1]
    scale = 1.0 / (1.0 - p)
    inp_reshaped = inp_clone.view(batch_size, num_channels, -1)
    out_reshaped = res_out.view(batch_size, num_channels, -1)
    for b in range(batch_size):
        for c in range(num_channels):
            channel_out = out_reshaped[b, c]
            channel_inp = inp_reshaped[b, c]
            if not torch.all(channel_out == 0).item():
                assert torch.allclose(
                    channel_out, channel_inp * scale, rtol=1e-4, atol=1e-5
                )
