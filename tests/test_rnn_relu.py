import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

RNN_HIDDEN_SIZES = [8, 16]


@pytest.mark.rnn_relu
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", RNN_HIDDEN_SIZES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rnn_relu(seq_len, batch_size, input_size, hidden_size, dtype, batch_first):
    """Test rnn_relu accuracy against PyTorch implementation"""
    if batch_first:
        input_tensor = torch.randn(
            batch_size, seq_len, input_size, dtype=dtype, device=flag_gems.device
        )
    else:
        input_tensor = torch.randn(
            seq_len, batch_size, input_size, dtype=dtype, device=flag_gems.device
        )

    # Create RNN model and get params
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu")
    rnn = rnn.to(dtype=dtype, device=flag_gems.device)
    params = tuple(rnn._flat_weights)
    hx = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor)
    ref_hx = utils.to_reference(hx)
    ref_params = tuple(utils.to_reference(p) for p in params)

    # Run PyTorch reference
    ref_out = torch.rnn_relu(
        ref_input, ref_hx, ref_params, True, 1, 0.0, False, False, batch_first
    )

    # Run FlagGems implementation
    with flag_gems.use_gems():
        res_out = torch.rnn_relu(
            input_tensor, hx, params, True, 1, 0.0, False, False, batch_first
        )

    # Compare outputs
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_close(res_out[1], ref_out[1], dtype)
