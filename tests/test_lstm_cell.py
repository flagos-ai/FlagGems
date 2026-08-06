import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes: (batch_size, input_size, hidden_size)
LSTM_CELL_SHAPES = [
    (1, 64, 64),
    (16, 128, 128),
    (32, 256, 256),
    (64, 256, 256),
    (64, 512, 512),
]


@pytest.mark.lstm_cell
@pytest.mark.parametrize("shape", LSTM_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_lstm_cell(shape, dtype):
    """Test lstm_cell accuracy against PyTorch reference."""
    batch_size, input_size, hidden_size = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create inputs
    input_tensor = torch.randn(
        batch_size, input_size, dtype=dtype, device=flag_gems.device
    )
    h_prev = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    c_prev = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    w_ih = torch.randn(
        4 * hidden_size, input_size, dtype=dtype, device=flag_gems.device
    )
    w_hh = torch.randn(
        4 * hidden_size, hidden_size, dtype=dtype, device=flag_gems.device
    )
    b_ih = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)
    b_hh = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)

    # Reference computation (upcast to fp32/fp64 for fair comparison since
    # our kernel computes GEMM in fp32 via explicit cast + allow_tf32=False)
    ref_input = utils.to_reference(input_tensor, upcast=True)
    ref_h = utils.to_reference(h_prev, upcast=True)
    ref_c = utils.to_reference(c_prev, upcast=True)
    ref_w_ih = utils.to_reference(w_ih, upcast=True)
    ref_w_hh = utils.to_reference(w_hh, upcast=True)
    ref_b_ih = utils.to_reference(b_ih, upcast=True)
    ref_b_hh = utils.to_reference(b_hh, upcast=True)

    ref_hy, ref_cy = torch.lstm_cell(
        ref_input, [ref_h, ref_c], ref_w_ih, ref_w_hh, ref_b_ih, ref_b_hh
    )

    # FlagGems computation
    with flag_gems.use_gems():
        res_hy, res_cy = torch.lstm_cell(
            input_tensor, [h_prev, c_prev], w_ih, w_hh, b_ih, b_hh
        )

    # Kernel computes in fp32 (all inputs cast to fp32 before tl.dot)
    # Error comes only from fp32 associativity differences in tiling
    atol = 1e-3
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("shape", LSTM_CELL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_lstm_cell_no_bias(shape, dtype):
    """Test lstm_cell accuracy without biases."""
    batch_size, input_size, hidden_size = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create inputs without biases
    input_tensor = torch.randn(
        batch_size, input_size, dtype=dtype, device=flag_gems.device
    )
    h_prev = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    c_prev = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    w_ih = torch.randn(
        4 * hidden_size, input_size, dtype=dtype, device=flag_gems.device
    )
    w_hh = torch.randn(
        4 * hidden_size, hidden_size, dtype=dtype, device=flag_gems.device
    )

    # Reference computation (upcast for fair comparison)
    ref_input = utils.to_reference(input_tensor, upcast=True)
    ref_h = utils.to_reference(h_prev, upcast=True)
    ref_c = utils.to_reference(c_prev, upcast=True)
    ref_w_ih = utils.to_reference(w_ih, upcast=True)
    ref_w_hh = utils.to_reference(w_hh, upcast=True)

    ref_hy, ref_cy = torch.lstm_cell(ref_input, [ref_h, ref_c], ref_w_ih, ref_w_hh)

    # FlagGems computation
    with flag_gems.use_gems():
        res_hy, res_cy = torch.lstm_cell(input_tensor, [h_prev, c_prev], w_ih, w_hh)

    atol = 1e-3
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)
