import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lstm
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    (
        "batch_size",
        "seq_len",
        "input_size",
        "hidden_size",
        "num_layers",
        "has_biases",
        "bidirectional",
        "batch_first",
    ),
    [
        # Core single-layer batch-first case used by the generated benchmark.
        (2, 4, 8, 16, 1, True, False, True),
        # Multi-layer and sequence-first coverage checks layer-to-layer wiring.
        (2, 3, 5, 7, 2, False, False, False),
        # Bidirectional output concatenation uses both forward and reverse time order.
        (2, 3, 4, 6, 1, True, True, True),
        # Combined multi-layer bidirectional case verifies PyTorch flat-weight order.
        (2, 3, 4, 5, 2, False, True, False),
    ],
)
def test_lstm(
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    dtype,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=flag_gems.device, dtype=dtype)

    input_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    input = torch.randn(input_shape, device=flag_gems.device, dtype=dtype)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, device=flag_gems.device, dtype=dtype)
    c0 = torch.randn(state_shape, device=flag_gems.device, dtype=dtype)
    params = tuple(lstm._flat_weights)

    # The reference is computed in float64 on CPU whenever the input is float32.
    # Comparing two independent float32 implementations (FlagGems vs native CUDA)
    # against each other measures the *difference* of their rounding errors, which
    # for an LSTM accumulates to ~1e-4 -- the same order as the tolerance it would
    # have to be checked against, so the comparison would not be meaningful.
    # FlagGems' own deviation from the float64 result is ~1.2e-7, so a tight
    # tolerance against an accurate reference is both stricter and more informative.
    # PPU supports native Half/BFloat16 LSTM but not the promoted Float64
    # reference. Keep only these default-mode references in their input dtype;
    # CPU mode and other backends retain the original high-precision reference.
    upcast_ref = not (
        flag_gems.vendor_name == "thead"
        and not utils.TO_CPU
        and dtype in (torch.float16, torch.bfloat16)
    )

    ref_input = utils.to_reference(input, upcast_ref)
    ref_h0 = utils.to_reference(h0, upcast_ref)
    ref_c0 = utils.to_reference(c0, upcast_ref)
    ref_params = tuple(utils.to_reference(param, upcast_ref) for param in params)
    if (
        flag_gems.vendor_name == "thead"
        and not utils.TO_CPU
        and ref_input.dtype in (torch.float32, torch.float64)
    ):
        pytest.skip("PPU native LSTM reference does not support float32/float64")
    ref_out, ref_hn, ref_cn = torch.lstm(
        ref_input,
        (ref_h0, ref_c0),
        ref_params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )

    res_out, res_hn, res_cn = flag_gems.lstm(
        input,
        (h0, c0),
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )

    for result in (res_out, res_hn, res_cn):
        assert result.device == input.device
        assert result.dtype == dtype
    assert res_out.shape == (*input_shape[:-1], hidden_size * num_directions)
    assert res_hn.shape == state_shape
    assert res_cn.shape == state_shape

    atol = 3e-2 if dtype is torch.bfloat16 else 2e-2 if dtype is torch.float16 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
    utils.gems_assert_close(res_hn, ref_hn, dtype, atol=atol)
    utils.gems_assert_close(res_cn, ref_cn, dtype, atol=atol)
