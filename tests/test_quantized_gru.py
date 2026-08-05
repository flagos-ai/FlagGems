import pytest
import torch

import flag_gems


def _make_dynamic_quantized_gru(
    input_size,
    hidden_size,
    num_layers=1,
    bidirectional=False,
    batch_first=True,
    dtype=torch.qint8,
):
    float_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=batch_first,
        bidirectional=bidirectional,
    ).eval()
    float_gru.qconfig = (
        torch.ao.quantization.default_dynamic_qconfig
        if dtype == torch.qint8
        else torch.ao.quantization.float16_dynamic_qconfig
    )
    quantized_gru = torch.ao.nn.quantized.dynamic.GRU.from_float(float_gru)
    params = [module.param for module in quantized_gru._all_weight_values]
    return quantized_gru, params


@pytest.mark.parametrize("shape", [(4, 10, 16), (8, 32, 64)], ids=["small", "medium"])
@pytest.mark.parametrize("hidden_size", [16, 32, 257])
@pytest.mark.parametrize("weight_dtype", [torch.qint8, torch.float16])
@pytest.mark.parametrize(
    "num_layers,bidirectional", [(1, False), (2, True)], ids=["single", "stacked_bidir"]
)
def test_quantized_gru(shape, hidden_size, weight_dtype, num_layers, bidirectional):
    """Compare the dense Triton implementation with packed dynamic GRU."""
    from flag_gems.ops.quantized_gru import quantized_gru_input

    batch_size, seq_len, input_size = shape
    directions = 2 if bidirectional else 1
    ref_gru, params = _make_dynamic_quantized_gru(
        input_size,
        hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dtype=weight_dtype,
    )
    input_tensor = torch.randn(
        batch_size,
        seq_len,
        input_size,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    hx = torch.zeros(
        num_layers * directions,
        batch_size,
        hidden_size,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    with torch.no_grad():
        ref_output, ref_hx = ref_gru(input_tensor.cpu(), hx.cpu())

    output, out_hx = quantized_gru_input(
        input_tensor,
        hx,
        params,
        True,
        num_layers,
        0.0,
        False,
        bidirectional,
        True,
    )
    assert output.shape == ref_output.shape
    assert out_hx.shape == ref_hx.shape
    atol = 0.15 if weight_dtype == torch.qint8 else 0.03
    torch.testing.assert_close(output.cpu(), ref_output, rtol=0.08, atol=atol)
    torch.testing.assert_close(out_hx.cpu(), ref_hx, rtol=0.08, atol=atol)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_quantized_gru_packed_data(bidirectional):
    from flag_gems.ops.quantized_gru import quantized_gru_data

    batch_size, seq_len, input_size, hidden_size = 3, 5, 8, 20
    directions = 2 if bidirectional else 1
    ref_gru, params = _make_dynamic_quantized_gru(
        input_size,
        hidden_size,
        bidirectional=bidirectional,
        batch_first=False,
    )
    padded = torch.randn(seq_len, batch_size, input_size)
    lengths = torch.tensor([5, 3, 2], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded, lengths, enforce_sorted=True
    )
    hx_cpu = torch.zeros(directions, batch_size, hidden_size)
    with torch.no_grad():
        ref_output, ref_hx = ref_gru(packed, hx_cpu)

    output, out_hx = quantized_gru_data(
        packed.data.to(flag_gems.device),
        packed.batch_sizes,
        hx_cpu.to(flag_gems.device),
        params,
        True,
        1,
        0.0,
        False,
        bidirectional,
    )
    torch.testing.assert_close(output.cpu(), ref_output.data, rtol=0.08, atol=0.15)
    torch.testing.assert_close(out_hx.cpu(), ref_hx, rtol=0.08, atol=0.15)


def test_quantized_gru_rejects_training():
    from flag_gems.ops.quantized_gru import quantized_gru_input

    batch_size, seq_len, input_size = 4, 10, 16
    hidden_size = 16
    _, params = _make_dynamic_quantized_gru(input_size, hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, input_size, device=flag_gems.device)
    hx = torch.zeros(1, batch_size, hidden_size, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        quantized_gru_input(input_tensor, hx, params, True, 1, 0.0, True, False, True)
