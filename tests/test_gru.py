import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# torch.nn.GRU has precision issues when using fp16 or bf16 on CPU and GPU.
DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

_GRU_INPUT_SHAPES = [
    (6, 7, 10, 6, 2),
    (1, 7, 10, 6, 2),
    (1, 7, 10, 128, 2),
    (256, 5, 64, 512, 1),
    (4, 7, 10, 32, 1),
    (4, 7, 10, 64, 1),
    (4, 7, 10, 128, 1),
    (128, 3, 32, 512, 1),
    (136, 3, 32, 512, 1),
    (6, 7, 10, 6, 3),
    (6, 2, 10, 6, 2),
    (4, 7, 128, 32, 1),
]


@pytest.mark.gru
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_biases", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "input_size", "hidden_size", "num_layers"),
    _GRU_INPUT_SHAPES,
)
def test_gru(
    dtype,
    has_biases,
    bidirectional,
    batch_first,
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    num_layers,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    gru = torch.nn.GRU(
        input_size,
        hidden_size,
        num_layers,
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
    params = tuple(gru._flat_weights)

    ref_input = utils.to_reference(input)
    ref_h0 = utils.to_reference(h0)
    ref_params = tuple(utils.to_reference(param) for param in params)
    ref_out, ref_hn = torch.gru(
        ref_input,
        ref_h0,
        ref_params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )

    res_out, res_hn = flag_gems.gru(
        input,
        h0,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_hn, ref_hn, dtype)


_GRU_DATA_CASES = [
    (10, 6, 2, [7, 6, 5, 4, 3, 2]),
    (10, 6, 2, [7, 7, 5, 5, 3, 2]),
    (10, 6, 1, [7, 6, 5, 4, 3, 2]),
    (10, 6, 2, [7]),
    (10, 128, 2, [7]),
    (64, 512, 1, [5] * 128 + [3] * 128),
    (10, 64, 1, [7, 6, 5, 4, 3, 2]),
]


def _packed_fixture(
    dtype, input_size, hidden_size, num_layers, has_biases, bidirectional, lengths
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch = len(lengths)
    gru = torch.nn.GRU(
        input_size,
        hidden_size,
        num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
    ).to(device=flag_gems.device, dtype=dtype)
    gru.flatten_parameters()
    padded = torch.randn(
        (batch, max(lengths), input_size), device=flag_gems.device, dtype=dtype
    )
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded,
        torch.tensor(lengths, dtype=torch.long),
        batch_first=True,
        enforce_sorted=True,
    )
    num_directions = 2 if bidirectional else 1
    h0 = torch.randn(
        (num_layers * num_directions, batch, hidden_size),
        device=flag_gems.device,
        dtype=dtype,
    )
    return gru, packed, h0


@pytest.mark.gru_data
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("has_biases", [True, False])
@pytest.mark.parametrize(
    ("input_size", "hidden_size", "num_layers", "lengths"),
    _GRU_DATA_CASES,
)
def test_gru_data(
    dtype, bidirectional, input_size, hidden_size, num_layers, has_biases, lengths
):
    gru, packed, h0 = _packed_fixture(
        dtype,
        input_size,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        lengths,
    )
    params = tuple(gru._flat_weights)

    data = packed.data
    batch_sizes = packed.batch_sizes

    ref_data = utils.to_reference(data)
    ref_batch_sizes = utils.to_reference(batch_sizes)
    ref_h0 = utils.to_reference(h0)
    ref_params = tuple(utils.to_reference(param) for param in params)
    ref_out, ref_hn = torch.ops.aten.gru.data(
        ref_data,
        ref_batch_sizes,
        ref_h0,
        ref_params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
    )
    res_out, res_hn = flag_gems.gru_data(
        data,
        batch_sizes,
        h0,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_hn, ref_hn, dtype)
