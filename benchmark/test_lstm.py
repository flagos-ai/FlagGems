import pytest
import torch

from . import base, consts


class LSTMBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # LSTM core shapes follow generated single-layer benchmark coverage.
        self.shapes = [
            (2, 4, 8, 16, 1, False),
            (4, 8, 16, 32, 1, False),
            (8, 8, 16, 32, 1, False),
        ]


def lstm_input_fn(shape, dtype, device):
    batch_size, seq_len, input_size, hidden_size, num_layers, bidirectional = shape
    has_biases = True
    batch_first = True

    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=device, dtype=dtype)

    input = torch.randn(batch_size, seq_len, input_size, dtype=dtype, device=device)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    c0 = torch.randn(state_shape, dtype=dtype, device=device)
    params = tuple(lstm._flat_weights)

    yield (
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


@pytest.mark.lstm
def test_lstm():
    bench = LSTMBenchmark(
        input_fn=lstm_input_fn,
        op_name="lstm",
        torch_op=torch.lstm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
