import pytest
import torch

import flag_gems

from . import base

# torch.nn.GRU has precision issues when using fp16 or bf16 on CPU and GPU.
DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

_GRU_INPUT_SHAPES = [
    (1, 512, 128, 128, 1, False, False, True),
    (1, 10, 512, 512, 1, False, False, True),
    (16, 100, 512, 1024, 1, False, False, True),
    (64, 100, 512, 512, 1, False, False, True),
    (64, 100, 512, 512, 2, False, False, True),
    (16, 128, 256, 256, 2, False, False, True),
    (256, 100, 512, 512, 1, False, False, True),
    (64, 100, 512, 512, 1, True, False, True),
    (64, 100, 512, 512, 1, False, True, True),
    (64, 100, 512, 512, 1, False, False, False),
    (64, 100, 512, 2048, 1, False, False, True),
    (8, 1000, 80, 512, 1, False, False, True),
]


_GRU_DATA_SHAPES = [
    (64, 100, 512, 512, 1, False),
    (64, 100, 512, 512, 2, False),
    (16, 128, 256, 256, 2, True),
    (1, 512, 128, 128, 1, False),
    (16, 100, 512, 1024, 1, False),
    (256, 100, 512, 512, 1, False),
    (1, 10, 512, 512, 1, False),
    (8, 1000, 80, 512, 1, False),
]


class GRUBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _GRU_INPUT_SHAPES


class GRUDataBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _GRU_DATA_SHAPES


def gru_input_fn(shape, dtype, device):
    (
        batch_size,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        batch_first,
        has_biases,
    ) = shape

    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=device, dtype=dtype)
    gru.flatten_parameters()

    input_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    input = torch.randn(input_shape, dtype=dtype, device=device)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    params = tuple(gru._flat_weights)

    yield (
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


def gru_data_input_fn(shape, dtype, device):
    batch_size, seq_len, input_size, hidden_size, num_layers, has_biases = shape

    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=False,
    ).to(device=device, dtype=dtype)
    gru.flatten_parameters()

    lengths = [max(1, seq_len - i) for i in range(batch_size)]
    padded = torch.randn((batch_size, seq_len, input_size), dtype=dtype, device=device)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded,
        torch.tensor(lengths, dtype=torch.long),
        batch_first=True,
        enforce_sorted=True,
    )
    h0 = torch.randn((num_layers, batch_size, hidden_size), dtype=dtype, device=device)
    params = tuple(gru._flat_weights)

    yield (
        packed.data,
        packed.batch_sizes,
        h0,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        False,
    )


@pytest.mark.gru
def test_gru():
    bench = GRUBenchmark(
        input_fn=gru_input_fn,
        op_name="gru",
        torch_op=torch.gru,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.gru_data
def test_gru_data():
    bench = GRUDataBenchmark(
        input_fn=gru_data_input_fn,
        op_name="gru.data",
        torch_op=torch.ops.aten.gru.data,
        dtypes=DTYPES,
    )
    bench.run()
