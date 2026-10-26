import pytest
import torch

from . import base, consts


class LSTMBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 4, 8, 16, 1, False), (4, 8, 16, 32, 1, False), (8, 8, 16, 32, 1, False)]


def _case_fn(shape, dtype):
    del dtype
    batch, seq, input_size, hidden, layers, bidirectional = shape
    directions = 2 if bidirectional else 1
    yield base.BenchmarkCasePlan(
        shape={"input": [batch, seq, input_size], "hx": [layers * directions, batch, hidden]},
        params={"has_biases": True, "num_layers": layers, "dropout": 0.0, "train": False, "bidirectional": bidirectional, "batch_first": True},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    if base.vendor_name == "thead" and dtype == torch.float32:
        pytest.skip("PPU native LSTM does not support float32")
    batch, seq, input_size, hidden, layers, bidirectional = plan.builder_args[0]
    module = torch.nn.LSTM(input_size, hidden, layers, bias=True, bidirectional=bidirectional, batch_first=True).to(device=device, dtype=dtype)
    directions = 2 if bidirectional else 1
    state_shape = (layers * directions, batch, hidden)
    inp = torch.randn((batch, seq, input_size), dtype=dtype, device=device)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    c0 = torch.randn(state_shape, dtype=dtype, device=device)
    p = plan.params
    return inp, (h0, c0), tuple(module._flat_weights), p["has_biases"], p["num_layers"], p["dropout"], p["train"], p["bidirectional"], p["batch_first"]


@pytest.mark.lstm
def test_lstm():
    LSTMBenchmark(op_name="lstm", torch_op=torch.lstm, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
