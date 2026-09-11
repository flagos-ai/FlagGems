from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

# mkldnn_rnn_layer is LSTM (mode=2), single-layer, unidirectional.
_MODE = 2
_NUM_LAYERS = 1
_BIDIRECTIONAL = False
_BATCH_FIRST = False


class MkldnnRnnLayerBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (16, 4, 32),
            (32, 8, 64),
            (64, 16, 128),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def mkldnn_rnn_layer_input_fn(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    lstm = torch.nn.LSTM(input_size, hidden_size, 1).to(dtype=dtype, device=device)
    w_ih = lstm.weight_ih_l0
    w_hh = lstm.weight_hh_l0
    b_ih = lstm.bias_ih_l0
    b_hh = lstm.bias_hh_l0
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    # All arguments are positional in the aten schema:
    # (input, w0, w1, w2, w3, hx, cx, reverse, batch_sizes, mode, hidden_size,
    #  num_layers, has_biases, bidirectional, batch_first, train)
    yield (
        inp,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hx,
        cx,
        False,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        False,
    )


def _torch_lstm_layer_baseline(inp, w_ih, w_hh, b_ih, b_hh, hx, cx, *rest):
    """Same-device LSTM baseline.

    The aten mkldnn_rnn_layer (oneDNN) is CPU-only, so it cannot serve as an
    on-device torch baseline. Reproduce the identical single-layer LSTM with
    native PyTorch on the input device instead (gate order i, f, g, o)."""
    reverse = rest[0]
    seq_len = inp.shape[0]
    h = hx
    c = cx
    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = [None] * seq_len
    for t in steps:
        gates = torch.addmm(b_ih, inp[t], w_ih.t()) + torch.addmm(b_hh, h, w_hh.t())
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs[t] = h
    return torch.stack(outputs, dim=0), h, c


@pytest.mark.mkldnn_rnn_layer
def test_mkldnn_rnn_layer():
    bench = MkldnnRnnLayerBenchmark(
        input_fn=mkldnn_rnn_layer_input_fn,
        op_name="mkldnn_rnn_layer",
        torch_op=_torch_lstm_layer_baseline,
        gems_op=flag_gems.ops.mkldnn_rnn_layer,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
