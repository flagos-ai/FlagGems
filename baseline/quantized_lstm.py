# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA native baseline for `quantized_lstm`.

Mirrors benchmark/test_quantized_lstm.py (QuantizedLSTMBenchmark, a
base.Benchmark subclass):
    op_name  = "quantized_lstm"
    torch_op = torch_quantized_lstm (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with QUANTIZED_LSTM_SHAPES -- read as
(seq_len, batch_size, input_size, hidden_size) -- so core_shapes.yaml is not
consulted.

CAUTION: `aten::quantized_lstm` has no CUDA kernel. The benchmark's baseline is
torch._VF.lstm (cuDNN) over the *dequantized* copies of the same int8 weights,
which is the device-side reference with matching arithmetic; comparing against
the CPU quantized operator would measure the host/device gap instead. That
wrapper is reproduced here, and so is `_build_cell_params`, which prepacks the
int8 weights on the CPU (the prepacked handle is CPU-only) and returns the
dequantized copies for the device-side call. The prepacked `cell_params` is
built and passed through so the argument list matches, but the timed reference
only reads `deq`. See Q&A.md.

The benchmark sets torch.backends.quantized.engine = "fbgemm" before building
inputs; the same is done here.
"""

import torch

OP_NAME = "quantized_lstm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch._VF.lstm is unavailable on this torch build"

# test_quantized_lstm.py::QUANTIZED_LSTM_SHAPES
# (seq_len, batch_size, input_size, hidden_size)
_SHAPES = [
    (4, 32, 256, 256),
    (4, 64, 256, 256),
    (4, 64, 512, 512),
    (4, 128, 256, 256),
    (8, 64, 256, 256),
]

_HAS_BIASES = True
_NUM_LAYERS = 1
_DROPOUT = 0.0
_TRAIN = False
_BIDIRECTIONAL = False
_BATCH_FIRST = False


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _build_cell_params(input_size, hidden_size, device, dtype):
    """benchmark::_build_cell_params -- prepack int8 weights, return dequantized.

    The prepacked CellParamsBase handle is CPU-only, so it is built on the host
    and the dequantized weights are returned for the device-side reference.
    """
    w_ih = torch.randn(4 * hidden_size, input_size) * 0.3
    w_hh = torch.randn(4 * hidden_size, hidden_size) * 0.3
    b_ih = torch.randn(4 * hidden_size) * 0.1
    b_hh = torch.randn(4 * hidden_size) * 0.1

    q_ih = torch.quantize_per_tensor(
        w_ih, w_ih.abs().max().item() / 127, 0, torch.qint8
    )
    q_hh = torch.quantize_per_tensor(
        w_hh, w_hh.abs().max().item() / 127, 0, torch.qint8
    )
    packed_ih = torch.ops.quantized.linear_prepack(q_ih, b_ih)
    packed_hh = torch.ops.quantized.linear_prepack(q_hh, b_hh)
    cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
        packed_ih, packed_hh, b_ih, b_hh, True
    )

    deq_ih = torch.ops.quantized.linear_unpack(packed_ih)[0].dequantize()
    deq_hh = torch.ops.quantized.linear_unpack(packed_hh)[0].dequantize()

    return cell_params, [
        deq_ih.to(device=device, dtype=dtype),
        deq_hh.to(device=device, dtype=dtype),
        b_ih.to(device=device, dtype=dtype),
        b_hh.to(device=device, dtype=dtype),
    ]


def _torch_quantized_lstm(inp, hx, params, deq, *args):
    """benchmark::torch_quantized_lstm -- cuDNN LSTM over dequantized weights."""
    return torch._VF.lstm(
        inp,
        hx,
        deq,
        _HAS_BIASES,
        _NUM_LAYERS,
        _DROPOUT,
        _TRAIN,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
    )


def native():
    if _resolve("torch._VF.lstm") is None:
        return None
    return _torch_quantized_lstm


def grid():
    return [
        {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "input_size": input_size,
            "hidden_size": hidden_size,
        }
        for seq_len, batch_size, input_size, hidden_size in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    seq_len = binding["seq_len"]
    batch_size = binding["batch_size"]
    input_size = binding["input_size"]
    hidden_size = binding["hidden_size"]
    torch.backends.quantized.engine = "fbgemm"
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    h0 = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=device)
    c0 = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=device)
    cell_params, deq = _build_cell_params(input_size, hidden_size, device, dtype)
    return (
        inp,
        [h0, c0],
        [cell_params],
        deq,
        _HAS_BIASES,
        _NUM_LAYERS,
        _DROPOUT,
        _TRAIN,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
    ), {}


def key_shape(binding):
    return [
        binding["seq_len"],
        binding["batch_size"],
        binding["input_size"],
        binding["hidden_size"],
    ]


def config(binding, dtype):
    seq_len = binding["seq_len"]
    batch_size = binding["batch_size"]
    input_size = binding["input_size"]
    hidden_size = binding["hidden_size"]
    dt = str(dtype)
    gate = 4 * hidden_size
    return {
        "inputs": {
            "args": [
                {"shape": [seq_len, batch_size, input_size], "dtype": dt},
                [
                    {"shape": [1, batch_size, hidden_size], "dtype": dt},
                    {"shape": [1, batch_size, hidden_size], "dtype": dt},
                ],
                [{"note": "prepacked quantized cell params (CPU handle)"}],
                [
                    {"shape": [gate, input_size], "dtype": dt},
                    {"shape": [gate, hidden_size], "dtype": dt},
                    {"shape": [gate], "dtype": dt},
                    {"shape": [gate], "dtype": dt},
                ],
                {"scalar": _HAS_BIASES},
                {"scalar": _NUM_LAYERS},
                {"scalar": _DROPOUT},
                {"scalar": _TRAIN},
                {"scalar": _BIDIRECTIONAL},
                {"scalar": _BATCH_FIRST},
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {"shape": [seq_len, batch_size, hidden_size], "dtype": dt},
            "h_n": {"shape": [1, batch_size, hidden_size], "dtype": dt},
            "c_n": {"shape": [1, batch_size, hidden_size], "dtype": dt},
        },
        "dims": {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "input_size": input_size,
            "hidden_size": hidden_size,
        },
    }
