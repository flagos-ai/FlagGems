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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

MKLDNN_RNN_HIDDEN_SIZES = [8, 16]

# mkldnn_rnn_layer is LSTM (mode=2), single-layer, unidirectional.
_MODE = 2
_NUM_LAYERS = 1
_BIDIRECTIONAL = False
_BATCH_FIRST = False

pytestmark = pytest.mark.mkldnn_rnn_layer


def _make_lstm_params(input_size, hidden_size, dtype, device):
    """Build packed (4H, *) LSTM weights/biases via torch.nn.LSTM."""
    lstm = torch.nn.LSTM(input_size, hidden_size, 1)
    lstm = lstm.to(dtype=dtype, device=device)
    w_ih = lstm.weight_ih_l0
    w_hh = lstm.weight_hh_l0
    b_ih = lstm.bias_ih_l0
    b_hh = lstm.bias_hh_l0
    return w_ih, w_hh, b_ih, b_hh


def _reference_mkldnn_rnn_layer(
    input_tensor, w_ih, w_hh, b_ih, b_hh, hx, cx, hidden_size, reverse
):
    """Analytical single-layer LSTM reference (mode=2).

    The aten mkldnn_rnn_layer is a plain LSTM; its oneDNN backend is CPU-only
    and only supports fp32 (fp16/bf16 raise "could not create a primitive
    descriptor" on some builds), so we reproduce the exact computation in pure
    PyTorch instead. Weights are packed (4H, *) in gate order i, f, g, o; the
    accumulation is done in fp32 to match the on-device kernel, then cast back.
    """
    dev = input_tensor.device
    dtype = input_tensor.dtype
    x = input_tensor.float()
    wih = w_ih.float()
    whh = w_hh.float()
    bih = b_ih.float()
    bhh = b_hh.float()
    h = hx.float()
    c = cx.float()
    seq_len = x.shape[0]
    time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = [None] * seq_len
    for t in time_steps:
        gates = torch.addmm(bih, x[t], wih.t()) + torch.addmm(bhh, h, whh.t())
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs[t] = h
    output = torch.stack(outputs, dim=0)
    return (
        output.to(device=dev, dtype=dtype),
        h.to(device=dev, dtype=dtype),
        c.to(device=dev, dtype=dtype),
    )


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.mkldnn_rnn_layer
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", MKLDNN_RNN_HIDDEN_SIZES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mkldnn_rnn_layer(seq_len, batch_size, input_size, hidden_size, dtype, reverse):
    """Test mkldnn_rnn_layer accuracy against the oneDNN reference (dispatch)."""
    input_tensor = torch.randn(
        seq_len, batch_size, input_size, dtype=dtype, device=flag_gems.device
    )
    w_ih, w_hh, b_ih, b_hh = _make_lstm_params(
        input_size, hidden_size, dtype, flag_gems.device
    )
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    # Route reference inputs through to_reference (oneDNN LSTM is CPU-only and
    # supports fp16/bf16/fp32 but not fp64, so keep the original dtype).
    ref_input = utils.to_reference(input_tensor)
    ref_w_ih = utils.to_reference(w_ih)
    ref_w_hh = utils.to_reference(w_hh)
    ref_b_ih = utils.to_reference(b_ih)
    ref_b_hh = utils.to_reference(b_hh)
    ref_hx = utils.to_reference(hx)
    ref_cx = utils.to_reference(cx)
    ref_out, ref_hy, ref_cy = _reference_mkldnn_rnn_layer(
        ref_input,
        ref_w_ih,
        ref_w_hh,
        ref_b_ih,
        ref_b_hh,
        ref_hx,
        ref_cx,
        hidden_size,
        reverse,
    )

    with flag_gems.use_gems():
        res_out, res_hy, res_cy, _ws = torch.mkldnn_rnn_layer(
            input_tensor,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hx,
            cx,
            reverse,
            [],
            _MODE,
            hidden_size,
            _NUM_LAYERS,
            True,
            _BIDIRECTIONAL,
            _BATCH_FIRST,
            False,
        )

    # fp16 LSTM accumulation over a sequence disagrees with oneDNN at the
    # ~1-ULP level; use FlagGems' RNN-level fp16 tolerance (5e-3).
    atol = {torch.float32: 2e-3, torch.float16: 5e-3, torch.bfloat16: 3e-2}[dtype]
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.mkldnn_rnn_layer
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", MKLDNN_RNN_HIDDEN_SIZES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mkldnn_rnn_layer_direct_wrapper(
    seq_len, batch_size, input_size, hidden_size, dtype, reverse
):
    """Direct wrapper smoke test: call the gems wrapper directly and compare
    against the oneDNN reference."""
    from flag_gems.ops.mkldnn_rnn_layer import mkldnn_rnn_layer as gems_op

    input_tensor = torch.randn(
        seq_len, batch_size, input_size, dtype=dtype, device=flag_gems.device
    )
    w_ih, w_hh, b_ih, b_hh = _make_lstm_params(
        input_size, hidden_size, dtype, flag_gems.device
    )
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    out, hy, cy, _ws = gems_op(
        input_tensor,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hx,
        cx,
        reverse,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        False,
    )

    ref_out, ref_hy, ref_cy = _reference_mkldnn_rnn_layer(
        utils.to_reference(input_tensor),
        utils.to_reference(w_ih),
        utils.to_reference(w_hh),
        utils.to_reference(b_ih),
        utils.to_reference(b_hh),
        utils.to_reference(hx),
        utils.to_reference(cx),
        hidden_size,
        reverse,
    )

    atol = {torch.float32: 2e-3, torch.float16: 5e-3, torch.bfloat16: 3e-2}[dtype]
    utils.gems_assert_close(out, ref_out, dtype, atol=atol)
    utils.gems_assert_close(hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(cy, ref_cy, dtype, atol=atol)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.mkldnn_rnn_layer
def test_mkldnn_rnn_layer_direct_backward():
    """Direct wrapper backward: compare gradients against a native PyTorch
    LSTM recomputation (matching the backward internals)."""
    from flag_gems.ops.mkldnn_rnn_layer import mkldnn_rnn_layer as gems_op

    seq, batch, input_size, hidden_size = 4, 2, 8, 8
    dtype = torch.float32

    inp_data = torch.randn(seq, batch, input_size, device=flag_gems.device, dtype=dtype)
    hx_data = torch.randn(batch, hidden_size, device=flag_gems.device, dtype=dtype)
    cx_data = torch.randn(batch, hidden_size, device=flag_gems.device, dtype=dtype)
    w_ih, w_hh, b_ih, b_hh = _make_lstm_params(
        input_size, hidden_size, dtype, flag_gems.device
    )
    w_ih_d = w_ih.detach().clone()
    w_hh_d = w_hh.detach().clone()
    b_ih_d = b_ih.detach().clone()
    b_hh_d = b_hh.detach().clone()

    # Side 1: FlagGems (through custom autograd)
    inp_g = inp_data.detach().clone().requires_grad_(True)
    hx_g = hx_data.detach().clone().requires_grad_(True)
    cx_g = cx_data.detach().clone().requires_grad_(True)
    wih_g = w_ih_d.detach().clone().requires_grad_(True)
    whh_g = w_hh_d.detach().clone().requires_grad_(True)
    bih_g = b_ih_d.detach().clone().requires_grad_(True)
    bhh_g = b_hh_d.detach().clone().requires_grad_(True)
    out_g, hy_g, cy_g, _ws = gems_op(
        inp_g,
        wih_g,
        whh_g,
        bih_g,
        bhh_g,
        hx_g,
        cx_g,
        False,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        True,
    )
    (out_g.sum() + hy_g.sum() + cy_g.sum()).backward()

    # Side 2: native PyTorch LSTM recompute (same as backward internals)
    inp_r = inp_data.detach().clone().requires_grad_(True)
    hx_r = hx_data.detach().clone().requires_grad_(True)
    cx_r = cx_data.detach().clone().requires_grad_(True)
    wih_r = w_ih_d.detach().clone().requires_grad_(True)
    whh_r = w_hh_d.detach().clone().requires_grad_(True)
    bih_r = b_ih_d.detach().clone().requires_grad_(True)
    bhh_r = b_hh_d.detach().clone().requires_grad_(True)
    h = hx_r.clone()
    c = cx_r.clone()
    outputs = []
    for t_idx in range(seq):
        xt = inp_r[t_idx]
        gates = torch.addmm(bih_r, xt, wih_r.t()) + torch.addmm(bhh_r, h, whh_r.t())
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs.append(h)
    out_r = torch.stack(outputs, dim=0)
    (out_r.sum() + h.sum() + c.sum()).backward()

    utils.gems_assert_close(inp_g.grad, inp_r.grad, dtype)
    utils.gems_assert_close(hx_g.grad, hx_r.grad, dtype)
    utils.gems_assert_close(cx_g.grad, cx_r.grad, dtype)
    utils.gems_assert_close(wih_g.grad, wih_r.grad, dtype)
    utils.gems_assert_close(whh_g.grad, whh_r.grad, dtype)
    utils.gems_assert_close(bih_g.grad, bih_r.grad, dtype)
    utils.gems_assert_close(bhh_g.grad, bhh_r.grad, dtype)
