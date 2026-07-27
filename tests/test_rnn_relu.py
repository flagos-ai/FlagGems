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

RNN_HIDDEN_SIZES = [8, 16]

pytestmark = pytest.mark.rnn_relu


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.rnn_relu
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", RNN_HIDDEN_SIZES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rnn_relu(seq_len, batch_size, input_size, hidden_size, dtype, batch_first):
    """Test rnn_relu accuracy against PyTorch implementation"""
    if batch_first:
        input_tensor = torch.randn(
            batch_size, seq_len, input_size, dtype=dtype, device=flag_gems.device
        )
    else:
        input_tensor = torch.randn(
            seq_len, batch_size, input_size, dtype=dtype, device=flag_gems.device
        )

    # Create RNN model and get params
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu")
    rnn = rnn.to(dtype=dtype, device=flag_gems.device)
    params = tuple(rnn._flat_weights)
    hx = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor)
    ref_hx = utils.to_reference(hx)
    ref_params = tuple(utils.to_reference(p) for p in params)

    # Run PyTorch reference
    ref_out = torch.rnn_relu(
        ref_input, ref_hx, ref_params, True, 1, 0.0, False, False, batch_first
    )

    # Run FlagGems implementation (via torch dispatch)
    with flag_gems.use_gems():
        res_out = torch.rnn_relu(
            input_tensor, hx, params, True, 1, 0.0, False, False, batch_first
        )

    # Compare outputs
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    utils.gems_assert_close(res_out[1], ref_out[1], dtype)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.rnn_relu
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("input_size", [8, 16])
@pytest.mark.parametrize("hidden_size", RNN_HIDDEN_SIZES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rnn_relu_direct_wrapper(
    seq_len, batch_size, input_size, hidden_size, dtype, batch_first
):
    """Direct wrapper smoke test: call flag_gems.ops.rnn_relu.rnn_relu directly
    and compare against PyTorch's native rnn_relu (float64 reference)."""
    from flag_gems.ops.rnn_relu import rnn_relu as gems_rnn_relu

    if batch_first:
        input_tensor = torch.randn(
            batch_size, seq_len, input_size, dtype=dtype, device=flag_gems.device
        )
    else:
        input_tensor = torch.randn(
            seq_len, batch_size, input_size, dtype=dtype, device=flag_gems.device
        )

    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu")
    rnn = rnn.to(dtype=dtype, device=flag_gems.device)
    params = tuple(rnn._flat_weights)
    hx = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    # Run direct wrapper call
    out, hidden = gems_rnn_relu(
        input_tensor, hx, params, True, 1, 0.0, False, False, batch_first
    )

    # Run PyTorch reference in float64
    ref_input = utils.to_reference(input_tensor)
    ref_hx = utils.to_reference(hx)
    ref_params = tuple(utils.to_reference(p) for p in params)
    ref_out = torch.rnn_relu(
        ref_input, ref_hx, ref_params, True, 1, 0.0, False, False, batch_first
    )

    # The Triton kernel uses tiled mat-vec with float32 accumulation while
    # PyTorch's native rnn_relu uses cuDNN fused kernels.  The algorithmic
    # difference causes small numerical discrepancies (worst-case over all
    # configs: float32≈1e-3, float16≈2e-3, bfloat16≈1.6e-2).  We use a
    # 2× safety margin for atol.
    atol = {torch.float32: 2e-3, torch.float16: 5e-3, torch.bfloat16: 3e-2}[dtype]
    utils.gems_assert_close(out, ref_out[0], dtype, atol=atol)
    utils.gems_assert_close(hidden, ref_out[1], dtype, atol=atol)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.rnn_relu
def test_rnn_relu_direct_backward():
    """Direct wrapper backward: compare gradients against native PyTorch recomputation."""
    from flag_gems.ops.rnn_relu import _params_unpack
    from flag_gems.ops.rnn_relu import rnn_relu as gems_rnn_relu

    seq, batch, input_size, hidden_size = 4, 2, 8, 8
    dtype = torch.float32

    inp_data = torch.randn(seq, batch, input_size, device=flag_gems.device, dtype=dtype)
    hx_data = torch.randn(1, batch, hidden_size, device=flag_gems.device, dtype=dtype)
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu").to(
        device=flag_gems.device, dtype=dtype
    )
    rnn.flatten_parameters()
    params_data = tuple(p.detach().clone() for p in rnn._flat_weights)

    # Side 1: FlagGems (through custom autograd)
    inp_g = inp_data.detach().clone().requires_grad_(True)
    hx_g = hx_data.detach().clone().requires_grad_(True)
    params_g = tuple(p.detach().clone().requires_grad_(True) for p in params_data)
    out_g, hid_g = gems_rnn_relu(
        inp_g, hx_g, params_g, True, 1, 0.0, True, False, False
    )
    (out_g.sum() + hid_g.sum()).backward()

    # Side 2: Native PyTorch recompute (same as backward internals)
    inp_r = inp_data.detach().clone().requires_grad_(True)
    hx_r = hx_data.detach().clone().requires_grad_(True)
    params_r = tuple(p.detach().clone().requires_grad_(True) for p in params_data)
    weight_ih, weight_hh, bias_ih, bias_hh = _params_unpack(params_r, True)
    h = hx_r[0].clone()
    outputs = []
    for t_idx in range(seq):
        xt = inp_r[t_idx, :, :]
        pre_act = torch.addmm(bias_ih, xt, weight_ih.t())
        pre_act += torch.addmm(bias_hh, h, weight_hh.t())
        h = torch.relu(pre_act)
        outputs.append(h)
    out_r = torch.stack(outputs, dim=0)
    hid_r = h.unsqueeze(0)
    (out_r.sum() + hid_r.sum()).backward()

    utils.gems_assert_close(inp_g.grad, inp_r.grad, dtype)
    utils.gems_assert_close(hx_g.grad, hx_r.grad, dtype)
    for pg, pr in zip(params_g, params_r):
        utils.gems_assert_close(pg.grad, pr.grad, dtype)


@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)
@pytest.mark.rnn_relu
@pytest.mark.parametrize("hidden_size", [128, 256, 512])
def test_rnn_relu_large_hidden(hidden_size):
    """Regression: hidden_size previously had chunk-level read-after-write bug."""
    from flag_gems.ops.rnn_relu import rnn_relu as gems_rnn_relu

    seq, batch, input_size = 3, 1, 16
    dtype = torch.float32

    inp = torch.randn(seq, batch, input_size, device=flag_gems.device, dtype=dtype)
    hx = torch.randn(1, batch, hidden_size, device=flag_gems.device, dtype=dtype)
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu").to(
        device=flag_gems.device, dtype=dtype
    )
    params = tuple(p.detach() for p in rnn._flat_weights)

    ref = torch.rnn_relu(inp, hx, params, True, 1, 0.0, False, False, False)
    out_gems = gems_rnn_relu(inp, hx, params, True, 1, 0.0, False, False, False)

    atol = 2e-3
    utils.gems_assert_close(out_gems[0], ref[0], dtype, atol=atol)
    utils.gems_assert_close(out_gems[1], ref[1], dtype, atol=atol)
