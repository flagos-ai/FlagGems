# Copyright 2026, The FlagOS Contributors.
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

# LSTM backward test shapes: (seq_len, batch, input_size, hidden_size)
CUDNN_RNN_SHAPES = [
    (3, 2, 4, 5),
    (5, 4, 8, 8),
    (4, 3, 16, 16),
    (6, 8, 32, 32),
    (2, 1, 10, 7),
]


@pytest.mark.cudnn_rnn_backward
@pytest.mark.parametrize("shape", CUDNN_RNN_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_rnn_backward(shape, dtype):
    """Test accuracy for cudnn_rnn_backward (single-layer unidirectional LSTM)."""
    seq_len, batch, input_size, hidden_size = shape
    dev = flag_gems.device
    gates = 4  # LSTM has 4 gates

    # Fixed seed for reproducibility (cuDNN RNN backward is nondeterministic).
    torch.manual_seed(42)

    # Create input tensors (cuDNN RNN is CUDA-only)
    input = torch.randn(seq_len, batch, input_size, dtype=dtype, device=dev)
    hx = torch.randn(1, batch, hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(1, batch, hidden_size, dtype=dtype, device=dev)
    w_ih = torch.randn(gates * hidden_size, input_size, dtype=dtype, device=dev)
    w_hh = torch.randn(gates * hidden_size, hidden_size, dtype=dtype, device=dev)
    b_ih = torch.randn(gates * hidden_size, dtype=dtype, device=dev)
    b_hh = torch.randn(gates * hidden_size, dtype=dtype, device=dev)
    weight = [w_ih, w_hh, b_ih, b_hh]

    # Forward pass to obtain the cuDNN reserve / weight_buf for the reference.
    output, hy, cy, reserve, weight_buf = torch.ops.aten._cudnn_rnn(
        input,
        weight,
        4,
        None,
        hx,
        cx,
        2,
        hidden_size,
        0,
        1,
        False,
        0.0,
        True,
        False,
        [],
        None,
    )

    # Upstream gradients for the backward pass.
    grad_output = torch.randn_like(output)
    grad_hy = torch.randn_like(hy)
    grad_cy = torch.randn_like(cy)

    output_mask = [True, True, True, True]

    # ATen reference (outside use_gems, on CUDA).
    ref_out = torch.ops.aten._cudnn_rnn_backward(
        input,
        weight,
        4,
        weight_buf,
        hx,
        cx,
        output,
        grad_output,
        grad_hy,
        grad_cy,
        2,
        hidden_size,
        0,
        1,
        False,
        0.0,
        True,
        False,
        [],
        None,
        reserve,
        output_mask,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._cudnn_rnn_backward(
            input,
            weight,
            4,
            weight_buf,
            hx,
            cx,
            output,
            grad_output,
            grad_hy,
            grad_cy,
            2,
            hidden_size,
            0,
            1,
            False,
            0.0,
            True,
            False,
            [],
            None,
            reserve,
            output_mask,
        )

    # ref_out order: (grad_input, grad_hx, grad_cx, grad_weight_list)
    ref_grad_input, ref_grad_hx, ref_grad_cx, ref_grad_weight = ref_out
    res_grad_input, res_grad_hx, res_grad_cx, res_grad_weight = res_out

    ref_grad_input = utils.to_reference(ref_grad_input)
    ref_grad_hx = utils.to_reference(ref_grad_hx)
    ref_grad_cx = utils.to_reference(ref_grad_cx)
    ref_grad_weight = [utils.to_reference(w) for w in ref_grad_weight]

    # cuDNN's RNN backward accumulates gradients in a different (and
    # nondeterministic) order than the pure-torch recomputation. Over a
    # multi-step BPTT with unit-variance weights, these rounding differences
    # compound into large-magnitude gradients, so a relaxed absolute tolerance
    # is required (the single-cell _thnn_fused_lstm_cell_backward_impl sibling
    # uses 5e-2; BPTT over several timesteps in bf16 needs a larger bound).
    atol = 2e-1

    for name, ref, res in [
        ("grad_input", ref_grad_input, res_grad_input),
        ("grad_hx", ref_grad_hx, res_grad_hx),
        ("grad_cx", ref_grad_cx, res_grad_cx),
    ]:
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at {name}: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at {name}: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res, ref, dtype, atol=atol)

    assert len(res_grad_weight) == len(ref_grad_weight)
    for i, (ref, res) in enumerate(zip(ref_grad_weight, res_grad_weight)):
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at grad_weight[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at grad_weight[{i}]: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res, ref, dtype, atol=atol)
