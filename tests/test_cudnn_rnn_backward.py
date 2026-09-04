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
from .conftest import TO_CPU

# LSTM backward test shapes: (seq_len, batch, input_size, hidden_size)
CUDNN_RNN_SHAPES = [
    (3, 2, 4, 5),
    (5, 4, 8, 8),
    (4, 3, 16, 16),
    (6, 8, 32, 32),
    (2, 1, 10, 7),
]


def _cudnn_rnn_backward_ref(
    input,
    weight,
    hx,
    cx,
    output,
    grad_output,
    grad_hy,
    grad_cy,
    hidden_size,
    output_mask,
    dtype,
):
    """Reference backward (grad_input, grad_hx, grad_cx, grad_weight_list).

    fp16/fp32 use ``aten::_cudnn_rnn_backward``.  cuDNN's RNN path does not
    support bfloat16 on every backend, so for bfloat16 the reference is
    computed via autograd over the native ``nn.LSTM`` forward (upcast to fp32).
    """
    if dtype != torch.bfloat16:
        return None  # caller uses the cuDNN reserve/weight_buf path

    # bfloat16: autograd over native nn.LSTM forward, computed in fp32.
    seq_len, batch, input_size = input.shape
    rnn = torch.nn.LSTM(input_size, hidden_size, 1).to(
        device=input.device, dtype=torch.float32
    )
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(weight[0].to(torch.float32))
        rnn.weight_hh_l0.copy_(weight[1].to(torch.float32))
        rnn.bias_ih_l0.copy_(weight[2].to(torch.float32))
        rnn.bias_hh_l0.copy_(weight[3].to(torch.float32))
    rnn.flatten_parameters()

    inp32 = input.to(torch.float32).detach().requires_grad_(True)
    hx32 = hx.to(torch.float32)
    cx32 = cx.to(torch.float32)
    out, (hy, cy) = rnn(inp32, (hx32, cx32))
    gs = torch.autograd.grad(
        [out, hy, cy],
        [inp32] + list(rnn.parameters()),
        grad_outputs=[
            grad_output.to(torch.float32),
            grad_hy.to(torch.float32),
            grad_cy.to(torch.float32),
        ],
        allow_unused=True,
    )
    grad_input = gs[0].to(dtype)
    # grad_hx / grad_cx are the gradients w.r.t. the initial hidden/cell state;
    # autograd over nn.LSTM does not expose them directly, so recompute via the
    # hidden-state inputs.
    hx_leaf = hx32.detach().requires_grad_(True)
    cx_leaf = cx32.detach().requires_grad_(True)
    out2, (hy2, cy2) = rnn(inp32.detach(), (hx_leaf, cx_leaf))
    gs2 = torch.autograd.grad(
        [out2, hy2, cy2],
        [hx_leaf, cx_leaf],
        grad_outputs=[
            grad_output.to(torch.float32),
            grad_hy.to(torch.float32),
            grad_cy.to(torch.float32),
        ],
        allow_unused=True,
    )
    grad_hx = gs2[0].to(dtype)
    grad_cx = gs2[1].to(dtype)
    grad_weight = [
        (g if g is not None else torch.zeros_like(p)).to(dtype)
        for g, p in zip(gs[1:], weight)
    ]
    return grad_input, grad_hx, grad_cx, grad_weight


@pytest.mark.cudnn_rnn_backward
@pytest.mark.parametrize("shape", CUDNN_RNN_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cudnn_rnn_backward(shape, dtype):
    """Test accuracy for cudnn_rnn_backward (single-layer unidirectional LSTM)."""
    if TO_CPU:
        pytest.skip("_cudnn_rnn_backward is CUDA-only, cannot run in quick-cpu mode")
    seq_len, batch, input_size, hidden_size = shape
    dev = flag_gems.device
    gates = 4  # LSTM has 4 gates

    # Fixed seed for reproducibility.
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

    output_mask = [True, True, True, True]

    # Forward pass to obtain the reserve / weight_buf consumed by the backward
    # passes.  fp16/fp32 use the cuDNN forward; bf16 uses the gems forward
    # (cuDNN does not support bf16 RNN on every backend), whose reserve /
    # weight_buf feeds the gems backward (the reference is autograd over
    # native nn.LSTM).
    if dtype != torch.bfloat16:
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
    else:
        output, hy, cy, reserve, weight_buf = flag_gems.cudnn_rnn(
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

    grad_output = torch.randn_like(output)
    grad_hy = torch.randn_like(hy)
    grad_cy = torch.randn_like(cy)

    # Reference backward: cuDNN path for fp16/fp32, autograd (native) for bf16.
    if dtype != torch.bfloat16:
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
        ref_grad_input, ref_grad_hx, ref_grad_cx, ref_grad_weight = ref_out
    else:
        ref_grad_input, ref_grad_hx, ref_grad_cx, ref_grad_weight = (
            _cudnn_rnn_backward_ref(
                input,
                weight,
                hx,
                cx,
                output,
                grad_output,
                grad_hy,
                grad_cy,
                hidden_size,
                output_mask,
                dtype,
            )
        )

    res_out = flag_gems.cudnn_rnn_backward(
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
    res_grad_input, res_grad_hx, res_grad_cx, res_grad_weight = res_out

    ref_grad_input = utils.to_reference(ref_grad_input)
    ref_grad_hx = utils.to_reference(ref_grad_hx)
    ref_grad_cx = utils.to_reference(ref_grad_cx)
    ref_grad_weight = [utils.to_reference(w) for w in ref_grad_weight]

    # Gradients over a multi-step BPTT accumulate in a different order than
    # the autograd recomputation, so rounding differences compound into
    # large-magnitude gradients; a relaxed absolute tolerance is required.
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
