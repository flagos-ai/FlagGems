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

import copy

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

LSTM_SHAPES = (
    [(2, 17)]
    if cfg.QUICK_MODE
    else [
        (1, 4),
        (2, 8),
        (4, 16),
        (8, 32),
        (16, 64),
        (5, 132),
    ]
)

GRAD_MODES = [(True, True), (True, False), (False, True), (False, False)]


def _use_cpu_reference(sample_tensor: torch.Tensor) -> bool:
    return cfg.TO_CPU or sample_tensor.device.type != "cuda"


def _cpu_fused_lstm_cell_forward(
    input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None
):
    dtype = input_gates.dtype
    acc_dtype = torch.float64
    gates = input_gates.detach().cpu().to(acc_dtype) + hidden_gates.detach().cpu().to(
        acc_dtype
    )
    if input_bias is not None:
        gates = gates + input_bias.detach().cpu().to(acc_dtype)
    if hidden_bias is not None:
        gates = gates + hidden_bias.detach().cpu().to(acc_dtype)
    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
    i_gate = torch.sigmoid(i_gate)
    f_gate = torch.sigmoid(f_gate)
    g_gate = torch.tanh(g_gate)
    o_gate = torch.sigmoid(o_gate)
    cy = f_gate * cx.detach().cpu().to(acc_dtype) + i_gate * g_gate
    hy = o_gate * torch.tanh(cy)
    workspace = torch.cat((i_gate, f_gate, g_gate, o_gate), dim=1)
    dev = input_gates.device
    return (
        hy.to(dtype=dtype, device=dev),
        cy.to(dtype=dtype, device=dev),
        workspace.to(dtype=dtype, device=dev),
    )


def _cpu_fused_lstm_cell_backward_impl_ref(
    grad_hy, grad_cy, cx, cy, workspace, has_bias, *, out_device="cpu"
):
    if grad_hy is None and grad_cy is None:
        return None, None, None

    out_dtype = workspace.dtype
    acc_dtype = torch.float64
    batch_size, hidden_size = cx.shape
    ws_shape = workspace.shape
    if batch_size == 0 or hidden_size == 0:
        grad_gates = torch.zeros(ws_shape, dtype=out_dtype, device=out_device)
        grad_cx = torch.zeros(
            (batch_size, hidden_size), dtype=out_dtype, device=out_device
        )
        grad_bias = (
            torch.zeros((ws_shape[1],), dtype=out_dtype, device=out_device)
            if has_bias
            else None
        )
        return grad_gates, grad_cx, grad_bias

    cx_cpu = cx.detach().cpu().to(acc_dtype)
    cy_cpu = cy.detach().cpu().to(acc_dtype)
    ws_flat = (
        workspace.detach().cpu().to(acc_dtype).reshape(batch_size, 4 * hidden_size)
    )
    ghy_cpu = (
        torch.zeros_like(cy_cpu)
        if grad_hy is None
        else grad_hy.detach().cpu().to(acc_dtype)
    )
    gcy_cpu = (
        torch.zeros_like(cy_cpu)
        if grad_cy is None
        else grad_cy.detach().cpu().to(acc_dtype)
    )

    i_gate, f_gate, g_gate, o_gate = ws_flat.split(hidden_size, dim=1)
    tanh_cy = torch.tanh(cy_cpu)
    d_cy = ghy_cpu * o_gate * (1.0 - tanh_cy * tanh_cy) + gcy_cpu
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cx_cpu * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy_cpu * tanh_cy * o_gate * (1.0 - o_gate)
    grad_gates = torch.cat((grad_i, grad_f, grad_g, grad_o), dim=1).reshape(ws_shape)
    grad_cx = d_cy * f_gate
    grad_bias = grad_gates.sum(dim=0) if has_bias else None

    return (
        grad_gates.to(dtype=out_dtype, device=out_device),
        grad_cx.to(dtype=out_dtype, device=out_device),
        None if grad_bias is None else grad_bias.to(dtype=out_dtype, device=out_device),
    )


def _forward_for_test(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    if _use_cpu_reference(cx):
        return _cpu_fused_lstm_cell_forward(
            input_gates, hidden_gates, cx, input_bias, hidden_bias
        )
    return torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )


def _backward_impl_ref(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if _use_cpu_reference(cx):
        target_dev = "cpu" if cfg.TO_CPU else cx.device
        return _cpu_fused_lstm_cell_backward_impl_ref(
            grad_hy, grad_cy, cx, cy, workspace, has_bias, out_device=target_dev
        )
    return torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )


def _assert_impl_outputs(res_out, ref_out, dtype, batch_size):
    assert len(res_out) == len(ref_out) == 3
    atol = {
        torch.float16: 1.5e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 1e-5,
        torch.float64: 1e-10,
    }[dtype]
    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        if ref is None:
            assert res is None, f"Expected None at output[{i}], got {type(res)}"
            continue
        assert res is not None, f"Expected Tensor at output[{i}], got None"
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        assert res.is_contiguous()
        utils.gems_assert_close(
            res, ref, dtype, reduce_dim=max(batch_size, 1), atol=atol
        )


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("has_bias", [True, False])
def test_thnn_fused_lstm_cell_backward_impl(shape, dtype, has_bias):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.bfloat16:
        pytest.skip("Issue #5254: Not supported")
    torch.manual_seed(0)
    batch_size, hidden_size = shape
    dev = flag_gems.device

    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    input_bias = (
        torch.randn(4 * hidden_size, dtype=dtype, device=dev) if has_bias else None
    )
    hidden_bias = (
        torch.randn(4 * hidden_size, dtype=dtype, device=dev) if has_bias else None
    )

    hx, cy, workspace = _forward_for_test(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )
    grad_hy = torch.randn_like(hx)
    grad_cy = torch.randn_like(cy)

    ref_out = _backward_impl_ref(grad_hy, grad_cy, cx, cy, workspace, has_bias)
    res_out = flag_gems._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )
    _assert_impl_outputs(res_out, ref_out, dtype, batch_size)


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("has_grad_hy,has_grad_cy", GRAD_MODES)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_thnn_fused_lstm_cell_backward_impl_optional_and_strided(
    dtype, has_grad_hy, has_grad_cy, has_bias, noncontiguous
):
    """Test optional grad_hy/grad_cy=None and non-contiguous tensor layouts."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.bfloat16:
        pytest.skip("Issue #5254: Not supported")
    torch.manual_seed(17)
    batch_size, hidden_size = 5, 19
    dev = flag_gems.device

    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    bias = torch.randn(4 * hidden_size, dtype=dtype, device=dev) if has_bias else None

    hy, cy, workspace = flag_gems._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, bias, bias
    )
    grad_hy = torch.randn_like(hy) if has_grad_hy else None
    grad_cy = torch.randn_like(cy) if has_grad_cy else None

    if noncontiguous:

        def _noncontig(t):
            return None if t is None else t.T.contiguous().T

        grad_hy = _noncontig(grad_hy)
        grad_cy = _noncontig(grad_cy)
        cx = _noncontig(cx)
        cy = _noncontig(cy)
        workspace = _noncontig(workspace)

    ref_out = _backward_impl_ref(grad_hy, grad_cy, cx, cy, workspace, has_bias)
    res_out = flag_gems._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )
    _assert_impl_outputs(res_out, ref_out, dtype, batch_size)


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "loss_mode",
    ["seq_only", "hn_only", "cn_only", "joint", "unused"],
)
@pytest.mark.parametrize(
    "num_layers,bidirectional,has_bias,batch_first,noncontiguous",
    [
        (1, False, True, True, False),
        (1, False, False, False, True),
        (2, True, True, True, True),
    ],
)
def test_nn_lstm_backward_optional_grads_cpu_parity(
    dtype,
    loss_mode,
    num_layers,
    bidirectional,
    has_bias,
    batch_first,
    noncontiguous,
    monkeypatch,
):
    """End-to-end nn.LSTM forward/backward parity vs independent CPU reference."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.bfloat16:
        pytest.skip("Issue #5254: Not supported")
    torch.manual_seed(123)
    dev = flag_gems.device

    batch_size, seq_len, input_size, hidden_size = 2, 5, 3, 4
    num_directions = 2 if bidirectional else 1

    # Build CPU reference model in float64 (for fp32) or float32 (for fp16/bf16)
    # so CPU reference avoids half-precision CPU op gaps and rounding drift.
    ref_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    base_model = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_bias,
        batch_first=batch_first,
        bidirectional=bidirectional,
    ).to(dtype=dtype)

    cpu_model = copy.deepcopy(base_model).cpu().to(dtype=ref_dtype)
    dev_model = copy.deepcopy(base_model).to(device=dev, dtype=dtype)

    x_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    state_shape = (num_layers * num_directions, batch_size, hidden_size)

    x_base = torch.randn(x_shape, dtype=dtype)
    h0_base = torch.randn(state_shape, dtype=dtype)
    c0_base = torch.randn(state_shape, dtype=dtype)

    x_cpu = x_base.to(dtype=ref_dtype).requires_grad_(True)
    h0_cpu = h0_base.to(dtype=ref_dtype).requires_grad_(True)
    c0_cpu = c0_base.to(dtype=ref_dtype).requires_grad_(True)

    if noncontiguous:
        x_dev_leaf = (
            x_base.to(device=dev, dtype=dtype)
            .permute(2, 0, 1)
            .contiguous()
            .permute(1, 2, 0)
            .detach()
            .requires_grad_(True)
        )
        h0_dev_leaf = (
            h0_base.to(device=dev, dtype=dtype)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
            .detach()
            .requires_grad_(True)
        )
        c0_dev_leaf = (
            c0_base.to(device=dev, dtype=dtype)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
            .detach()
            .requires_grad_(True)
        )
        assert not x_dev_leaf.is_contiguous()
    else:
        x_dev_leaf = x_base.to(device=dev, dtype=dtype).detach().requires_grad_(True)
        h0_dev_leaf = h0_base.to(device=dev, dtype=dtype).detach().requires_grad_(True)
        c0_dev_leaf = c0_base.to(device=dev, dtype=dtype).detach().requires_grad_(True)

    def _compute_loss(x_in, y_out, h_n, c_n):
        if loss_mode == "seq_only":
            return y_out.square().sum()
        if loss_mode == "hn_only":
            return h_n.square().sum()
        if loss_mode == "cn_only":
            return c_n.square().sum()
        if loss_mode == "joint":
            return (
                y_out.square().sum()
                + 0.7 * h_n.square().sum()
                - 0.3 * c_n.square().sum()
            )
        # "unused": LSTM outputs do not participate in loss (both grad_hy/grad_cy None)
        return x_in.square().sum()

    y_cpu, (hn_cpu, cn_cpu) = cpu_model(x_cpu, (h0_cpu, c0_cpu))
    if noncontiguous and loss_mode == "seq_only":
        grad_y_base = torch.randn_like(y_cpu, dtype=dtype)
        grad_y_cpu = grad_y_base.to(dtype=ref_dtype)
        grad_y_dev = (
            grad_y_base.to(device=dev, dtype=dtype)
            .transpose(0, 2)
            .contiguous()
            .transpose(0, 2)
        )
        assert not grad_y_dev.is_contiguous()
        torch.autograd.backward([y_cpu], [grad_y_cpu])
    else:
        grad_y_dev = None
        _compute_loss(x_cpu, y_cpu, hn_cpu, cn_cpu).backward()

    # Exercise FlagGems fused cell ops directly (avoiding aten::lstm forward
    # override so Autograd runs THNNFusedLSTMCellBackward0 ->
    # _thnn_fused_lstm_cell_backward_impl on both CUDA and PrivateUse1/flagos).
    with monkeypatch.context() as registration:
        registration.setattr(flag_gems, "current_work_registrar", None)
        library = torch.library.Library("aten", "IMPL")
        try:
            flag_gems.only_enable(
                lib=library,
                include=[
                    "_thnn_fused_lstm_cell",
                    "_thnn_fused_lstm_cell_backward_impl",
                ],
            )
            # Vendor RNN frontends require contiguous initial hidden states.
            # Retain strided leaves and let autograd track the copies.
            y_dev, (hn_dev, cn_dev) = dev_model(
                x_dev_leaf, (h0_dev_leaf.contiguous(), c0_dev_leaf.contiguous())
            )
            if grad_y_dev is not None:
                torch.autograd.backward([y_dev], [grad_y_dev])
            else:
                _compute_loss(x_dev_leaf, y_dev, hn_dev, cn_dev).backward()
        finally:
            library._destroy()

    atol, rtol = {
        torch.float32: (5e-4, 3e-3),
        torch.float16: (1e-2, 2e-2),
        torch.bfloat16: (4e-2, 5e-2),
    }[dtype]

    torch.testing.assert_close(
        y_dev.detach().cpu().to(ref_dtype), y_cpu.detach(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        hn_dev.detach().cpu().to(ref_dtype), hn_cpu.detach(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        cn_dev.detach().cpu().to(ref_dtype), cn_cpu.detach(), atol=atol, rtol=rtol
    )

    for dev_leaf, cpu_leaf in (
        (x_dev_leaf, x_cpu),
        (h0_dev_leaf, h0_cpu),
        (c0_dev_leaf, c0_cpu),
    ):
        if cpu_leaf.grad is None:
            assert dev_leaf.grad is None
        else:
            assert dev_leaf.grad is not None
            torch.testing.assert_close(
                dev_leaf.grad.detach().cpu().to(ref_dtype),
                cpu_leaf.grad.detach(),
                atol=atol,
                rtol=rtol,
            )

    for (name, p_dev), (_, p_cpu) in zip(
        dev_model.named_parameters(), cpu_model.named_parameters()
    ):
        if p_cpu.grad is None:
            assert p_dev.grad is None, f"Expected None grad for {name}"
        else:
            assert p_dev.grad is not None, f"Missing grad for {name}"
            torch.testing.assert_close(
                p_dev.grad.detach().cpu().to(ref_dtype),
                p_cpu.grad.detach(),
                atol=atol,
                rtol=rtol,
            )
