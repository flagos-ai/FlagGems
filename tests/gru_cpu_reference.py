# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GRU CPU oracle for the Ascend tests, independently checked against H800 CUDA.

The equations and all five workspace planes were checked on the original
accuracy/core recipes and arbitrary workspaces. This is a reference/bootstrap,
not an optimized target implementation. The existing Gems out preconditions
(shared forward output storage and backward has_bias=False) remain stricter
than, and independent of, CUDA's overlap behavior.
"""

import torch

_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _tensor(t, shape, dtype, device, name):
    if not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(shape):
        raise RuntimeError(f"{name} must have shape {tuple(shape)}")
    if t.dtype != dtype or t.device != device:
        raise RuntimeError(f"{name} must have dtype {dtype} on {device}")


def _forward_inputs(ig, hg, hx, ib, hb):
    if ig.ndim != 2 or ig.dtype not in _FLOATS or ig.shape[1] % 3:
        raise RuntimeError(
            "GRU input_gates requires two-dimensional floating gates divisible by3"
        )
    b, gates = ig.shape
    h = gates // 3
    _tensor(hg, (b, gates), ig.dtype, ig.device, "hidden_gates")
    _tensor(hx, (b, h), ig.dtype, ig.device, "hx")
    if ib is not None:
        _tensor(ib, (gates,), ig.dtype, ig.device, "input_bias")
        if hb is None:
            raise RuntimeError("hidden_bias must be defined when input_bias is defined")
    if hb is not None:
        _tensor(hb, (gates,), ig.dtype, ig.device, "hidden_bias")
    return b, h


def _backward_inputs(grad, ws):
    if grad.ndim != 2 or grad.dtype not in _FLOATS:
        raise RuntimeError("GRU grad_hy requires a two-dimensional floating tensor")
    b, h = grad.shape
    _tensor(ws, (b, 5 * h), grad.dtype, grad.device, "workspace")
    return b, h


def _cpu(x):
    return x.cpu() if isinstance(x, torch.Tensor) else x


def _on_device(values, device):
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in values)


def _forward_math(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    if input_bias is not None and hidden_bias is None:
        raise RuntimeError("hidden_bias must be defined when input_bias is defined")
    dtype = input_gates.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    input_reset, input_update, input_new = (
        value.to(compute_dtype) for value in input_gates.chunk(3, dim=1)
    )
    hidden_reset, hidden_update, hidden_new = (
        value.to(compute_dtype) for value in hidden_gates.chunk(3, dim=1)
    )
    if input_bias is not None:
        bias_reset, bias_update, bias_new = (
            value.to(compute_dtype) for value in input_bias.chunk(3)
        )
        hidden_bias_reset, hidden_bias_update, hidden_bias_new = (
            value.to(compute_dtype) for value in hidden_bias.chunk(3)
        )
        input_reset = input_reset + bias_reset
        input_update = input_update + bias_update
        input_new = input_new + bias_new
        hidden_reset = hidden_reset + hidden_bias_reset
        hidden_update = hidden_update + hidden_bias_update
        hidden_new = hidden_new + hidden_bias_new
    reset_gate = torch.sigmoid(input_reset + hidden_reset)
    update_gate = torch.sigmoid(input_update + hidden_update)
    new_gate = torch.tanh(input_new + reset_gate * hidden_new)
    hx_compute = hx.to(compute_dtype)
    hy = new_gate + update_gate * (hx_compute - new_gate)
    workspace = torch.cat(
        (reset_gate, update_gate, new_gate, hx_compute, hidden_new), dim=1
    )
    return hy.to(dtype), workspace.to(dtype)


def _backward_math(grad_hy, workspace, has_bias):
    dtype = grad_hy.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    grad_hy_compute = grad_hy.to(compute_dtype)
    reset_gate, input_gate, new_gate, hx, hidden_new = (
        value.to(compute_dtype) for value in workspace.chunk(5, dim=1)
    )
    grad_input_gate = grad_hy_compute * (hx - new_gate)
    grad_input_gate *= 1.0 - input_gate
    grad_input_gate *= input_gate
    grad_hx = grad_hy_compute * input_gate
    grad_new_input = grad_hy_compute * (1.0 - input_gate)
    grad_new_input *= 1.0 - new_gate * new_gate
    grad_hidden_new = grad_new_input * reset_gate
    grad_reset = grad_new_input * hidden_new
    grad_reset *= 1.0 - reset_gate
    grad_reset *= reset_gate
    grad_input_gates = torch.cat(
        (grad_reset, grad_input_gate, grad_new_input), dim=1
    ).to(dtype)
    grad_hidden_gates = torch.cat(
        (grad_reset, grad_input_gate, grad_hidden_new), dim=1
    ).to(dtype)
    grad_hx = grad_hx.to(dtype)
    if not has_bias:
        return grad_input_gates, grad_hidden_gates, grad_hx, None, None
    grad_input_bias = grad_input_gates.to(compute_dtype).sum(dim=0).to(dtype)
    grad_hidden_bias = grad_hidden_gates.to(compute_dtype).sum(dim=0).to(dtype)
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )


class _ForwardWithNativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_gates, hidden_gates, hx, input_bias, hidden_bias):
        hy, workspace = _forward_math(
            input_gates, hidden_gates, hx, input_bias, hidden_bias
        )
        # Native CUDA backward consumes the rounded saved planes, including for
        # Half/BFloat16, rather than differentiating the unrounded intermediates.
        ctx.save_for_backward(workspace)
        ctx.has_bias = input_bias is not None
        ctx.mark_non_differentiable(workspace)
        return hy, workspace

    @staticmethod
    def backward(ctx, grad_hy, grad_workspace):
        (workspace,) = ctx.saved_tensors
        return _backward_math(grad_hy, workspace, ctx.has_bias)


def forward(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    ig, hg, ib, hb = input_gates, hidden_gates, input_bias, hidden_bias
    _forward_inputs(ig, hg, hx, ib, hb)
    values = _ForwardWithNativeBackward.apply(*[_cpu(x) for x in (ig, hg, hx, ib, hb)])
    return _on_device(values, ig.device)


def backward(grad_hy, workspace, has_bias):
    grad = grad_hy
    _backward_inputs(grad, workspace)
    return _on_device(
        _backward_math(grad.cpu(), workspace.cpu(), has_bias), grad.device
    )


def _out_metadata(out, dtype, device):
    if not isinstance(out, torch.Tensor) or out.dtype != dtype or out.device != device:
        raise RuntimeError(f"GRU output requires dtype {dtype} on {device}")


def _copy_outputs(outputs, values, dtype, device):
    # Values are fully computed before resizing/copying; out may alias an input.
    records = [(out, tuple(out.shape), out.stride()) for out in outputs]
    for out, value in zip(outputs, values):
        if tuple(out.shape) != tuple(value.shape):
            out.resize_(value.shape)
        out.copy_(value.to(device=device, dtype=dtype))
    for (out, shape, stride), value in zip(records, values):
        assert out.device == device and out.dtype == dtype and out.shape == value.shape
        if shape == tuple(value.shape):
            assert out.stride() == stride
    return tuple(outputs)


def forward_out(
    input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None, *, out0, out1
):
    ig, hg, ib, hb = input_gates, hidden_gates, input_bias, hidden_bias
    _forward_inputs(ig, hg, hx, ib, hb)
    outputs = (out0, out1)
    for out in outputs:
        _out_metadata(out, ig.dtype, ig.device)
    # Existing Gems condition, checked on the true target storages before CPU.
    # Distinct empty buffers have zero storage and must not compare as aliases.
    if (
        out0.device == out1.device
        and out0.untyped_storage().nbytes() > 0
        and out0.untyped_storage().data_ptr() == out1.untyped_storage().data_ptr()
    ):
        raise RuntimeError("out0 and out1 must not share storage")
    values = _forward_math(*[_cpu(x) for x in (ig, hg, hx, ib, hb)])
    return _copy_outputs(outputs, values, ig.dtype, ig.device)


def backward_out(grad_hy, workspace, has_bias, *, out0, out1, out2, out3, out4):
    grad = grad_hy
    _backward_inputs(grad, workspace)
    if not has_bias:
        raise RuntimeError("the out overload requires has_bias=True")
    outputs = (out0, out1, out2, out3, out4)
    for out in outputs:
        _out_metadata(out, grad.dtype, grad.device)
    values = _backward_math(grad.cpu(), workspace.cpu(), True)
    return _copy_outputs(outputs, values, grad.dtype, grad.device)
