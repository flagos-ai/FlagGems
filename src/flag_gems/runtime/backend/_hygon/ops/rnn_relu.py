import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SMALL_HIDDEN_LIMIT = 64
_SMALL_DOT_BATCH_BLOCK = 16


def _params_unpack(params, has_biases):
    """Unpack RNN _flat_weights in PyTorch order."""
    if has_biases:
        weight_ih, weight_hh, bias_ih, bias_hh = params[:4]
        return weight_ih, weight_hh, bias_ih, bias_hh
    return params[0], params[1], None, None


def _small_block_size(size):
    return 32 if size <= 32 else 64


@libentry()
@triton.jit
def rnn_relu_forward_small_bf16_dot_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_output_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_first: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """BF16 fast path for small Elman RNNs.

    One program owns a batch tile and the full hidden vector, so recurrent state
    stays in registers across the whole sequence. Weight matrices are loaded
    once and reused for all time steps.
    """
    b_offs = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    i_offs = tl.arange(0, BLOCK_I)
    h_offs = tl.arange(0, BLOCK_H)

    b_mask = b_offs < batch_size
    i_mask = i_offs < input_size
    h_mask = h_offs < hidden_size

    weight_ih_t = tl.load(
        weight_ih_ptr + i_offs[:, None] + h_offs[None, :] * input_size,
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    )
    weight_hh_t = tl.load(
        weight_hh_ptr + h_offs[:, None] + h_offs[None, :] * hidden_size,
        mask=h_mask[:, None] & h_mask[None, :],
        other=0.0,
    )

    if HAS_BIAS:
        bias = tl.load(bias_ih_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        bias += tl.load(bias_hh_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_H], dtype=tl.float32)

    h = tl.load(
        hx_ptr + b_offs[:, None] * hidden_size + h_offs[None, :],
        mask=b_mask[:, None] & h_mask[None, :],
        other=0.0,
    )

    for t in tl.static_range(seq_len):
        if batch_first:
            x_ptrs = (
                input_ptr
                + b_offs[:, None] * seq_len * input_size
                + t * input_size
                + i_offs[None, :]
            )
            out_ptrs = (
                output_ptr
                + b_offs[:, None] * seq_len * hidden_size
                + t * hidden_size
                + h_offs[None, :]
            )
        else:
            x_ptrs = (
                input_ptr
                + t * batch_size * input_size
                + b_offs[:, None] * input_size
                + i_offs[None, :]
            )
            out_ptrs = (
                output_ptr
                + t * batch_size * hidden_size
                + b_offs[:, None] * hidden_size
                + h_offs[None, :]
            )

        x = tl.load(
            x_ptrs,
            mask=b_mask[:, None] & i_mask[None, :],
            other=0.0,
        )

        acc = tl.dot(x, weight_ih_t, out_dtype=tl.float32)
        acc += tl.dot(h, weight_hh_t, out_dtype=tl.float32)
        acc += bias[None, :]

        h = tl.maximum(acc, 0.0).to(tl.bfloat16)
        tl.store(out_ptrs, h, mask=b_mask[:, None] & h_mask[None, :])

    tl.store(
        hidden_output_ptr + b_offs[:, None] * hidden_size + h_offs[None, :],
        h,
        mask=b_mask[:, None] & h_mask[None, :],
    )


@libentry()
@triton.jit
def rnn_relu_forward_small_scalar_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_output_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_first: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Correct scalar fallback for small non-BF16 inputs."""
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    h_mask = offs < hidden_size
    x_mask = offs < input_size

    if batch_first:
        input_step = input_size
        output_step = hidden_size
        input_base = input_ptr + batch_idx * seq_len * input_size
        output_base = output_ptr + batch_idx * seq_len * hidden_size
    else:
        input_step = batch_size * input_size
        output_step = batch_size * hidden_size
        input_base = input_ptr + batch_idx * input_size
        output_base = output_ptr + batch_idx * hidden_size

    weight_ih = tl.load(
        weight_ih_ptr + offs[:, None] * input_size + offs[None, :],
        mask=h_mask[:, None] & x_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weight_hh = tl.load(
        weight_hh_ptr + offs[:, None] * hidden_size + offs[None, :],
        mask=h_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ih_ptr + offs, mask=h_mask, other=0.0).to(tl.float32)
        bias += tl.load(bias_hh_ptr + offs, mask=h_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    h = tl.load(
        hx_ptr + batch_idx * hidden_size + offs,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)

    for t in tl.static_range(seq_len):
        x = tl.load(
            input_base + t * input_step + offs,
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)

        acc = bias
        acc += tl.sum(weight_ih * x[None, :], axis=1)
        acc += tl.sum(weight_hh * h[None, :], axis=1)
        h = tl.maximum(acc, 0.0)
        tl.store(output_base + t * output_step + offs, h, mask=h_mask)

    tl.store(
        hidden_output_ptr + batch_idx * hidden_size + offs,
        h,
        mask=h_mask,
    )


@libentry()
@triton.jit
def rnn_relu_forward_large_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_output_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_first: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Blocked fallback for larger hidden states."""
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    num_hid_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)

    if batch_first:
        input_step = input_size
        output_step = hidden_size
        input_t_ptr = input_ptr + batch_idx * seq_len * input_size
        output_t_ptr = output_ptr + batch_idx * seq_len * hidden_size
    else:
        input_step = batch_size * input_size
        output_step = batch_size * hidden_size
        input_t_ptr = input_ptr + batch_idx * input_size
        output_t_ptr = output_ptr + batch_idx * hidden_size

    prev_h_ptr = hx_ptr + batch_idx * hidden_size

    for t in tl.static_range(seq_len):
        cur_input_ptr = input_t_ptr + t * input_step
        cur_output_ptr = output_t_ptr + t * output_step

        for hid_block in range(num_hid_blocks):
            h_start = hid_block * BLOCK_SIZE
            h_offs = h_start + offs
            h_mask = h_offs < hidden_size
            h_mask_2d = h_mask[:, None]
            ih_row_offs = h_offs * input_size
            hh_row_offs = h_offs * hidden_size

            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            if HAS_BIAS:
                bias = tl.load(bias_ih_ptr + h_offs, mask=h_mask, other=0.0).to(
                    tl.float32
                )
                bias += tl.load(bias_hh_ptr + h_offs, mask=h_mask, other=0.0).to(
                    tl.float32
                )
                acc += bias

            for inp_start in range(0, input_size, BLOCK_SIZE):
                i_offs = inp_start + offs
                i_mask = i_offs < input_size
                weight_ih = tl.load(
                    weight_ih_ptr + ih_row_offs[:, None] + i_offs[None, :],
                    mask=h_mask_2d & i_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                x_tile = tl.load(
                    cur_input_ptr + i_offs,
                    mask=i_mask,
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(weight_ih * x_tile[None, :], axis=1)

            for hid_in_start in range(0, hidden_size, BLOCK_SIZE):
                j_offs = hid_in_start + offs
                j_mask = j_offs < hidden_size
                weight_hh = tl.load(
                    weight_hh_ptr + hh_row_offs[:, None] + j_offs[None, :],
                    mask=h_mask_2d & j_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                h_tile = tl.load(
                    prev_h_ptr + j_offs,
                    mask=j_mask,
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(weight_hh * h_tile[None, :], axis=1)

            h_new = tl.maximum(acc, 0.0)
            tl.store(cur_output_ptr + h_offs, h_new, mask=h_mask)
            if t == seq_len - 1:
                tl.store(
                    hidden_output_ptr + batch_idx * hidden_size + h_offs,
                    h_new,
                    mask=h_mask,
                )

        prev_h_ptr = cur_output_ptr


rnn_relu_forward_kernel = rnn_relu_forward_large_kernel


def rnn_relu_kernel_forward(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Launch the optimized Triton RNN ReLU kernel after validation."""
    logger.debug("GEMS RNN_RELU FORWARD KERNEL LAUNCH")

    if num_layers > 1 or bidirectional:
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional"
        )
    if dropout > 0 and train:
        raise NotImplementedError(
            "GEMS RNN_RELU does not support dropout in train mode"
        )

    if batch_first:
        batch_size, seq_len, input_size = input.shape
    else:
        seq_len, batch_size, input_size = input.shape

    if hx is not None:
        hidden_size = hx.shape[2]
    elif len(params) > 0:
        hidden_size = params[0].shape[0]
    else:
        raise ValueError("Cannot determine hidden_size")

    num_directions = 2 if bidirectional else 1
    if batch_first:
        output_shape = (batch_size, seq_len, hidden_size * num_directions)
    else:
        output_shape = (seq_len, batch_size, hidden_size * num_directions)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
    hidden = torch.empty(hidden_shape, dtype=input.dtype, device=input.device)

    if hx is None:
        hx = torch.zeros(hidden_shape, dtype=input.dtype, device=input.device)

    weight_ih, weight_hh, bias_ih, bias_hh = _params_unpack(params, has_biases)

    input = input.contiguous()
    output = output.contiguous()
    hidden = hidden.contiguous()
    hx = hx.contiguous()
    weight_ih = weight_ih.contiguous()
    weight_hh = weight_hh.contiguous()
    if bias_ih is not None:
        bias_ih = bias_ih.contiguous()
    if bias_hh is not None:
        bias_hh = bias_hh.contiguous()

    has_bias = bool(has_biases)

    with runtime.torch_device_fn.device(input.device):
        if (
            input.dtype == torch.bfloat16
            and hidden_size <= _SMALL_HIDDEN_LIMIT
            and input_size <= _SMALL_HIDDEN_LIMIT
        ):
            block_i = _small_block_size(input_size)
            block_h = _small_block_size(hidden_size)
            grid = (
                (batch_size + _SMALL_DOT_BATCH_BLOCK - 1) // _SMALL_DOT_BATCH_BLOCK,
            )
            rnn_relu_forward_small_bf16_dot_kernel[grid](
                input,
                hx,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                output,
                hidden,
                seq_len,
                batch_size,
                input_size,
                hidden_size,
                batch_first,
                has_bias,
                BLOCK_B=_SMALL_DOT_BATCH_BLOCK,
                BLOCK_I=block_i,
                BLOCK_H=block_h,
                num_warps=4,
                num_stages=3,
            )
        elif hidden_size <= _SMALL_HIDDEN_LIMIT and input_size <= _SMALL_HIDDEN_LIMIT:
            block_size = max(
                _small_block_size(input_size), _small_block_size(hidden_size)
            )
            grid = (batch_size,)
            rnn_relu_forward_small_scalar_kernel[grid](
                input,
                hx,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                output,
                hidden,
                seq_len,
                batch_size,
                input_size,
                hidden_size,
                batch_first,
                has_bias,
                BLOCK_SIZE=block_size,
                num_warps=1 if block_size <= 32 else 2,
                num_stages=1,
            )
        else:
            grid = (batch_size,)
            block_size = 32 if hidden_size <= 128 else 64
            rnn_relu_forward_large_kernel[grid](
                input,
                hx,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                output,
                hidden,
                seq_len,
                batch_size,
                input_size,
                hidden_size,
                batch_first,
                has_bias,
                BLOCK_SIZE=block_size,
                num_warps=2 if block_size == 32 else 4,
                num_stages=2,
            )

    return output, hidden


class RnnReluFunction(torch.autograd.Function):
    """Autograd wrapper for single-layer unidirectional RNN ReLU."""

    @staticmethod
    def forward(
        ctx,
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        logger.debug("GEMS RNN_RELU FUNCTION FORWARD")

        if num_layers > 1 or bidirectional or (dropout > 0 and train):
            raise NotImplementedError(
                "GEMS RNN_RELU only supports single-layer unidirectional"
            )

        ctx.save_for_backward(input, hx)
        ctx.params = params
        ctx.has_biases = has_biases
        ctx.num_layers = num_layers
        ctx.bidirectional = bidirectional
        ctx.batch_first = batch_first

        return rnn_relu_kernel_forward(
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        logger.debug("GEMS RNN_RELU FUNCTION BACKWARD")

        input, hx = ctx.saved_tensors
        params = ctx.params
        has_biases = ctx.has_biases
        num_layers = ctx.num_layers
        bidirectional = ctx.bidirectional
        batch_first = ctx.batch_first

        if num_layers > 1 or bidirectional:
            raise NotImplementedError(
                "GEMS RNN_RELU backward: unsupported configuration"
            )

        weight_ih, weight_hh, bias_ih, bias_hh = _params_unpack(params, has_biases)

        if batch_first:
            batch_size, seq_len, _ = input.shape
        else:
            seq_len, batch_size, _ = input.shape

        with torch.enable_grad():
            h = hx[0].clone()
            outputs = []
            for t_idx in range(seq_len):
                xt = input[:, t_idx, :] if batch_first else input[t_idx, :, :]
                pre_act = (
                    torch.addmm(bias_ih, xt, weight_ih.t())
                    if has_biases
                    else torch.mm(xt, weight_ih.t())
                )
                pre_act += (
                    torch.addmm(bias_hh, h, weight_hh.t())
                    if has_biases
                    else torch.mm(h, weight_hh.t())
                )
                h = torch.relu(pre_act)
                outputs.append(h)

            output_native = (
                torch.stack(outputs, dim=1)
                if batch_first
                else torch.stack(outputs, dim=0)
            )
            hx_native = h.unsqueeze(0)

            if grad_output is None:
                grad_output = torch.zeros_like(output_native)
            if grad_hidden is None:
                grad_hidden = torch.zeros_like(hx_native)

            all_weight_grads = torch.autograd.grad(
                outputs=[output_native, hx_native],
                inputs=[input, hx] + list(params),
                grad_outputs=[
                    grad_output.reshape(output_native.shape),
                    grad_hidden.reshape(hx_native.shape),
                ],
                retain_graph=False,
            )

        grad_input = all_weight_grads[0]
        grad_hx = all_weight_grads[1]
        grad_params = all_weight_grads[2:]

        for p, g in zip(params, grad_params):
            if g is not None:
                if p.grad is None:
                    p.grad = g.to(p.dtype)
                else:
                    p.grad.add_(g)

        return (
            grad_input,
            grad_hx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rnn_relu(
    input,
    hx=None,
    params=None,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    """Applies a single-layer unidirectional Elman RNN with ReLU."""
    logger.debug("GEMS RNN_RELU")

    if params is None:
        raise ValueError("params must be provided")
    if hx is None:
        raise ValueError("hx must be provided to match torch.rnn_relu schema")

    if num_layers == 1 and not bidirectional and dropout == 0:
        if not train:
            return rnn_relu_kernel_forward(
                input,
                hx,
                params,
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first,
            )

        return RnnReluFunction.apply(
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )

    raise NotImplementedError(
        "GEMS RNN_RELU only supports single-layer unidirectional without dropout"
    )
