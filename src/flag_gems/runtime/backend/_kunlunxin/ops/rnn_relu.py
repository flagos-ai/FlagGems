import logging

import torch
import triton
import triton.language as tl

from .stack import stack

logger = logging.getLogger(__name__)


def _gems_mm(a, b):
    """Call the selected FlagGems matmul explicitly.

    ``torch.mm`` only resolves to a FlagGems kernel while ``use_gems()`` is
    active, so the same call would run the native kernel on direct paths.
    """
    import flag_gems

    return flag_gems.mm(a, b)


def _gems_relu(x):
    import flag_gems

    return flag_gems.relu(x)


@triton.jit
def _rnn_relu_step_kernel(
    h_ptr,
    w_hh_t_ptr,
    pre_ptr,
    b_hh_ptr,
    h_out_ptr,
    out_step_ptr,
    H_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_B_HH: tl.constexpr,
):
    """One fused RNN-ReLU step for every batch row (grid = batch_size).

    h_new = relu(mm_result + b_hh + pre): the matmul accumulates in fp32 and
    is truncated to the native dtype before b_hh and pre are added, matching
    native torch order and precision. The host passes zero-padded buffers whose
    last dim is H_PAD (next pow2 of hidden_size), so every tile is unmasked.
    """
    native_ty = h_ptr.dtype.element_ty
    b = tl.program_id(0)
    for hb in range(H_PAD // BLOCK_H):
        o_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for kb in range(H_PAD // BLOCK_H):
            k_offs = kb * BLOCK_H + tl.arange(0, BLOCK_H)
            h_vec = tl.load(h_ptr + b * H_PAD + k_offs).to(tl.float32)
            w_tile = tl.load(w_hh_t_ptr + o_offs[:, None] * H_PAD + k_offs[None, :]).to(
                tl.float32
            )
            acc += tl.sum(w_tile * h_vec[None, :], axis=1)
        # truncate matmul result to native dtype (torch.mm semantics)
        acc_native = acc.to(native_ty)
        # hgates = mm_result + b_hh
        if HAS_B_HH:
            b_hh_val = tl.load(b_hh_ptr + o_offs)
            acc_native = acc_native + b_hh_val
        # igates + hgates
        p = tl.load(pre_ptr + b * H_PAD + o_offs)
        combined = acc_native + p
        h_new = tl.where(combined > 0, combined, tl.zeros([BLOCK_H], dtype=native_ty))
        tl.store(h_out_ptr + b * H_PAD + o_offs, h_new)
        tl.store(out_step_ptr + b * H_PAD + o_offs, h_new)


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
    """Single-layer unidirectional Elman RNN with ReLU activation (kunlunxin).

    Inference uses a fused Triton step kernel per time step; training and other
    cases use a per-step matmul/add/relu recurrence with autograd tracking.
    """
    logger.debug("GEMS_KUNLUNXIN RNN_RELU")

    if params is None:
        raise ValueError("params must be provided")
    if hx is None:
        raise ValueError("hx must be provided to match torch.rnn_relu schema")
    if not (num_layers == 1 and not bidirectional and dropout == 0):
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional without dropout"
        )

    w_ih = params[0]
    w_hh = params[1]
    if has_biases:
        b_ih = params[2]
        b_hh = params[3]
    else:
        b_ih = None
        b_hh = None

    x = input.transpose(0, 1).contiguous() if batch_first else input
    seq_len, batch_size, input_size = x.shape
    hidden_size = w_hh.shape[0]
    hx2d = hx.reshape(batch_size, hidden_size)

    x2d = x.reshape(seq_len * batch_size, input_size)

    # only input/hx gradients require the autograd-safe native chain
    need_autograd = (
        train or input.requires_grad or (hx is not None and hx.requires_grad)
    )

    # Fused path is gated to pow2 hidden_size <= 128 and non-bf16; bf16 uses the
    # native-chain path with fp32 recurrence instead (see below).
    is_bf16 = x.dtype == torch.bfloat16
    if (
        not need_autograd
        and not is_bf16
        and hidden_size <= 128
        and ((hidden_size & (hidden_size - 1)) == 0)
    ):
        # ---- fused kernel path (inference-style, pow2 hidden, fp16/fp32) ----
        # One kernel per time step: h_new = relu(h @ W_hh^T + b_hh + pre).
        # BLOCK_H equals hidden_size (pow2), so all tiles are unmasked.
        #
        # For fp16, compute the pre-projection matmul in fp32 then cast back to
        # match native mm output.
        low_prec = x.dtype != torch.float32
        if low_prec:
            pre = _gems_mm(x2d.float(), w_ih.t().float()).to(x.dtype)
        else:
            pre = _gems_mm(x2d, w_ih.t())
        if b_ih is not None:
            pre = pre + b_ih
        pre = pre.reshape(seq_len, batch_size, hidden_size)
        # b_hh is added per step to match native torch accumulation order.
        hp = hidden_size
        blk = hp
        # zero-pad b_hh to hp
        if b_hh is not None:
            b_hh_buf = torch.zeros((hp,), dtype=x.dtype, device=x.device)
            b_hh_buf[:hidden_size] = b_hh
        else:
            b_hh_buf = torch.zeros((hp,), dtype=x.dtype, device=x.device)
        h_buf = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in[:, :hidden_size] = hx2d
        out_buf = torch.zeros((seq_len, batch_size, hp), dtype=x.dtype, device=x.device)
        for t in range(seq_len):
            _rnn_relu_step_kernel[(batch_size,)](
                h_in,
                w_hh,
                pre[t],
                b_hh_buf,
                h_buf,
                out_buf[t],
                H_PAD=hp,
                BLOCK_H=blk,
                HAS_B_HH=b_hh is not None,
            )
            h_in, h_buf = h_buf, h_in
        output = out_buf[..., :hidden_size]
        h = h_in[..., :hidden_size]
    else:
        # ---- per-step chain (autograd-friendly, also bf16 inference) ----
        # For bf16, per-step mm is computed in fp32 then truncated to bf16.
        # Direct flag_gems.<op> calls do not build autograd nodes, so the grad
        # path keeps the differentiable aten ops; the inference path calls the
        # FlagGems kernels explicitly instead of relying on the use_gems()
        # context to redirect torch.mm / torch.relu.
        if need_autograd:
            mm, relu = torch.mm, torch.relu
        else:
            mm, relu = _gems_mm, _gems_relu
        w_hh_t = w_hh.t().contiguous()
        is_bf16 = x.dtype == torch.bfloat16
        try:
            if is_bf16:
                pre = mm(x2d.float(), w_ih.t().float()).to(x.dtype)
            else:
                pre = (
                    (mm(x2d, w_ih.t()) + b_ih)
                    if b_ih is not None
                    else mm(x2d, w_ih.t())
                )
            if is_bf16 and b_ih is not None:
                pre = pre + b_ih
            pre = pre.reshape(seq_len, batch_size, hidden_size)
            # b_hh is added per step to match native torch order
            if is_bf16:
                w_hh_t_f = w_hh_t.float()
            h = hx2d
            outputs = []
            for t in range(seq_len):
                if is_bf16:
                    hgates = mm(h.float(), w_hh_t_f).to(h.dtype)
                else:
                    hgates = mm(h, w_hh_t)
                if b_hh is not None:
                    hgates = hgates + b_hh
                h = relu(hgates + pre[t])
                outputs.append(h)
            # autograd-safe assembly
            output = stack(outputs, 0)
        except ZeroDivisionError:
            # per-step small-shape recurrence, same math
            h = hx2d
            outputs = []
            for t in range(seq_len):
                ih_t = (
                    mm(x[t], w_ih.t()) + b_ih
                    if b_ih is not None
                    else mm(x[t], w_ih.t())
                )
                hh_t = mm(h, w_hh.t()) + b_hh if b_hh is not None else mm(h, w_hh.t())
                h = relu(ih_t.to(torch.float32) + hh_t.to(torch.float32)).to(x.dtype)
                outputs.append(h)
            output = stack(outputs, 0)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h.unsqueeze(0)


__all__ = ["rnn_relu"]


class KunlunxinRnnReluFunction(torch.autograd.Function):
    """Kunlunxin-specific autograd for single-layer unidirectional RNN-ReLU.

    Forward uses the kunlunxin rnn_relu implementation; backward recomputes the
    forward with mm + add (not addmm) and calls torch.autograd.grad.
    """

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
        logger.debug("GEMS_KUNLUNXIN RNN_RELU FUNCTION FORWARD")

        ctx.save_for_backward(input, hx)
        ctx.params = params
        ctx.has_biases = has_biases
        ctx.num_layers = num_layers
        ctx.bidirectional = bidirectional
        ctx.batch_first = batch_first

        # fused kernel or native chain
        return rnn_relu(
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
        logger.debug("GEMS_KUNLUNXIN RNN_RELU FUNCTION BACKWARD")

        input, hx = ctx.saved_tensors
        params = ctx.params
        has_biases = ctx.has_biases
        batch_first = ctx.batch_first

        w_ih = params[0]
        w_hh = params[1]
        if has_biases:
            b_ih = params[2]
            b_hh = params[3]
        else:
            b_ih = None
            b_hh = None

        if batch_first:
            batch_size, seq_len, input_size = input.shape
        else:
            seq_len, batch_size, input_size = input.shape

        # Recompute forward with autograd tracking using mm + add.
        # The aten calls below are deliberate: a direct flag_gems.mm call does
        # not create the grad_fn that torch.autograd.grad needs.
        # w_hh.t() must be inside enable_grad so autograd tracks w_hh.
        with torch.enable_grad():
            w_hh_t = w_hh.t().contiguous()
            h = hx[0].clone()
            outputs = []
            for t_idx in range(seq_len):
                if batch_first:
                    xt = input[:, t_idx, :]
                else:
                    xt = input[t_idx, :, :]
                # igates = mm(xt, w_ih.t()) + b_ih
                pre_act = torch.mm(xt, w_ih.t())
                if has_biases and b_ih is not None:
                    pre_act = pre_act + b_ih
                # hgates = mm(h, w_hh.t()) + b_hh
                hh = torch.mm(h, w_hh_t)
                if has_biases and b_hh is not None:
                    hh = hh + b_hh
                pre_act = pre_act + hh
                h = torch.relu(pre_act)
                outputs.append(h)

            if batch_first:
                output_native = torch.stack(outputs, dim=1)
            else:
                output_native = torch.stack(outputs, dim=0)
            hx_native = h.unsqueeze(0)

            grad_output_flat = grad_output.reshape(output_native.shape)
            grad_hidden_flat = grad_hidden.reshape(hx_native.shape)

            all_weight_grads = torch.autograd.grad(
                outputs=[output_native, hx_native],
                inputs=[input, hx] + list(params),
                grad_outputs=[grad_output_flat, grad_hidden_flat],
                retain_graph=False,
                allow_unused=True,
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
            None,  # params
            None,  # has_biases
            None,  # num_layers
            None,  # dropout
            None,  # train
            None,  # bidirectional
            None,  # batch_first
        )


def _rnn_relu_with_autograd(
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
    """Dispatch wrapper: autograd uses KunlunxinRnnReluFunction, inference uses
    the rnn_relu path.
    """
    need_autograd = (
        train or input.requires_grad or (hx is not None and hx.requires_grad)
    )
    if need_autograd:
        return KunlunxinRnnReluFunction.apply(
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
    return rnn_relu(
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


def _patch_generic_wrapper():
    """Route direct calls to the generic module's ``rnn_relu`` to this override."""
    import sys

    # ``flag_gems.ops.rnn_relu`` is the function, not the module, so look the
    # module up in sys.modules.
    _generic_module = sys.modules["flag_gems.ops.rnn_relu"]
    _generic_module.rnn_relu = _rnn_relu_with_autograd


_patch_generic_wrapper()
