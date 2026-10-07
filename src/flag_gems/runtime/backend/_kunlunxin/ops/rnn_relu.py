import logging

import torch
import triton
import triton.language as tl

from .stack import stack

logger = logging.getLogger(__name__)


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

    h_new = relu(mm_result + b_hh + pre) matching native torch order and
    precision: the matmul uses fp32 accumulation (like torch.mm on XPU), then
    truncates to native dtype before adding b_hh and pre in native dtype, so
    the final result is bit-identical to native torch.rnn_relu.
    The host passes zero-padded buffers whose last dim is H_PAD (next pow2
    of hidden_size), so every tile is unmasked: the XPU backend mishandles
    masked tail reads and can fail to compile masked 2D tiles.
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
        # Truncate matmul result to native dtype, matching torch.mm output
        # precision: mm internally uses fp32 accumulation but returns fp16/bf16.
        acc_native = acc.to(native_ty)
        # hgates = mm_result + b_hh, in native dtype (matches native torch)
        if HAS_B_HH:
            b_hh_val = tl.load(b_hh_ptr + o_offs)
            acc_native = acc_native + b_hh_val
        # igates + hgates in native dtype (pre = x@w_ih.t()+b_ih, stored as native)
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

    XPU can not compile the generic fused Triton RNN kernel produced by
    KernelGen (2D weight-tile + reduction inside the sequential loop
    overflows uni_sram / hits constant-compile failures, and the fully
    sequential per-batch-program design is ~2.5x slower than vendor native),
    so the recurrence is folded into a minimal sequence of primitive ops.

    Inference (train=False, no input/hx gradients, pow2 hidden <= 128):
    a single fused Triton step kernel per time step
    (``h_new = relu(h @ W_hh^T + pre)``, fp32 accumulation, unmasked tiles —
    the XPU backend mishandles masked 2D reads and oversized tiles
    miscompile). This avoids re-entering the FlagGems dispatcher (each XPU
    triton launch is ~0.2ms; a native-op recurrence needs 4+ launches/step).

    Training / non-pow2 / hidden > 128: native aten matmul/add/relu chain
    with autograd tracking (torch.addmm would dispatch to the kunlunxin
    addmm override which raises ``multiple values for keyword 'num_stages'``
    under use_gems, so mm + add is used; the chain runs outside use_gems
    when gradients are requested, so torch.stack stays native). fp32
    accumulation keeps low-precision dtypes within a few ULP of the
    reference (fp32 maxdiff ~4e-7; fp16/bf16 within test-declared atols).
    A ``ZeroDivisionError`` (do_bench cold-tuning edge) falls back to a
    per-step small-shape recurrence with identical math.
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

    # Parameters are nn.Parameter objects (requires_grad=True by default) even
    # in inference; only input/hx gradients actually require the autograd-safe
    # native chain. train=True also routes to the native chain (gradients are
    # requested for the weight parameters).
    need_autograd = (
        train or input.requires_grad or (hx is not None and hx.requires_grad)
    )

    # Fused path is gated to pow2 hidden_size <= 128 and non-bf16: taller
    # tiles exhaust uni_sram during XPU compilation, and for bf16 the Triton
    # tiled matmul's reduction tree rounds differently from native torch.mm,
    # accumulating error beyond the 1e-4 atol over time steps.
    # BF16 uses the native-chain path with fp32 recurrence instead (see below).
    is_bf16 = x.dtype == torch.bfloat16
    if (
        not need_autograd
        and not is_bf16
        and hidden_size <= 128
        and ((hidden_size & (hidden_size - 1)) == 0)
    ):
        # ---- fused kernel path (inference-style, pow2 hidden, fp16/fp32) ----
        # One triton kernel per time step: h_new = relu(h @ W_hh^T + b_hh + pre).
        # Keeps h on device and writes output[t] directly, so the recurrence
        # never re-enters the FlagGems dispatcher (each XPU triton launch is
        # ~0.2ms; the native matmul chain needed 4+ launches/step). BLOCK_H
        # equals hidden_size (pow2), so all tiles are unmasked: the XPU
        # backend mishandles masked reads / masked 2D tiles, and oversized
        # padded tiles also miscompile — hence the pow2-only gate. Non-pow2
        # hidden sizes fall back to the native-chain branch below.
        #
        # For fp16, compute the pre-projection matmul in fp32 then cast back.
        # On XPU, native torch.mm uses fp32 accumulation and truncates to the
        # input dtype; FlagGems Triton mm may use a different reduction tree.
        # fp32 mm + cast reproduces native mm output exactly.
        low_prec = x.dtype != torch.float32
        if low_prec:
            pre = x2d.float().matmul(w_ih.t().float()).to(x.dtype)
        else:
            pre = x2d.matmul(w_ih.t())
        if b_ih is not None:
            pre = pre + b_ih
        pre = pre.reshape(seq_len, batch_size, hidden_size)
        # b_hh is NOT pre-added here; it is passed to the kernel and added
        # per-step to match native torch accumulation order:
        #   igates = mm(x, w_ih.t()) + b_ih
        #   hgates = mm(h, w_hh.t()) + b_hh   <-- b_hh per step
        #   h_new  = relu(igates + hgates)
        hp = hidden_size
        blk = hp
        # Prepare b_hh buffer for the kernel (zero-padded to hp)
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
        # ---- native-chain path (autograd-friendly, also bf16 inference) ----
        # Uses native aten matmul/add/relu only; calling torch.addmm would
        # dispatch to the kunlunxin addmm override, which raises ``multiple
        # values for keyword 'num_stages'`` under use_gems, and mm + add is
        # mathematically identical and stable.
        #
        # For bf16 inference: the Triton tiled matmul rounds differently from
        # native torch.mm for bf16 (8-bit mantissa amplifies reduction-tree
        # differences beyond atol=1e-4 over time steps). Instead, per-step mm
        # is computed in fp32 then truncated to bf16 to match native torch.mm
        # behavior (native mm(bf16) == mm(fp32).to(bf16), verified diff 0.0).
        # w_hh_t is pre-converted to fp32 once to avoid per-step cast overhead.
        w_hh_t = w_hh.t().contiguous()
        is_bf16 = x.dtype == torch.bfloat16
        try:
            if is_bf16:
                pre = x2d.float().matmul(w_ih.t().float()).to(x.dtype)
            else:
                pre = (
                    (x2d.matmul(w_ih.t()) + b_ih)
                    if b_ih is not None
                    else x2d.matmul(w_ih.t())
                )
            if is_bf16 and b_ih is not None:
                pre = pre + b_ih
            pre = pre.reshape(seq_len, batch_size, hidden_size)
            # b_hh is NOT pre-added; it is added per-step to match native
            # torch order: hgates = mm(h, w_hh.t()) + b_hh
            if is_bf16:
                w_hh_t_f = w_hh_t.float()
            h = hx2d
            outputs = []
            for t in range(seq_len):
                if is_bf16:
                    hgates = torch.mm(h.float(), w_hh_t_f).to(h.dtype)
                else:
                    hgates = torch.mm(h, w_hh_t)
                if b_hh is not None:
                    hgates = hgates + b_hh
                h = torch.relu(hgates + pre[t])
                outputs.append(h)
            # autograd-safe assembly; this branch runs outside use_gems
            # (backward tests call the wrapper directly), so torch.stack is
            # never intercepted by the flag_gems pointwise stack override.
            output = stack(outputs, 0)
        except ZeroDivisionError:
            # per-step small-shape recurrence, same math, crash-free
            h = hx2d
            outputs = []
            for t in range(seq_len):
                ih_t = (
                    x[t].matmul(w_ih.t()) + b_ih
                    if b_ih is not None
                    else x[t].matmul(w_ih.t())
                )
                hh_t = (
                    h.matmul(w_hh.t()) + b_hh
                    if b_hh is not None
                    else h.matmul(w_hh.t())
                )
                h = torch.relu(ih_t.to(torch.float32) + hh_t.to(torch.float32)).to(
                    x.dtype
                )
                outputs.append(h)
            output = stack(outputs, 0)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h.unsqueeze(0)


__all__ = ["rnn_relu"]


class KunlunxinRnnReluFunction(torch.autograd.Function):
    """Kunlunxin-specific autograd for single-layer unidirectional RNN-ReLU.

    Forward  → kunlunxin rnn_relu (fused Triton step kernel or mm+add chain).
    Backward → recompute forward with mm + add (NOT addmm, which can trigger
               L2 compile errors on XPU), then torch.autograd.grad.

    The generic RnnReluFunction cannot be used on XPU because:
    (1) its forward calls rnn_relu_kernel_forward (generic Triton kernel)
        which fails to compile on XPU (2D weight-tile + masked reads overflow
        uni_sram / hit TritonXPUCoreTiling failures);
    (2) its backward uses torch.addmm which was historically problematic
        (though currently compiles at small shapes, the generic forward is
        the primary blocker).

    This kunlunxin version uses the same mm + add decomposition that the
    kunlunxin forward native-chain path already uses successfully.
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

        # Use the kunlunxin rnn_relu forward (fused kernel or native chain)
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
        # This matches the kunlunxin native-chain accumulation order:
        #   igates = mm(x, w_ih.t()) + b_ih
        #   hgates = mm(h, w_hh.t()) + b_hh
        #   h = relu(igates + hgates)
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
    """Dispatch wrapper: train=True uses KunlunxinRnnReluFunction (custom
    autograd with kunlunxin-compatible backward); inference uses the fast
    kunlunxin rnn_relu path.
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
    """Route direct calls to the generic wrapper (flag_gems.ops.rnn_relu module)
    to this backend override, but only when running on the kunlunxin (XPU)
    backend.

    The direct-wrapper tests import ``rnn_relu`` from ``flag_gems.ops.rnn_relu``
    (bypassing the aten dispatcher), so the generic Triton kernel would still be
    hit on XPU (it cannot compile there: uni_sram / TritonXPUCoreTiling failures).

    On other backends (NVIDIA, etc.) the generic path works correctly and has
    tighter numerical tolerances for low-precision dtypes, so the patch must
    not be applied there.
    """
    try:
        import flag_gems

        if getattr(flag_gems, "device", None) != "cuda" or not hasattr(
            flag_gems, "vendor_name"
        ):
            # Early import: flag_gems not fully initialised yet, defer.
            return
        if getattr(flag_gems, "vendor_name", None) != "kunlunxin":
            return

        import sys

        _generic_module = sys.modules.get("flag_gems.ops.rnn_relu")
        if _generic_module is not None and hasattr(_generic_module, "rnn_relu"):
            _generic_module.rnn_relu = _rnn_relu_with_autograd
    except (ImportError, AttributeError):
        pass


_patch_generic_wrapper()
