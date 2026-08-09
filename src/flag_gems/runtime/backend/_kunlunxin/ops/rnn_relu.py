import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)



@triton.jit
def _rnn_relu_step_kernel(
    h_ptr,
    w_hh_t_ptr,
    pre_ptr,
    h_out_ptr,
    out_step_ptr,
    H_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One fused RNN-ReLU step for every batch row (grid = batch_size).

    h_new = relu(h @ W_hh^T + pre) with fp32 accumulation. The host passes
    zero-padded buffers whose last dim is H_PAD (next pow2 of hidden_size),
    so every tile is unmasked: the XPU backend mishandles masked tail reads
    and can fail to compile masked 2D tiles, so masking is avoided entirely.
    """
    b = tl.program_id(0)
    for hb in range(H_PAD // BLOCK_H):
        o_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for kb in range(H_PAD // BLOCK_H):
            k_offs = kb * BLOCK_H + tl.arange(0, BLOCK_H)
            h_vec = tl.load(h_ptr + b * H_PAD + k_offs).to(tl.float32)
            w_tile = tl.load(
                w_hh_t_ptr + o_offs[:, None] * H_PAD + k_offs[None, :]
            ).to(tl.float32)
            acc += tl.sum(w_tile * h_vec[None, :], axis=1)
        p = tl.load(pre_ptr + b * H_PAD + o_offs).to(tl.float32)
        h_new = tl.where(acc + p > 0, acc + p, 0.0)
        h_new = h_new.to(h_ptr.dtype.element_ty)
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
    need_autograd = train or input.requires_grad or (
        hx is not None and hx.requires_grad
    )

    # Fused path is gated to pow2 hidden_size <= 128: taller tiles (256/512)
    # exhaust uni_sram during XPU kernel compilation (OOM at compile time).
    if (
        not need_autograd
        and hidden_size <= 128
        and ((hidden_size & (hidden_size - 1)) == 0)
    ):
        # ---- fused kernel path (inference-style, pow2 hidden) ----
        # One triton kernel per time step: h_new = relu(h @ W_hh^T + pre).
        # Keeps h on device and writes output[t] directly, so the recurrence
        # never re-enters the FlagGems dispatcher (each XPU triton launch is
        # ~0.2ms; the native matmul chain needed 4+ launches/step). BLOCK_H
        # equals hidden_size (pow2), so all tiles are unmasked: the XPU
        # backend mishandles masked reads / masked 2D tiles, and oversized
        # padded tiles also miscompile — hence the pow2-only gate. Non-pow2
        # hidden sizes fall back to the native-chain branch below.
        pre = (x2d.matmul(w_ih.t()) + b_ih) if b_ih is not None else x2d.matmul(w_ih.t())
        pre = pre.reshape(seq_len, batch_size, hidden_size)
        if b_hh is not None:
            pre = pre + b_hh
        hp = hidden_size
        blk = hp
        h_buf = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in = torch.zeros((batch_size, hp), dtype=x.dtype, device=x.device)
        h_in[:, :hidden_size] = hx2d
        out_buf = torch.zeros((seq_len, batch_size, hp), dtype=x.dtype, device=x.device)
        for t in range(seq_len):
            _rnn_relu_step_kernel[(batch_size,)](
                h_in, w_hh, pre[t], h_buf, out_buf[t],
                H_PAD=hp, BLOCK_H=blk,
            )
            h_in, h_buf = h_buf, h_in
        output = out_buf[..., :hidden_size]
        h = h_in[..., :hidden_size]
    else:
        # ---- native-chain path (autograd-friendly) ----
        # Uses native aten matmul/add/relu only; calling torch.addmm would
        # dispatch to the kunlunxin addmm override, which raises ``multiple
        # values for keyword 'num_stages'`` under use_gems, and mm + add is
        # mathematically identical and stable.  fp32 accumulation inside the
        # matmuls keeps low-precision dtypes within a few ULP of the
        # reference (validated: fp32 maxdiff ~4e-7 on the benchmark matrix;
        # fp16/bf16 within the test-declared relaxed atols).
        w_hh_t = w_hh.t().contiguous()
        try:
            pre = (x2d.matmul(w_ih.t()) + b_ih) if b_ih is not None else x2d.matmul(w_ih.t())
            pre = pre.reshape(seq_len, batch_size, hidden_size)
            if b_hh is not None:
                pre = pre + b_hh
            h = hx2d
            outputs = []
            for t in range(seq_len):
                h = torch.relu(torch.mm(h, w_hh_t) + pre[t])
                outputs.append(h)
            # autograd-safe assembly; this branch runs outside use_gems
            # (backward tests call the wrapper directly), so torch.stack is
            # never intercepted by the flag_gems pointwise stack override.
            output = torch.stack(outputs, 0)
        except ZeroDivisionError:
            # per-step small-shape recurrence, same math, crash-free
            h = hx2d
            outputs = []
            for t in range(seq_len):
                ih_t = x[t].matmul(w_ih.t()) + b_ih if b_ih is not None else x[t].matmul(w_ih.t())
                hh_t = h.matmul(w_hh.t()) + b_hh if b_hh is not None else h.matmul(w_hh.t())
                h = torch.relu(ih_t.to(torch.float32) + hh_t.to(torch.float32)).to(x.dtype)
                outputs.append(h)
            output = torch.stack(outputs, 0)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h.unsqueeze(0)


__all__ = ["rnn_relu"]


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper (flag_gems.ops.rnn_relu module)
    to this backend override.

    The direct-wrapper tests import ``rnn_relu`` from ``flag_gems.ops.rnn_relu``
    (bypassing the aten dispatcher), so the generic Triton kernel would still be
    hit on XPU (it cannot compile there: uni_sram / TritonXPUCoreTiling failures).
    Patching the module attribute at import time keeps the change backend-local:
    the generic module source is untouched and other vendor backends are
    unaffected (this module is only imported for the kunlunxin backend).
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.rnn_relu")
        if _generic_module is not None and hasattr(_generic_module, "rnn_relu"):
            _generic_module.rnn_relu = rnn_relu
    except ImportError:
        pass


_patch_generic_wrapper()