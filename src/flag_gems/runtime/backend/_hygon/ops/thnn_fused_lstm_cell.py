import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _thnn_fused_lstm_cell_kernel(
    ig_ptr,  # input_gates [B, 4H]
    hg_ptr,  # hidden_gates [B, 4H]
    cx_ptr,  # cx [B, H]
    ib_ptr,  # input_bias [4H] or empty
    hb_ptr,  # hidden_bias [4H] or empty
    hy_ptr,  # hy [B, H]
    cy_ptr,  # cy [B, H]
    ws_ptr,  # workspace [B, 4H]
    H: tl.constexpr,
    HAS_IB: tl.constexpr,
    HAS_HB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    hb_id = tl.program_id(1)
    offs = hb_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H

    row_base = b * (4 * H)
    h_base = b * H

    cx = tl.load(cx_ptr + h_base + offs, mask=mask, other=0.0).to(tl.float32)

    # Gate 0: input gate i, gate 1: forget gate f, gate 2: cell gate g,
    # gate 3: output gate o  (PyTorch ordering)
    i0 = tl.load(ig_ptr + row_base + 0 * H + offs, mask=mask, other=0.0).to(tl.float32)
    i1 = tl.load(ig_ptr + row_base + 1 * H + offs, mask=mask, other=0.0).to(tl.float32)
    i2 = tl.load(ig_ptr + row_base + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
    i3 = tl.load(ig_ptr + row_base + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)

    h0 = tl.load(hg_ptr + row_base + 0 * H + offs, mask=mask, other=0.0).to(tl.float32)
    h1 = tl.load(hg_ptr + row_base + 1 * H + offs, mask=mask, other=0.0).to(tl.float32)
    h2 = tl.load(hg_ptr + row_base + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
    h3 = tl.load(hg_ptr + row_base + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)

    if HAS_IB:
        i0 += tl.load(ib_ptr + 0 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i1 += tl.load(ib_ptr + 1 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i2 += tl.load(ib_ptr + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i3 += tl.load(ib_ptr + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)

    if HAS_HB:
        i0 += tl.load(hb_ptr + 0 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i1 += tl.load(hb_ptr + 1 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i2 += tl.load(hb_ptr + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
        i3 += tl.load(hb_ptr + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)

    # Pre-activations
    g0 = i0 + h0
    g1 = i1 + h1
    g2 = i2 + h2
    g3 = i3 + h3

    # Activations (workspace semantics match torch aten op)
    w0 = tl.sigmoid(g0)  # i
    w1 = tl.sigmoid(g1)  # f
    w2 = libdevice.tanh(g2)  # g
    w3 = tl.sigmoid(g3)  # o

    cy = w1 * cx + w0 * w2
    hy = w3 * libdevice.tanh(cy)

    ws_dtype = ws_ptr.dtype.element_ty
    out_dtype = cy_ptr.dtype.element_ty

    tl.store(ws_ptr + row_base + 0 * H + offs, w0.to(ws_dtype), mask=mask)
    tl.store(ws_ptr + row_base + 1 * H + offs, w1.to(ws_dtype), mask=mask)
    tl.store(ws_ptr + row_base + 2 * H + offs, w2.to(ws_dtype), mask=mask)
    tl.store(ws_ptr + row_base + 3 * H + offs, w3.to(ws_dtype), mask=mask)
    tl.store(cy_ptr + h_base + offs, cy.to(out_dtype), mask=mask)
    tl.store(hy_ptr + h_base + offs, hy.to(out_dtype), mask=mask)


def run(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    batch_size, gates_dim = input_gates.shape
    hidden_size = gates_dim // 4

    hy = torch.empty_like(cx)
    cy = torch.empty_like(cx)
    workspace = torch.empty_like(input_gates)

    ib = input_gates.new_empty(0) if input_bias is None else input_bias
    hb = hidden_gates.new_empty(0) if hidden_bias is None else hidden_bias

    BLOCK = 32
    nw = 1
    grid = (batch_size, triton.cdiv(hidden_size, BLOCK))
    _thnn_fused_lstm_cell_kernel[grid](
        input_gates,
        hidden_gates,
        cx,
        ib,
        hb,
        hy,
        cy,
        workspace,
        hidden_size,
        input_bias is not None,
        hidden_bias is not None,
        BLOCK,
        num_warps=nw,
    )
    return hy, cy, workspace


# Alias for FlagGems import convention
thnn_fused_lstm_cell = run
