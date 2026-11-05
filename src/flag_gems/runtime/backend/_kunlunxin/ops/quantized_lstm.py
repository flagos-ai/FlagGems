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
#
# Kunlunxin (TritonXPU) specialization of quantized_lstm.
#
# Why this override exists (2026-08-31, XPU 5)
# --------------------------------------------
# The general implementation in src/flag_gems/ops/quantized_lstm.py passes
# ``input_precision="tf32x3"`` to ``tl.dot``. TritonXPU only accepts
# ('ieee', 'tf32'), so *every* cell of the marker died at compile time with
#     input_precision must be one of ('ieee', 'tf32'). Got tf32x3
# -> tests were 0 passed / 45 failed and the benchmark aborted on cell #1.
#
# Deltas versus the general kernel
# --------------------------------
# 1. ``tl.dot(..., out_dtype=tl.float32, allow_tf32=False)`` -- the idiom the
#    vendor mm.py already uses -- instead of the unsupported tf32x3.
# 2. The x @ w_ih.T projection is hoisted out of the recurrence and executed as
#    ONE GEMM over all (seq_len * batch) rows with both biases folded in. The
#    serial loop then only does h @ w_hh.T, which roughly halves the per-step
#    work when input_size == hidden_size and replaces seq_len small GEMMs with
#    a single large one.
# 3. No ``tl.load(..., other=)`` anywhere: on this backend ``other=`` has been
#    observed to corrupt in-mask lanes.
# 4. No masked loads and no masked stores at all. Every address is a *plain
#    affine* function of the tile offsets. This is mandatory, not stylistic:
#    probe_proj_variants.py on XPU 5 shows that wrapping an index that feeds a
#    ``tl.dot`` operand address in ``tl.minimum`` / ``%`` / ``tl.where`` makes
#    the TritonXPU ``make_ttxir`` pipeline abort with an MLIR assertion
#    ("Result number is out of range") surfacing as
#    ``OutOfResources ... Required: 0, Hardware limit: 0``.
#    Ragged shapes are therefore handled entirely on the host by zero-padding
#    the operands out to tile boundaries (zeros are exact for a GEMM tail) and
#    narrowing the results back.
# 5. Block sizes are resolved in Python, so nothing is derived from heuristics
#    and the libentry cache key stays complete.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

# Dequantized (and, if the shape is ragged, tile-padded) weights keyed by
# (params object, device, dtype, geometry). The prepacked int8 handles are
# CPU-only, so unpacking involves a host copy; the weights are constant across
# timesteps and calls, so it happens once.
_PARAM_CACHE = {}

_MIN_BLOCK = 16  # tl.dot requires each tile dim to be at least 16
# BLOCK_N below 64 is unusable on this backend: probe_block_combos.py shows the
# 4-dot recurrence kernel refuses to lower at BLOCK_N == 16, and
# probe_gate_kernel.py shows the *pointwise* gate kernel compiles but returns
# SILENTLY WRONG values at BLOCK_N == 32 (cy err 6.4e-1 / hy err 2.8e-1 against
# fp64 math over the identical input buffers) while BLOCK_N == 64 is exact.
_MIN_BLOCK_N = 64
_MIN_BLOCK_K = 32  # 16 is unverified for BLOCK_N == 64; 32 is measured exact
_MAX_MN = 64
_MAX_K = 128


@triton.jit
def quantized_lstm_input_proj_kernel(
    x_ptr,  # (pad_m, pad_k)
    w_ih_ptr,  # (4 * gate_step, pad_k)
    b_ih_ptr,  # (4 * gate_step,) or empty
    b_hh_ptr,  # (4 * gate_step,) or empty
    g_ptr,  # (pad_m, 4 * gate_step) fp32
    n_k,
    gate_step,
    stride_x_row,
    stride_x_col,
    stride_w_row,
    stride_w_col,
    stride_g_row,
    stride_g_col,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """G[:, gate*gate_step + n] = x @ w_ih[gate*gate_step + n, :].T + b_ih + b_hh.

    Grid is (rows, hidden columns, 4 gates). Every extent is a multiple of the
    corresponding block size by construction, so there is no mask anywhere.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    gate = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = gate * gate_step + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(n_k, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_x_row + offs_k[None, :] * stride_x_col
        )
        w_tile = tl.load(
            w_ih_ptr + offs_n[None, :] * stride_w_row + offs_k[:, None] * stride_w_col
        )
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(b_ih_ptr + offs_n).to(tl.float32) + tl.load(
            b_hh_ptr + offs_n
        ).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        g_ptr + offs_m[:, None] * stride_g_row + offs_n[None, :] * stride_g_col,
        acc,
    )


@triton.jit
def quantized_lstm_recur_kernel(
    g_ptr,  # (pad_m, 4 * gate_step) fp32, all-timestep input contribution
    h_prev_ptr,  # (pad_mb, pad_col)
    w_hh_ptr,  # (4 * gate_step, pad_col)
    s_ptr,  # (4, pad_mb, gate_step) fp32, this step's pre-activations
    row_off,  # t * batch_size
    n_k,  # reduction extent (padded hidden size)
    gate_step,
    n_total,  # pad_mb * gate_step, the stride between gate planes of s
    stride_g_row,
    stride_g_col,
    stride_h_row,
    stride_h_col,
    stride_w_row,
    stride_w_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """S = G[t] + h_prev @ w_hh.T for all four gates.

    Kept free of any transcendental on purpose: on TritonXPU a kernel that
    contains both ``tl.dot`` and ``tl.sigmoid``/``tanh`` fails to lower
    (``ConvertTritonSDNNToLLVM`` -> ``PassManager::run failed``); see
    probe_step_variants.py. The gate update therefore lives in its own kernel,
    and S is written gate-major so that kernel can be a flat 1-D pointwise pass.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    g_base = g_ptr + (row_off + offs_m)[:, None] * stride_g_row
    acc_i = tl.load(g_base + offs_n[None, :] * stride_g_col)
    acc_f = tl.load(g_base + (gate_step + offs_n)[None, :] * stride_g_col)
    acc_g = tl.load(g_base + (2 * gate_step + offs_n)[None, :] * stride_g_col)
    acc_o = tl.load(g_base + (3 * gate_step + offs_n)[None, :] * stride_g_col)

    for k in range(0, tl.cdiv(n_k, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        h_tile = tl.load(
            h_prev_ptr + offs_m[:, None] * stride_h_row + offs_k[None, :] * stride_h_col
        )
        w_base = w_hh_ptr + offs_k[:, None] * stride_w_col
        wt_i = tl.load(w_base + offs_n[None, :] * stride_w_row)
        wt_f = tl.load(w_base + (gate_step + offs_n)[None, :] * stride_w_row)
        wt_g = tl.load(w_base + (2 * gate_step + offs_n)[None, :] * stride_w_row)
        wt_o = tl.load(w_base + (3 * gate_step + offs_n)[None, :] * stride_w_row)

        acc_i += tl.dot(h_tile, wt_i, out_dtype=tl.float32, allow_tf32=False)
        acc_f += tl.dot(h_tile, wt_f, out_dtype=tl.float32, allow_tf32=False)
        acc_g += tl.dot(h_tile, wt_g, out_dtype=tl.float32, allow_tf32=False)
        acc_o += tl.dot(h_tile, wt_o, out_dtype=tl.float32, allow_tf32=False)

    s_base = s_ptr + offs_m[:, None] * gate_step + offs_n[None, :]
    tl.store(s_base, acc_i)
    tl.store(s_base + n_total, acc_f)
    tl.store(s_base + 2 * n_total, acc_g)
    tl.store(s_base + 3 * n_total, acc_o)


@triton.jit
def quantized_lstm_gate_kernel(
    s_ptr,  # (4, n_total) fp32 pre-activations, gate-major
    c_prev_ptr,  # (n_total,) flat contiguous
    hy_ptr,  # (n_total,) flat contiguous
    cy_ptr,  # (n_total,) flat contiguous
    n_total,
    BLOCK: tl.constexpr,
):
    """cy = sigmoid(f)*c_prev + sigmoid(i)*tanh(g); hy = sigmoid(o)*tanh(cy).

    Deliberately 1-D and mask-free. The 2-D tiled version of exactly this math
    cost 195-476 us per timestep on XPU 5 (probe_kernel_timing.py) versus 9-14
    us for this flat form at BLOCK 2048-4096 (probe_gate_flat.py) -- a 20-40x
    difference that came purely from the tile layout, not from the
    transcendentals (`tl.sigmoid` and `1/(1+exp)` and `0.5*(1+tanh(x/2))` all
    land within 10% of each other here, and `1/(1+exp(-x))` does not even
    compile).
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pre_i = tl.load(s_ptr + off)
    pre_f = tl.load(s_ptr + n_total + off)
    pre_g = tl.load(s_ptr + 2 * n_total + off)
    pre_o = tl.load(s_ptr + 3 * n_total + off)
    c_prev = tl.load(c_prev_ptr + off).to(tl.float32)

    cy = tl.sigmoid(pre_f) * c_prev + tl.sigmoid(pre_i) * _tanh(pre_g)
    hy = tl.sigmoid(pre_o) * _tanh(cy)

    tl.store(cy_ptr + off, cy.to(cy_ptr.dtype.element_ty))
    tl.store(hy_ptr + off, hy.to(hy_ptr.dtype.element_ty))


def _pad_to(n, b):
    return ((n + b - 1) // b) * b


def _blk(n, hi, lo=_MIN_BLOCK):
    """Largest power of two in [lo, hi] dividing n, else lo."""
    b = hi
    while b > lo:
        if n % b == 0:
            return b
        b //= 2
    return lo


_FLAT_BLOCKS = (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def _flat_block(n):
    """Tile for the 1-D gate kernel: keep it in the measured 1k-4k sweet spot
    and aim for >= 8 programs, but it must divide n exactly (no mask)."""
    for b in _FLAT_BLOCKS:
        if n % b == 0 and n // b >= 8:
            return b
    for b in _FLAT_BLOCKS:
        if n % b == 0:
            return b
    return 1


def _geometry(seq_len, batch_size, input_size, hidden_size):
    """Resolve block sizes and the padded extents they imply.

    Every extent handed to a kernel is a multiple of its block size, which is
    what lets all three kernels run completely mask-free. BLOCK_N is pinned to
    64 and BLOCK_K floored at 32 for the reasons documented next to
    _MIN_BLOCK_N; the hidden dimension is zero-padded up to that pitch.
    """
    bn = _blk(hidden_size, _MAX_MN, _MIN_BLOCK_N)
    bk_in = _blk(input_size, _MAX_K, _MIN_BLOCK_K)
    bk_h = _blk(hidden_size, _MAX_K, _MIN_BLOCK_K)
    step_bm = _blk(batch_size, _MAX_MN)
    m_rows = seq_len * batch_size
    proj_bm = _blk(m_rows, _MAX_MN)

    gate_step = _pad_to(hidden_size, bn)
    pad_kh = _pad_to(hidden_size, bk_h)
    assert pad_kh <= gate_step, (hidden_size, bk_h, bn)
    pad_mb = _pad_to(batch_size, step_bm)
    # Rows must cover the projection grid AND the row block the last timestep
    # reads out of the gate buffer.
    pad_m = _pad_to(
        max(_pad_to(m_rows, proj_bm), (seq_len - 1) * batch_size + pad_mb), proj_bm
    )
    return {
        "bn": bn,
        "bk_in": bk_in,
        "bk_h": bk_h,
        "step_bm": step_bm,
        "proj_bm": proj_bm,
        "m_rows": m_rows,
        "gate_step": gate_step,
        "pad_m": pad_m,
        "pad_kin": _pad_to(input_size, bk_in),
        "pad_kh": pad_kh,
        "pad_mb": pad_mb,
        # pad_kh <= gate_step always holds (bk_h <= 64 whenever hidden is not a
        # multiple of 128), so one padded hidden pitch serves both the reduction
        # extent and the state buffers, and every state buffer stays contiguous
        # -- which is what lets the gate kernel be flat 1-D.
        "pad_col": gate_step,
    }


def _pad_gate_matrix(w, hidden_size, gate_step, pad_k):
    """Re-lay a (4*hidden, k) gate-stacked matrix on gate_step / pad_k pitch."""
    if gate_step == hidden_size and pad_k == w.shape[1] and w.is_contiguous():
        return w
    out = torch.zeros((4 * gate_step, pad_k), device=w.device, dtype=w.dtype)
    for g in range(4):
        out[g * gate_step : g * gate_step + hidden_size, : w.shape[1]] = w[
            g * hidden_size : (g + 1) * hidden_size
        ]
    return out


def _pad_gate_vector(v, hidden_size, gate_step):
    if gate_step == hidden_size:
        return v
    out = torch.zeros((4 * gate_step,), device=v.device, dtype=v.dtype)
    for g in range(4):
        out[g * gate_step : g * gate_step + hidden_size] = v[
            g * hidden_size : (g + 1) * hidden_size
        ]
    return out


def _unpack_cell_params(param, device, dtype, geo, cache):
    """Dequantize one layer's weights and pre-pad them to the tile geometry.

    Same recovery path as the general implementation: CellParamsBase keeps the
    weights inside prepacked linear handles, so they come back through
    ``quantized.linear_unpack`` + ``dequantize`` and the int8 rounding stays
    baked in. The prepacked handles are host-side, so this is cached per params
    object -- repeating the host copy every call would dominate the runtime.
    """
    key = (
        id(param),
        device,
        dtype,
        geo["gate_step"],
        geo["pad_kin"],
        geo["pad_col"],
    )
    hit = cache.get(key)
    if hit is not None:
        return hit

    if isinstance(param, torch.ScriptObject):
        state = param.__getstate__()[0]
        biases, packed = state[1], state[4]
        w_ih = torch.ops.quantized.linear_unpack(packed[0])[0].dequantize()
        w_hh = torch.ops.quantized.linear_unpack(packed[1])[0].dequantize()
        b_ih = biases[0] if len(biases) > 0 else None
        b_hh = biases[1] if len(biases) > 1 else None
    else:
        w_ih, w_hh = param[0], param[1]
        if w_ih.is_quantized:
            w_ih = w_ih.dequantize()
        if w_hh.is_quantized:
            w_hh = w_hh.dequantize()
        b_ih = param[2] if len(param) > 2 else None
        b_hh = param[3] if len(param) > 3 else None

    hidden_size = w_hh.shape[0] // 4
    w_ih = w_ih.to(device=device, dtype=dtype).contiguous()
    w_hh = w_hh.to(device=device, dtype=dtype).contiguous()
    w_ih = _pad_gate_matrix(w_ih, hidden_size, geo["gate_step"], geo["pad_kin"])
    w_hh = _pad_gate_matrix(w_hh, hidden_size, geo["gate_step"], geo["pad_col"])
    if b_ih is not None:
        b_ih = _pad_gate_vector(
            b_ih.to(device=device, dtype=dtype).contiguous(),
            hidden_size,
            geo["gate_step"],
        )
    if b_hh is not None:
        b_hh = _pad_gate_vector(
            b_hh.to(device=device, dtype=dtype).contiguous(),
            hidden_size,
            geo["gate_step"],
        )

    unpacked = (w_ih, w_hh, b_ih, b_hh)
    cache[key] = unpacked
    cache.setdefault("_keepalive", []).append(param)
    return unpacked


def _run_direction(x, h0, c0, w_ih, w_hh, b_ih, b_hh, geo, reverse):
    """Batched input projection followed by the serial recurrence."""
    seq_len, batch_size, input_size = x.shape
    hidden_size = h0.shape[1]
    device, dtype = x.device, x.dtype

    has_bias = b_ih is not None and b_hh is not None
    if not has_bias:
        b_ih = torch.empty(0, device=device, dtype=dtype)
        b_hh = torch.empty(0, device=device, dtype=dtype)

    m_rows = geo["m_rows"]
    pad_m = geo["pad_m"]
    pad_kin = geo["pad_kin"]
    gate_step = geo["gate_step"]
    pad_mb = geo["pad_mb"]

    x_flat = x.reshape(m_rows, input_size)
    if pad_m != m_rows or pad_kin != input_size:
        padded = torch.zeros((pad_m, pad_kin), device=device, dtype=dtype)
        padded[:m_rows, :input_size] = x_flat
        x_flat = padded
    elif not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    # --- stage 1: one GEMM covering every timestep's input contribution -----
    gates = torch.empty((pad_m, 4 * gate_step), device=device, dtype=torch.float32)
    quantized_lstm_input_proj_kernel[
        (pad_m // geo["proj_bm"], gate_step // geo["bn"], 4)
    ](
        x_flat,
        w_ih,
        b_ih,
        b_hh,
        gates,
        pad_kin,
        gate_step,
        x_flat.stride(0),
        x_flat.stride(1),
        w_ih.stride(0),
        w_ih.stride(1),
        gates.stride(0),
        gates.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=geo["proj_bm"],
        BLOCK_N=geo["bn"],
        BLOCK_K=geo["bk_in"],
    )

    # --- stage 2: serial recurrence ----------------------------------------
    # Buffers are allocated ONCE. With one torch.empty per timestep the fp32
    # cells spent more time in the host allocator than in the kernels (0.373 ms
    # measured against a 0.090 ms sum of do_bench'd kernel times), so the whole
    # sequence writes into a single (seq_len, pad_mb, gate_step) arena and the
    # cell state ping-pongs between two slices.
    n_total = pad_mb * gate_step
    hs = torch.empty((seq_len, pad_mb, gate_step), device=device, dtype=dtype)
    c_buf = torch.zeros((2, pad_mb, gate_step), device=device, dtype=dtype)
    c_buf[0, :batch_size, :hidden_size] = c0
    if pad_mb != batch_size or gate_step != hidden_size:
        h = torch.zeros((pad_mb, gate_step), device=device, dtype=dtype)
        h[:batch_size, :hidden_size] = h0
    else:
        h = h0 if h0.is_contiguous() else h0.contiguous()

    grid = (pad_mb // geo["step_bm"], gate_step // geo["bn"])
    flat_block = _flat_block(n_total)
    flat_grid = (n_total // flat_block,)
    pre = torch.empty((4, n_total), device=device, dtype=torch.float32)

    g_s0, g_s1 = gates.stride(0), gates.stride(1)
    w_s0, w_s1 = w_hh.stride(0), w_hh.stride(1)
    pad_kh, step_bm, bn, bk_h = (geo["pad_kh"], geo["step_bm"], geo["bn"], geo["bk_h"])
    h_s0, h_s1 = h.stride(0), h.stride(1)

    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    cur = 0
    for t in steps:
        quantized_lstm_recur_kernel[grid](
            gates,
            h,
            w_hh,
            pre,
            t * batch_size,
            pad_kh,
            gate_step,
            n_total,
            g_s0,
            g_s1,
            h_s0,
            h_s1,
            w_s0,
            w_s1,
            BLOCK_M=step_bm,
            BLOCK_N=bn,
            BLOCK_K=bk_h,
        )

        hy = hs[t]
        nxt = 1 - cur
        quantized_lstm_gate_kernel[flat_grid](
            pre,
            c_buf[cur],
            hy,
            c_buf[nxt],
            n_total,
            BLOCK=flat_block,
        )

        h, cur = hy, nxt
        h_s0, h_s1 = gate_step, 1

    hn = h[:batch_size, :hidden_size]
    cn = c_buf[cur, :batch_size, :hidden_size]
    return hs[:, :batch_size, :hidden_size], hn, cn


def quantized_lstm(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    dtype=None,
    use_dynamic=False,
):
    """Quantized LSTM on TritonXPU. See the module header for the deltas.

    Args:
        input: (seq_len, batch, input_size), or (batch, seq_len, input_size)
            when batch_first is set.
        hx: [h0, c0], each (num_layers * num_directions, batch, hidden_size).
        params: Per-layer weights, either CellParamsBase script objects or flat
            [w_ih, w_hh, b_ih, b_hh] tensor lists.
        has_biases: Whether each layer carries b_ih and b_hh.
        num_layers: Number of stacked layers.
        dropout: Inter-layer dropout probability, applied only when train.
        train: Whether dropout is active.
        bidirectional: Whether to run a second, reversed direction per layer.
        batch_first: Whether input and output put batch on dim 0.
        dtype: Accepted for aten schema compatibility; ignored.
        use_dynamic: Accepted for aten schema compatibility; ignored.

    Returns:
        output, hy, cy matching the aten::quantized_lstm.input contract.
    """
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_LSTM")

    cache = _PARAM_CACHE

    if batch_first:
        input = input.transpose(0, 1)
    input = input.contiguous()

    h0, c0 = hx[0], hx[1]
    num_directions = 2 if bidirectional else 1
    params_per_layer = len(params) // num_layers

    layer_input = input
    h_final = []
    c_final = []

    for layer in range(num_layers):
        seq_len, batch_size, in_size = layer_input.shape
        hidden_size = h0.shape[2]
        geo = _geometry(seq_len, batch_size, in_size, hidden_size)

        dir_outputs = []
        for direction in range(num_directions):
            idx = layer * num_directions + direction
            param = params[layer * params_per_layer + direction]
            w_ih, w_hh, b_ih, b_hh = _unpack_cell_params(
                param, input.device, input.dtype, geo, cache
            )
            if not has_biases:
                b_ih, b_hh = None, None

            out, hy, cy = _run_direction(
                layer_input,
                h0[idx].contiguous(),
                c0[idx].contiguous(),
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                geo,
                reverse=(direction == 1),
            )
            dir_outputs.append(out)
            h_final.append(hy)
            c_final.append(cy)

        layer_input = (
            dir_outputs[0] if num_directions == 1 else torch.cat(dir_outputs, dim=-1)
        )

        # Dropout applies between layers only, never after the last one.
        if train and dropout > 0.0 and layer < num_layers - 1:
            layer_input = torch.nn.functional.dropout(
                layer_input, p=dropout, training=True
            )

    output = layer_input
    if batch_first:
        output = output.transpose(0, 1)

    return output, torch.stack(h_final), torch.stack(c_final)
