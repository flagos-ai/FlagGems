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
# Kunlunxin XPU3 override for ``aten::quantized_gru.input`` / ``.data``.
#
# Why this override exists
# ------------------------
# The generic implementation (``ops/quantized_gru.py``) cannot be compiled by
# the XPU3 Triton backend at all.  Both of its recurrent paths die in the
# compiler: the persistent kernels trip ``arith.addf`` type unification,
# ``tt.addptr`` shape/base-type unification and ``uni_sram`` overflow, and the
# forced fallback (``recurrent_*_gate_kernel``) fails with the same
# ``tt.addptr`` / ``uni_sram`` errors.  Block sizes do not help (the fallback
# already runs BLOCK_N=64 / BLOCK_K=32), so the recurrent formulation itself has
# to be replaced.
#
# Design (mirrors the accepted ``quantized_lstm`` / ``gru_cell`` overrides)
# -----------------------------------------------------------------------
# Per direction:
#   1. one GEMM over all ``seq_len * batch`` rows for the input projection
#      (bias_ih folded in);
#   2. a serial loop over timesteps.  Each step is
#        (a) [int8 weights only] a rowwise symmetric activation quantize of the
#            previous hidden state -- max-abs reduction in a *separate* kernel,
#            because a kernel mixing a reduction/transcendental with ``tl.dot``
#            does not lower here;
#        (b) a recurrent GEMM adding the per-row/per-channel dequant scaling
#            and ``bias_hh`` in a pure-float epilogue;
#        (c) the pointwise gate chain (``_quantized_gru_gate.py``, no dot).
#
# Quantization semantics: the packed weights stay int8 on the device and are
# centred by their per-output-channel zero point on the device
# (``(w - zp).to(fp16)``, exactly the ``_wrapped_quantized_linear_prepacked``
# precedent) -- never host-side ``dequantize()``.  Activations are quantized
# symmetrically per row (``round(x / (max|x|/127))``), matching the generic
# implementation and ATen's dynamic GRU.  An fp16 dot of the integer-valued
# operands plus a float epilogue is then exactly ``(acc - hsum*zp) * hs * ws``.
# For the float16-weight path the weights are upcast to fp32 on the device and
# a full-precision fp32 dot is used (dodges fp16 activation-cast drift).
#
# All kernel extents are block multiples of powers of two / zero-padded on the
# host, so every dot operand load and every state/pre/gate store is mask-free
# and address-clamped.  The only masked store is the final scatter into the
# (possibly bidirectional / packed) output view, with both indices clamped.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.quantized_gru import (
    _packed_metadata,
    _prepare_cell,
    _validate_common,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops._quantized_gru_gate import (
    quantized_gru_gate_kernel,
)

logger = logging.getLogger(__name__)

BN = 64  # dot / gate tile over the hidden dimension (XPU3-safe minimum)
BK = 32  # dot reduction tile (measured-exact floor on this backend)
_PREP_CACHE = {}


def _pow2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _qgru_quant_kernel(x_ptr, q_ptr, s_ptr, COLS: tl.constexpr):
    """Rowwise symmetric int8 quantization: ``q = round(x / (max|x|/127))``.

    ``COLS`` is a power of two and every row is ``COLS`` wide (the host pads the
    buffers), so the whole row is loaded/stored without any mask.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, COLS)
    # Keep the tile 2-D: a 1-D ``tl.max`` returns a rank-0 value whose broadcast
    # to the vector is miscompiled on XPU3 (only lane 0 received the scale), so
    # the whole row is loaded as ``[1, COLS]`` and reduced along axis 1.
    v = tl.load(x_ptr + row * COLS + cols)[None, :]
    m = tl.max(tl.abs(v), axis=1)
    scale = tl.where(m > 0.0, m / 127.0, 1.0)
    t = v / scale[:, None]
    q = tl.where(t >= 0.0, tl.floor(t + 0.5), tl.ceil(t - 0.5))
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    tl.store(q_ptr + row * COLS + cols[None, :], q.to(tl.float16))
    tl.store(s_ptr + row + tl.arange(0, 1), scale)


@triton.jit
def _qgru_unpack_weight_kernel(
    w_ptr,  # (H, I) int8 (int8 path) or low precision (fp path), one gate
    zp_ptr,  # (H,) int32 per-output-channel zero point, or dummy
    out_ptr,  # (GS, KPAD) fp16 (int8 path) / fp32 (fp path), one gate
    H,
    I,
    out_row_stride,
    KCOL: tl.constexpr,
    QUANT: tl.constexpr,
):
    """Re-lay one gate's weight on the ``GS`` / ``KPAD`` tile pitch.

    One program per padded row (scalar row index).  On XPU3 the multi-program
    ``[KROW, KCOL]`` tile form silently produced wrong/missing rows once the
    grid was wider than one program, and a 1-D ``tl.max`` broadcast only reaches
    lane 0, so this setup pass keeps every address scalar-derived.  Called once
    per gate with host-offset base pointers; columns are zero padded to ``KPAD``
    and reads are index-clamped (never ``other=``).
    """
    rn = tl.program_id(0)
    rk = tl.arange(0, KCOL)
    raw = tl.minimum(rn, H - 1)
    kc = tl.minimum(rk, I - 1)
    w = tl.load(w_ptr + raw * I + kc).to(tl.float32)
    if QUANT:
        w = w - tl.load(zp_ptr + raw).to(tl.float32)
    # Only the real columns are stored; the host zero-fills the buffer so the
    # zero-padded columns stay exactly zero (masking the *store* is reliable on
    # XPU3, whereas a ``tl.where`` lane-select into the padded columns was not).
    tl.store(
        out_ptr + rn * out_row_stride + rk,
        w.to(out_ptr.dtype.element_ty),
        mask=(rk < I) & (rn < H),
    )


@triton.jit
def _qgru_proj_kernel(
    x_ptr,  # (pad_m, KPAD) fp32, or int8-quantized fp16
    w_ptr,  # (GS, KPAD) one gate
    bias_ptr,  # (GS,) fp32 (bias_ih, already gate-padded)
    xs_ptr,  # (pad_m,) fp32 activation scale (int8 path) or dummy
    ws_ptr,  # (GS,) fp32 weight scale (int8 path) or dummy
    g_ptr,  # (pad_m * GS) fp32 output for one gate
    x_row_stride,
    w_row_stride,
    g_row_stride,
    NK: tl.constexpr,
    BM: tl.constexpr,
    BN_C: tl.constexpr,
    BK_C: tl.constexpr,
    QUANT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN_C + tl.arange(0, BN_C)
    acc = tl.zeros((BM, BN_C), dtype=tl.float32)
    for k0 in range(0, NK, BK_C):
        rk = k0 + tl.arange(0, BK_C)
        a = tl.load(x_ptr + om[:, None] * x_row_stride + rk[None, :])
        w = tl.load(w_ptr + on[None, :] * w_row_stride + rk[:, None])
        if QUANT:
            acc += tl.dot(a, w, out_dtype=tl.float32)
        else:
            acc += tl.dot(a, w, out_dtype=tl.float32, allow_tf32=False)
    if QUANT:
        xs = tl.load(xs_ptr + om)
        ws = tl.load(ws_ptr + on)
        acc = acc * xs[:, None] * ws[None, :]
    bias = tl.load(bias_ptr + on)
    acc += bias[None, :]
    tl.store(g_ptr + om[:, None] * g_row_stride + on[None, :], acc)


@triton.jit
def _qgru_recur_kernel(
    gates_ptr,  # (3, pad_m * GS) fp32 input projections
    x_ptr,  # (pad_mb, GS) fp32 state or fp16 quantized state
    w_ptr,  # (3*GS, GS)
    bias_ptr,  # (3*GS,) fp32 bias_hh
    xs_ptr,  # (pad_mb,) fp32 activation scale (int8 path) or dummy
    ws_ptr,  # (3*GS,) fp32 weight scale (int8 path) or dummy
    pre_ptr,  # (3, pad_mb * GS) fp32 recurrent contributions
    row_off,
    g_row_stride,
    g_plane_stride,
    x_row_stride,
    w_row_stride,
    pre_plane_stride,
    NK: tl.constexpr,
    GS: tl.constexpr,
    BM: tl.constexpr,
    BN_C: tl.constexpr,
    BK_C: tl.constexpr,
    QUANT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN_C + tl.arange(0, BN_C)
    g_off = (row_off + om)[:, None] * g_row_stride + on[None, :]
    g_r = tl.load(gates_ptr + g_off)
    g_z = tl.load(gates_ptr + g_plane_stride + g_off)

    acc_r = tl.zeros((BM, BN_C), dtype=tl.float32)
    acc_z = tl.zeros((BM, BN_C), dtype=tl.float32)
    acc_n = tl.zeros((BM, BN_C), dtype=tl.float32)
    for k0 in range(0, NK, BK_C):
        rk = k0 + tl.arange(0, BK_C)
        a = tl.load(x_ptr + om[:, None] * x_row_stride + rk[None, :])
        w_base = w_ptr + rk[:, None]
        w_r = tl.load(w_base + on[None, :] * w_row_stride)
        w_z = tl.load(w_base + (GS + on)[None, :] * w_row_stride)
        w_n = tl.load(w_base + (2 * GS + on)[None, :] * w_row_stride)
        if QUANT:
            acc_r += tl.dot(a, w_r, out_dtype=tl.float32)
            acc_z += tl.dot(a, w_z, out_dtype=tl.float32)
            acc_n += tl.dot(a, w_n, out_dtype=tl.float32)
        else:
            acc_r += tl.dot(a, w_r, out_dtype=tl.float32, allow_tf32=False)
            acc_z += tl.dot(a, w_z, out_dtype=tl.float32, allow_tf32=False)
            acc_n += tl.dot(a, w_n, out_dtype=tl.float32, allow_tf32=False)

    if QUANT:
        xs = tl.load(xs_ptr + om)
        wr_s = tl.load(ws_ptr + on)
        wz_s = tl.load(ws_ptr + GS + on)
        wn_s = tl.load(ws_ptr + 2 * GS + on)
        acc_r = acc_r * xs[:, None] * wr_s[None, :]
        acc_z = acc_z * xs[:, None] * wz_s[None, :]
        acc_n = acc_n * xs[:, None] * wn_s[None, :]

    br = tl.load(bias_ptr + on)
    bz = tl.load(bias_ptr + GS + on)
    bn = tl.load(bias_ptr + 2 * GS + on)
    # r/z already carry their input projection; n's recurrent part keeps b_hn
    # inside the reset multiply, so gates_n is added in the gate kernel instead.
    pre_r = g_r + acc_r + br[None, :]
    pre_z = g_z + acc_z + bz[None, :]
    pre_n = acc_n + bn[None, :]
    p_off = om[:, None] * GS + on[None, :]
    tl.store(pre_ptr + p_off, pre_r)
    tl.store(pre_ptr + pre_plane_stride + p_off, pre_z)
    tl.store(pre_ptr + 2 * pre_plane_stride + p_off, pre_n)


def _pad_to(n, b):
    return ((n + b - 1) // b) * b


def _geometry(seq_len, batch_size, max_row_off, input_size, hidden_size):
    hpad = max(64, _pow2(hidden_size))
    kpad_in = max(BK, _pow2(input_size))
    bm = 16 if batch_size <= 16 else (32 if batch_size <= 32 else 64)
    pad_mb = _pad_to(batch_size, bm)
    m_rows = seq_len * batch_size
    rows_cov = max(m_rows, max_row_off + pad_mb)
    pbm = 64 if rows_cov >= 64 else (16 if rows_cov <= 16 else 32)
    pad_m = _pad_to(rows_cov, pbm)
    return {
        "gs": hpad,
        "kpad_in": kpad_in,
        "kpad_h": hpad,
        "bm": bm,
        "pad_mb": pad_mb,
        "pbm": pbm,
        "pad_m": pad_m,
    }


def _pad_vec(v, hidden_size, gs):
    out = torch.zeros((3 * gs,), device=v.device, dtype=torch.float32)
    for g in range(3):
        out[g * gs : g * gs + hidden_size] = v[g * hidden_size : (g + 1) * hidden_size]
    return out


def _prep_weights(param, device, hidden_size, input_size, geo):
    """Unpack/pre-pad the weights for one layer-direction (cached).

    The int8 path keeps the packed weight int8 and centres it by the
    per-output-channel zero point *on device* (a Triton kernel); the fp16 path
    upcasts to fp32 on device.  No host-side ``dequantize()``.
    """
    cell = _prepare_cell(param, device)
    ih, hh = cell.ih, cell.hh
    quant = ih.quantized
    key = (
        id(param),
        str(device),
        geo["gs"],
        geo["kpad_in"],
        geo["kpad_h"],
        quant,
    )
    hit = _PREP_CACHE.get(key)
    if hit is not None:
        return hit

    gs, kpad_in, kpad_h = geo["gs"], geo["kpad_in"], geo["kpad_h"]
    wdt = torch.float16 if quant else torch.float32

    def _unpack(w, k, zpd):
        """Gate-pad the weight on the tile pitch; centre it by zp on device."""
        out = torch.zeros((3 * gs, k), device=device, dtype=wdt)
        grid = (gs,)
        for g in range(3):
            wg = w[g * hidden_size :]
            zg = zpd[g * hidden_size :] if zpd is not None else w
            _qgru_unpack_weight_kernel[grid](
                wg,
                zg,
                out[g * gs :],
                hidden_size,
                w.shape[1],
                out.stride(0),
                KCOL=k,
                QUANT=quant,
            )
        return out

    dummy = torch.empty((1,), device=device, dtype=torch.float32)
    w_ih = _unpack(ih.weight, kpad_in, ih.zero_points if quant else None)
    w_hh = _unpack(hh.weight, kpad_h, hh.zero_points if quant else None)
    b_ih = _pad_vec(ih.bias.to(torch.float32), hidden_size, gs)
    b_hh = _pad_vec(hh.bias.to(torch.float32), hidden_size, gs)
    if quant:
        s_ih = _pad_vec(ih.scales.to(torch.float32), hidden_size, gs)
        s_hh = _pad_vec(hh.scales.to(torch.float32), hidden_size, gs)
    else:
        s_ih = dummy
        s_hh = dummy

    out = (quant, w_ih, w_hh, b_ih, b_hh, s_ih, s_hh)
    _PREP_CACHE[key] = out
    _PREP_CACHE.setdefault("_keepalive", []).append(param)
    return out


def _run_direction(
    x_flat,
    rows,
    seq_len,
    batch_size,
    h0,
    weight_state,
    out2d,
    final_h,
    dir_off,
    reverse,
    geo,
    row_offsets,
    active_batches,
):
    device = x_flat.device
    input_size = x_flat.shape[1]
    hidden_size = h0.shape[1]
    gs = geo["gs"]
    kpad_in, kpad_h = geo["kpad_in"], geo["kpad_h"]
    pad_m, pad_mb = geo["pad_m"], geo["pad_mb"]
    pbm, bm = geo["pbm"], geo["bm"]
    quant, w_ih, w_hh, b_ih, b_hh, s_ih, s_hh = weight_state

    # ---- pad the flattened input to the tile pitch ------------------------
    xp = torch.zeros((pad_m, kpad_in), device=device, dtype=torch.float32)
    xp[:rows, :input_size] = x_flat
    x_row_stride = kpad_in
    if quant:
        qx = torch.empty((pad_m, kpad_in), device=device, dtype=torch.float16)
        xs = torch.empty((pad_m,), device=device, dtype=torch.float32)
        _qgru_quant_kernel[(pad_m,)](xp, qx, xs, COLS=kpad_in, num_warps=4)
        proj_x, proj_xs = qx, xs
    else:
        proj_x, proj_xs = xp, w_ih.new_empty((1,), dtype=torch.float32)

    # ---- stage 1: input projection for every timestep ---------------------
    gates = torch.empty((3, pad_m * gs), device=device, dtype=torch.float32)
    g_row_stride, g_plane_stride = gs, pad_m * gs
    proj_grid = (pad_m // pbm, gs // BN)
    for g in range(3):
        _qgru_proj_kernel[proj_grid](
            proj_x,
            w_ih[g * gs :],
            b_ih[g * gs :],
            proj_xs,
            s_ih[g * gs :],
            gates[g],
            x_row_stride,
            w_ih.stride(0),
            g_row_stride,
            NK=kpad_in,
            BM=pbm,
            BN_C=BN,
            BK_C=BK,
            QUANT=quant,
        )

    # ---- stage 2: serial recurrence ---------------------------------------
    state = torch.zeros((2, pad_mb, gs), device=device, dtype=torch.float32)
    state[0, :batch_size, :hidden_size] = h0
    pre = torch.empty((3, pad_mb * gs), device=device, dtype=torch.float32)
    pre_plane_stride = pad_mb * gs
    qh = torch.empty((pad_mb, gs), device=device, dtype=torch.float16)
    hs = torch.empty((pad_mb,), device=device, dtype=torch.float32)
    dummy = w_ih.new_empty((1,), dtype=torch.float32)

    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    cur = 0
    grid = (pad_mb // bm, gs // BN)
    for t in steps:
        row_off = row_offsets[t]
        active = active_batches[t]
        if quant:
            _qgru_quant_kernel[(pad_mb,)](state[cur], qh, hs, COLS=gs, num_warps=4)
            recur_x, recur_xs = qh, hs
        else:
            recur_x, recur_xs = state[cur], dummy
        _qgru_recur_kernel[grid](
            gates,
            recur_x,
            w_hh,
            b_hh,
            recur_xs,
            s_hh,
            pre,
            row_off,
            g_row_stride,
            g_plane_stride,
            gs,
            w_hh.stride(0),
            pre_plane_stride,
            NK=kpad_h,
            GS=gs,
            BM=bm,
            BN_C=BN,
            BK_C=BK,
            QUANT=quant,
        )
        nxt = 1 - cur
        quantized_gru_gate_kernel[grid](
            pre,
            gates,
            state[cur],
            state[nxt],
            out2d,
            final_h,
            row_off,
            active,
            batch_size,
            pre_plane_stride,
            g_row_stride,
            g_plane_stride,
            out2d.stride(0),
            dir_off,
            H=hidden_size,
            GS=gs,
            BM=bm,
            BN=BN,
        )
        cur = nxt


def _run(
    layer_input,
    hx,
    params,
    num_layers,
    bidirectional,
    packed,
    seq_len,
    batch_size,
    rows,
    row_offsets,
    active_batches,
):
    directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]
    all_hidden = torch.empty_like(hx)
    max_row_off = row_offsets[-1]

    with torch_device_fn.device(layer_input.device):
        for layer in range(num_layers):
            in_size = layer_input.shape[-1]
            geo = _geometry(seq_len, batch_size, max_row_off, in_size, hidden_size)
            cells = [
                params[layer * directions + d] for d in range(directions)
            ]
            weights = [
                _prep_weights(c, layer_input.device, hidden_size, in_size, geo)
                for c in cells
            ]
            if packed:
                layer_output = torch.empty(
                    (rows, directions * hidden_size),
                    device=layer_input.device,
                    dtype=layer_input.dtype,
                )
            else:
                layer_output = torch.empty(
                    (seq_len, batch_size, directions * hidden_size),
                    device=layer_input.device,
                    dtype=layer_input.dtype,
                )
            out2d = layer_output.view(rows, directions * hidden_size)
            x_flat = layer_input.reshape(rows, in_size)
            for direction in range(directions):
                idx = layer * directions + direction
                _run_direction(
                    x_flat,
                    rows,
                    seq_len,
                    batch_size,
                    hx[idx],
                    weights[direction],
                    out2d,
                    all_hidden[idx],
                    direction * hidden_size,
                    direction == 1,
                    geo,
                    row_offsets,
                    active_batches,
                )
            layer_input = layer_output
    return layer_input, all_hidden


def quantized_gru_input(
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
    logger.debug("GEMS KUNLUNXIN QUANTIZED_GRU.INPUT")
    _validate_common(
        input, hx, params, has_biases, num_layers, dropout, train, bidirectional
    )
    if input.ndim != 3:
        raise ValueError("quantized_gru.input expects a 3-D input")
    if batch_first:
        input = input.transpose(0, 1)
    seq_len, batch_size, _ = input.shape
    if seq_len <= 0 or batch_size != hx.shape[1]:
        raise ValueError("invalid sequence length or batch size")
    rows = seq_len * batch_size
    input = input.reshape(rows, input.shape[-1])
    output, hidden = _run(
        input,
        hx,
        params,
        num_layers,
        bidirectional,
        False,
        seq_len,
        batch_size,
        rows,
        [t * batch_size for t in range(seq_len)],
        [batch_size] * seq_len,
    )
    output = output.view(seq_len, batch_size, output.shape[-1])
    if batch_first:
        output = output.transpose(0, 1)
    return output, hidden


def quantized_gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    logger.debug("GEMS KUNLUNXIN QUANTIZED_GRU.DATA")
    _validate_common(
        data, hx, params, has_biases, num_layers, dropout, train, bidirectional
    )
    if data.ndim != 2 or batch_sizes.ndim != 1:
        raise ValueError("quantized_gru.data expects packed 2-D data and batch_sizes")
    sizes, offsets, _, _, _, rows = _packed_metadata(batch_sizes, data.device)
    if rows != data.shape[0] or sizes[0] != hx.shape[1]:
        raise ValueError("packed data, batch_sizes and hx are inconsistent")
    output, hidden = _run(
        data,
        hx,
        params,
        num_layers,
        bidirectional,
        True,
        len(sizes),
        sizes[0],
        rows,
        offsets,
        sizes,
    )
    return output, hidden


quantized_gru_impl = quantized_gru_input


# The functional test and the benchmark both bind ``quantized_gru_input`` /
# ``quantized_gru_data`` from the *generic* module path
# (``from flag_gems.ops.quantized_gru import quantized_gru_input``), bypassing
# the registrar.  Rebind those attributes so both call paths reach the kernel
# that actually compiles on XPU3 (same pattern as
# ``_batch_norm_impl_index`` / ``div_mode``).
import sys as _sys  # noqa: E402

_generic_module = _sys.modules.get("flag_gems.ops.quantized_gru")
if _generic_module is not None:
    _generic_module.quantized_gru_input = quantized_gru_input
    _generic_module.quantized_gru_data = quantized_gru_data
    _generic_module.quantized_gru_impl = quantized_gru_impl
