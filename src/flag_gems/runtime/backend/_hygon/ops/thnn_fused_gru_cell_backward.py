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

"""Triton implementation of _thnn_fused_gru_cell_backward.

grad_hy     : (B, H)
workspace   : (B, 5H) holding, in column-chunk order, r, z, n, hx, ghn
Outputs     : grad_input_gates (B,3H), grad_hidden_gates (B,3H), grad_hx (B,H),
              grad_input_bias (3H,), grad_hidden_bias (3H,)  [None, None if no bias]

For each (b, i):
    gz  = g * (hx - n) * (1 - z) * z
    ghx = g * z
    dn  = g * (1 - z) * (1 - n*n)
    ghn = dn * r
    gr  = dn * ghn_ws * (1 - r) * r
    grad_input_gates  = [gr, gz, dn]
    grad_hidden_gates = [gr, gz, ghn]
Nothing transcendental is evaluated; every activation is read back from workspace.

has_bias=False uses a fully parallel 2-D grid. has_bias=True uses one launch
with a hidden-column tile and keeps the batch bias reduction in registers, so
all five outputs come from a single kernel; per-program parallelism
(num_warps) is raised for large batches, where that kernel is memory-latency
bound. Once a kernel binary exists for an exact (kernel, dtype, shape, stride,
pointer-alignment) key, launches go straight to the compiled kernel instead of
through the JIT binder, which is a measurable share of the time on tiny shapes.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _gru_cell_bwd_no_bias(
    grad_hy_ptr,
    ws_ptr,
    gin_ptr,
    ghid_ptr,
    ghx_ptr,
    B,
    H,
    ghy_s0,
    ghy_s1,
    ws_s0,
    ws_s1,
    gin_s0,
    gin_s1,
    ghid_s0,
    ghid_s1,
    ghx_s0,
    ghx_s1,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    cmask = cols < H
    mask = (rows < B)[:, None] & cmask[None, :]

    off_ghy = rows[:, None] * ghy_s0 + cols[None, :] * ghy_s1
    off_ws = rows[:, None] * ws_s0 + cols[None, :] * ws_s1

    g = tl.load(grad_hy_ptr + off_ghy, mask=mask, other=0.0).to(COMPUTE_DTYPE)
    r = tl.load(ws_ptr + off_ws, mask=mask, other=0.0).to(COMPUTE_DTYPE)
    z = tl.load(ws_ptr + off_ws + H * ws_s1, mask=mask, other=0.0).to(COMPUTE_DTYPE)
    n = tl.load(ws_ptr + off_ws + 2 * H * ws_s1, mask=mask, other=0.0).to(COMPUTE_DTYPE)
    hx = tl.load(ws_ptr + off_ws + 3 * H * ws_s1, mask=mask, other=0.0).to(
        COMPUTE_DTYPE
    )
    hn = tl.load(ws_ptr + off_ws + 4 * H * ws_s1, mask=mask, other=0.0).to(
        COMPUTE_DTYPE
    )

    gz = g * (hx - n)
    gz *= 1.0 - z
    gz *= z
    ghx_v = g * z
    dn = g * (1.0 - z)
    dn *= 1.0 - n * n
    ghn = dn * r
    gr = dn * hn
    gr *= 1.0 - r
    gr *= r

    off_gin = rows[:, None] * gin_s0 + cols[None, :] * gin_s1
    off_ghid = rows[:, None] * ghid_s0 + cols[None, :] * ghid_s1
    tl.store(gin_ptr + off_gin, gr, mask=mask)
    tl.store(gin_ptr + off_gin + H * gin_s1, gz, mask=mask)
    tl.store(gin_ptr + off_gin + 2 * H * gin_s1, dn, mask=mask)
    tl.store(ghid_ptr + off_ghid, gr, mask=mask)
    tl.store(ghid_ptr + off_ghid + H * ghid_s1, gz, mask=mask)
    tl.store(ghid_ptr + off_ghid + 2 * H * ghid_s1, ghn, mask=mask)
    off_ghx = rows[:, None] * ghx_s0 + cols[None, :] * ghx_s1
    tl.store(ghx_ptr + off_ghx, ghx_v, mask=mask)


@triton.jit
def _gru_cell_bwd_bias(
    grad_hy_ptr,
    ws_ptr,
    gin_ptr,
    ghid_ptr,
    ghx_ptr,
    bias_in_ptr,
    bias_hid_ptr,
    B,
    H,
    ghy_s0,
    ghy_s1,
    ws_s0,
    ws_s1,
    gin_s0,
    gin_s1,
    ghid_s0,
    ghid_s1,
    ghx_s0,
    ghx_s1,
    bin_s0,
    bhid_s0,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid_h = tl.program_id(0)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < H

    acc_gr = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)
    acc_gz = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)
    acc_dn = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)
    acc_ghn = tl.zeros((BLOCK_B, BLOCK_H), dtype=COMPUTE_DTYPE)

    for b0 in range(0, B, BLOCK_B):
        rows = b0 + tl.arange(0, BLOCK_B)
        mask = (rows < B)[:, None] & cmask[None, :]

        off_ghy = rows[:, None] * ghy_s0 + cols[None, :] * ghy_s1
        off_ws = rows[:, None] * ws_s0 + cols[None, :] * ws_s1

        g = tl.load(grad_hy_ptr + off_ghy, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        r = tl.load(ws_ptr + off_ws, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        z = tl.load(ws_ptr + off_ws + H * ws_s1, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        n = tl.load(ws_ptr + off_ws + 2 * H * ws_s1, mask=mask, other=0.0).to(
            COMPUTE_DTYPE
        )
        hx = tl.load(ws_ptr + off_ws + 3 * H * ws_s1, mask=mask, other=0.0).to(
            COMPUTE_DTYPE
        )
        hn = tl.load(ws_ptr + off_ws + 4 * H * ws_s1, mask=mask, other=0.0).to(
            COMPUTE_DTYPE
        )

        gz = g * (hx - n)
        gz *= 1.0 - z
        gz *= z
        ghx_v = g * z
        dn = g * (1.0 - z)
        dn *= 1.0 - n * n
        ghn = dn * r
        gr = dn * hn
        gr *= 1.0 - r
        gr *= r

        off_gin = rows[:, None] * gin_s0 + cols[None, :] * gin_s1
        off_ghid = rows[:, None] * ghid_s0 + cols[None, :] * ghid_s1
        tl.store(gin_ptr + off_gin, gr, mask=mask)
        tl.store(gin_ptr + off_gin + H * gin_s1, gz, mask=mask)
        tl.store(gin_ptr + off_gin + 2 * H * gin_s1, dn, mask=mask)
        tl.store(ghid_ptr + off_ghid, gr, mask=mask)
        tl.store(ghid_ptr + off_ghid + H * ghid_s1, gz, mask=mask)
        tl.store(ghid_ptr + off_ghid + 2 * H * ghid_s1, ghn, mask=mask)
        off_ghx = rows[:, None] * ghx_s0 + cols[None, :] * ghx_s1
        tl.store(ghx_ptr + off_ghx, ghx_v, mask=mask)

        acc_gr += gr
        acc_gz += gz
        acc_dn += dn
        acc_ghn += ghn

    # One cross-lane reduction per gate, after the whole batch loop.
    sum_gr = tl.sum(acc_gr, axis=0)
    sum_gz = tl.sum(acc_gz, axis=0)
    sum_dn = tl.sum(acc_dn, axis=0)
    sum_ghn = tl.sum(acc_ghn, axis=0)
    tl.store(bias_in_ptr + cols * bin_s0, sum_gr, mask=cmask)
    tl.store(bias_in_ptr + (cols + H) * bin_s0, sum_gz, mask=cmask)
    tl.store(bias_in_ptr + (cols + 2 * H) * bin_s0, sum_dn, mask=cmask)
    tl.store(bias_hid_ptr + cols * bhid_s0, sum_gr, mask=cmask)
    tl.store(bias_hid_ptr + (cols + H) * bhid_s0, sum_gz, mask=cmask)
    tl.store(bias_hid_ptr + (cols + 2 * H) * bhid_s0, sum_ghn, mask=cmask)


_NO_BIAS_BLOCK_B = 8
_NO_BIAS_BLOCK_H = 128
_BIAS_BLOCK_B = 32
_BIAS_BLOCK_H = 32
_BIAS_WARPS_SMALL = 4
_BIAS_WARPS_LARGE = 16
_BIAS_LARGE_ELEMENTS = 65536

_LAUNCH_CACHE = {}
_DRIVER = None


def _validate_inputs(grad_hy, workspace):
    if grad_hy.dim() != 2 or workspace.dim() != 2:
        raise RuntimeError("fused_gru_cell_backward expects two 2-D tensors")
    batch_size, hidden_size = grad_hy.shape
    if workspace.shape != (batch_size, 5 * hidden_size):
        raise RuntimeError(
            "workspace must have shape "
            f"({batch_size}, {5 * hidden_size}), got {tuple(workspace.shape)}"
        )
    if workspace.dtype != grad_hy.dtype:
        raise RuntimeError("grad_hy and workspace must have the same dtype")
    if workspace.device != grad_hy.device:
        raise RuntimeError("grad_hy and workspace must be on the same device")


def _prepare_out(out, shape, dtype, device, name):
    if out.dtype != dtype:
        raise RuntimeError(f"Expected {name} to have dtype {dtype}, got {out.dtype}")
    if out.device != device:
        raise RuntimeError(f"Expected {name} to be on {device}, got {out.device}")
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)


def _dispatch_direct(tag, jit_fn, grid, args, consts_kw, consts_vals, dev):
    """Launch through the compiled kernel once its binary exists for this key.

    Returns True when the launch was issued, False when the caller must fall
    back to the ordinary JIT launch for this key.
    """
    global _DRIVER
    g0 = grid[0]
    g1 = grid[1] if len(grid) > 1 else 1
    key = [tag]
    for a in args:
        if a.__class__ is int:
            key.append(a)
        else:
            key.append((a.dtype, a.data_ptr() & 15))
    key = tuple(key)
    kernel = _LAUNCH_CACHE.get(key)
    if kernel is None:
        try:
            kernel = jit_fn.warmup(*args, grid=(g0, g1, 1), **consts_kw)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            if not getattr(kernel, "function", None):
                raise RuntimeError("compiled kernel handles unavailable")
        except Exception:
            _LAUNCH_CACHE[key] = False
            return False
        _LAUNCH_CACHE[key] = kernel
    elif kernel is False:
        return False
    if _DRIVER is None:
        _DRIVER = triton.runtime.driver.active
    try:
        stream = _DRIVER.get_current_stream(dev)
        kernel.run(
            g0,
            g1,
            1,
            stream,
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            None,
            *args,
            *consts_vals,
        )
    except Exception:
        _LAUNCH_CACHE[key] = False
        return False
    return True


def _launch(grad_hy, workspace, has_bias, gin, ghid, ghx, gib, ghb):
    batch_size, hidden_size = grad_hy.shape
    if batch_size == 0 or hidden_size == 0:
        return
    H = hidden_size
    compute_dtype = tl.float64 if grad_hy.dtype == torch.float64 else tl.float32
    dev = grad_hy.get_device()
    if has_bias:
        if batch_size * H >= _BIAS_LARGE_ELEMENTS:
            num_warps = _BIAS_WARPS_LARGE
        else:
            num_warps = _BIAS_WARPS_SMALL
        grid = (-(-H // _BIAS_BLOCK_H),)
        ghy_s0, ghy_s1 = grad_hy.stride()
        ws_s0, ws_s1 = workspace.stride()
        gin_s0, gin_s1 = gin.stride()
        ghid_s0, ghid_s1 = ghid.stride()
        ghx_s0, ghx_s1 = ghx.stride()
        args = [
            grad_hy,
            workspace,
            gin,
            ghid,
            ghx,
            gib,
            ghb,
            batch_size,
            H,
            ghy_s0,
            ghy_s1,
            ws_s0,
            ws_s1,
            gin_s0,
            gin_s1,
            ghid_s0,
            ghid_s1,
            ghx_s0,
            ghx_s1,
            gib.stride(0),
            ghb.stride(0),
        ]
        if not _dispatch_direct(
            1,
            _gru_cell_bwd_bias,
            grid,
            args,
            {
                "BLOCK_B": _BIAS_BLOCK_B,
                "BLOCK_H": _BIAS_BLOCK_H,
                "COMPUTE_DTYPE": compute_dtype,
                "num_warps": num_warps,
            },
            [_BIAS_BLOCK_B, _BIAS_BLOCK_H, compute_dtype],
            dev,
        ):
            _gru_cell_bwd_bias[grid](
                *args,
                BLOCK_B=_BIAS_BLOCK_B,
                BLOCK_H=_BIAS_BLOCK_H,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=num_warps,
            )
    else:
        grid = (
            -(-H // _NO_BIAS_BLOCK_H),
            -(-batch_size // _NO_BIAS_BLOCK_B),
        )
        ghy_s0, ghy_s1 = grad_hy.stride()
        ws_s0, ws_s1 = workspace.stride()
        gin_s0, gin_s1 = gin.stride()
        ghid_s0, ghid_s1 = ghid.stride()
        ghx_s0, ghx_s1 = ghx.stride()
        args = [
            grad_hy,
            workspace,
            gin,
            ghid,
            ghx,
            batch_size,
            H,
            ghy_s0,
            ghy_s1,
            ws_s0,
            ws_s1,
            gin_s0,
            gin_s1,
            ghid_s0,
            ghid_s1,
            ghx_s0,
            ghx_s1,
        ]
        if not _dispatch_direct(
            0,
            _gru_cell_bwd_no_bias,
            grid,
            args,
            {
                "BLOCK_B": _NO_BIAS_BLOCK_B,
                "BLOCK_H": _NO_BIAS_BLOCK_H,
                "COMPUTE_DTYPE": compute_dtype,
                "num_warps": 4,
            },
            [_NO_BIAS_BLOCK_B, _NO_BIAS_BLOCK_H, compute_dtype],
            dev,
        ):
            _gru_cell_bwd_no_bias[grid](
                *args,
                BLOCK_B=_NO_BIAS_BLOCK_B,
                BLOCK_H=_NO_BIAS_BLOCK_H,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=4,
            )


def _thnn_fused_gru_cell_backward(
    grad_hy,
    workspace,
    has_bias,
    out0=None,
    out1=None,
    out2=None,
    out3=None,
    out4=None,
):
    logger.debug("GEMS_HYGON _THNN_FUSED_GRU_CELL_BACKWARD")
    _validate_inputs(grad_hy, workspace)
    batch_size, hidden_size = grad_hy.shape
    dtype = grad_hy.dtype
    device = grad_hy.device
    H = hidden_size

    if out0 is not None:
        if not has_bias:
            raise RuntimeError("the out overload requires has_bias=True")
        gate_shape = (batch_size, 3 * H)
        bias_shape = (3 * H,)
        for out, shape, name in (
            (out0, gate_shape, "out0"),
            (out1, gate_shape, "out1"),
            (out2, (batch_size, H), "out2"),
            (out3, bias_shape, "out3"),
            (out4, bias_shape, "out4"),
        ):
            _prepare_out(out, shape, dtype, device, name)
        gin, ghid, ghx, gib, ghb = out0, out1, out2, out3, out4
    elif batch_size == 0 or H == 0:
        # No kernel launch for an empty batch or hidden size; the bias sums over
        # an empty batch are exactly zero.
        gin = grad_hy.new_empty((batch_size, 3 * H))
        ghid = grad_hy.new_empty((batch_size, 3 * H))
        ghx = grad_hy.new_empty((batch_size, H))
        if has_bias:
            if batch_size == 0:
                gib = torch.zeros((3 * H,), dtype=dtype, device=device)
                ghb = torch.zeros((3 * H,), dtype=dtype, device=device)
            else:
                gib = grad_hy.new_empty((3 * H,))
                ghb = grad_hy.new_empty((3 * H,))
        else:
            gib = None
            ghb = None
        return (gin, ghid, ghx, gib, ghb)
    else:
        make = grad_hy.new_empty
        gin = make((batch_size, 3 * H))
        ghid = make((batch_size, 3 * H))
        ghx = make((batch_size, H))
        if has_bias:
            gib = make((3 * H,))
            ghb = make((3 * H,))
        else:
            gib = None
            ghb = None

    _launch(grad_hy, workspace, has_bias, gin, ghid, ghx, gib, ghb)
    return (gin, ghid, ghx, gib, ghb)


def _thnn_fused_gru_cell_backward_out(
    grad_hy, workspace, has_bias, out0, out1, out2, out3, out4
):
    return _thnn_fused_gru_cell_backward(
        grad_hy, workspace, has_bias, out0, out1, out2, out3, out4
    )
