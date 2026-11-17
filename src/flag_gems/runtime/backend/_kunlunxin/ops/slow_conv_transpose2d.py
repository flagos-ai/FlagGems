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

"""Kunlunxin (XPU3) override for ``aten::slow_conv_transpose2d``.

The generic implementation uses ``tl.dot`` (``_sct2d_dot``) and an FMA variant
whose tiles are built from masked loads.  On XPU3 both fail: the dot path
miscompiles masked 2-D operands, and the FMA path dies in MLIR legalisation
(``PassManager::run failed`` → ``OutOfResources: uni_sram``); measured baseline
21/21 functional cases failed to compile/compare.

This override computes the transposed convolution with a mask-free scalar
kernel: ``(BLOCK_W, BLOCK_OC)`` output tile, loops over ``(ci, kh, kw)``,
clamps every gather address and zeroes invalid taps with ``tl.where``.  The
fractional (non-divisible) taps of a strided transposed conv contribute an
exact zero, which is what the ``ok`` predicate encodes.
"""

import logging
import sys as _sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=["N", "C_IN", "C_OUT", "H", "W", "H_OUT", "W_OUT",
                       "SH", "SW", "PH", "PW", "OPH", "OPW", "DH", "DW"]
)
def _sct2d_fma_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    C_IN,
    C_OUT,
    H,
    W,
    H_OUT,
    W_OUT,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    xsn,
    xsc,
    xsh,
    xsw,
    won,
    woc,
    wok,
    wol,
    osn,
    osc,
    osh,
    osw,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    OC_TILES: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_w = tl.program_id(0)
    oh = tl.program_id(1)
    pid_nz = tl.program_id(2)
    pid_n = pid_nz // OC_TILES
    oc = (pid_nz % OC_TILES) * BLOCK_OC + tl.arange(0, BLOCK_OC)

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_valid = ow < W_OUT
    oc_valid = oc < C_OUT
    occ = tl.minimum(oc, C_OUT - 1)

    acc = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
    x_n = x_ptr + pid_n * xsn
    w_c = w_ptr + occ * woc
    for ci_i in range(0, C_IN):
        for kh_i in tl.static_range(0, KH):
            num_h = oh + PH - kh_i * DH
            ih = num_h // SH
            ok_h = (num_h >= 0) & (num_h - ih * SH == 0) & (ih < H)
            ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
            for kw_i in tl.static_range(0, KW):
                num_w = ow + PW - kw_i * DW
                iw = num_w // SW
                ok_w = ow_valid & (num_w >= 0) & (num_w - iw * SW == 0) & (iw < W)
                iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
                m = ok_h & ok_w
                xv = tl.load(x_n + ci_i * xsc + ihc * xsh + iwc * xsw)
                xv = tl.where(m, xv, 0.0).to(tl.float32)
                wv = tl.load(w_c + ci_i * won + (kh_i * wok + kw_i * wol))
                wv = tl.where(oc_valid, wv, 0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + occ)
        acc += tl.where(oc_valid, bv, 0.0).to(tl.float32)[None, :]

    optr = (
        o_ptr
        + pid_n * osn
        + (oh * osh + ow * osw)[:, None]
        + oc[None, :] * osc
    )
    tl.store(
        optr, acc.to(o_ptr.dtype.element_ty), mask=ow_valid[:, None] & oc_valid[None, :]
    )


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def slow_conv_transpose2d(
    input,
    weight,
    kernel_size,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    out=None,
):
    logger.debug("GEMS SLOW_CONV_TRANSPOSE2D")
    x = input
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if x.dim() != 4:
        raise RuntimeError(
            "slow_conv_transpose2d(): expected 3-D or 4-D input, got %d-D" % x.dim()
        )
    if weight.dim() != 4:
        raise RuntimeError(
            "slow_conv_transpose2d(): expected 4-D weight, got %d-D" % weight.dim()
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("slow_conv_transpose2d: unsupported dtype %s" % x.dtype)

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    n, c_in, h, w = x.shape
    wc_in, c_out, kh, kw = weight.shape
    if wc_in != c_in:
        raise RuntimeError(
            "slow_conv_transpose2d(): weight has %d input channels but the input "
            "has %d" % (wc_in, c_in)
        )

    k_h, k_w = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    dh, dw = _pair(dilation)
    if (k_h, k_w) != (kh, kw):
        raise RuntimeError(
            "slow_conv_transpose2d(): kernel_size %s does not match weight %s"
            % ((k_h, k_w), (kh, kw))
        )

    h_out = (h - 1) * sh - 2 * ph + dh * (k_h - 1) + oph + 1
    w_out = (w - 1) * sw - 2 * pw + dw * (k_w - 1) + opw + 1

    if out is not None:
        out4 = out if out.dim() == 4 else out.unsqueeze(0)
    else:
        out4 = torch.empty((n, c_out, h_out, w_out), device=x.device, dtype=x.dtype)
    if out4.numel() == 0:
        return out if out is not None else out4

    block_w = 64 if w_out >= 64 else max(8, min(32, 1 << (max(1, int(w_out)) - 1).bit_length()))
    block_oc = min(32, 1 << (max(1, int(c_out)) - 1).bit_length())
    num_warps = 8 if (w_out >= 64 and c_in * k_h * k_w > 1) else 4
    oc_tiles = triton.cdiv(c_out, block_oc)
    grid = (triton.cdiv(w_out, block_w), h_out, n * oc_tiles)
    _sct2d_fma_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out4,
        n,
        c_in,
        c_out,
        h,
        w,
        h_out,
        w_out,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        out4.stride(0),
        out4.stride(1),
        out4.stride(2),
        out4.stride(3),
        HAS_BIAS=(bias is not None),
        BLOCK_W=block_w,
        BLOCK_OC=block_oc,
        OC_TILES=oc_tiles,
        KH=k_h,
        KW=k_w,
        num_warps=num_warps,
    )
    if out is not None:
        return out
    return out4


# --------------------------------------------------------------------------- #
# Make the *generic* module path resolve to this implementation.
#
# benchmark/test_slow_conv_transpose2d.py binds
# ``gems_op=flag_gems.ops.slow_conv_transpose2d`` (the generic package
# attribute), which bypasses this vendor override; the generic dot kernel
# cannot compile here (SDNN event budget), so the benchmark reported FAILED
# instead of measuring the kernel that actually runs.  Rebind the generic
# submodule attribute and the package attribute to this implementation so both
# the test and benchmark entry points hit the vendor kernel (same pattern as
# ``_kunlunxin/ops/_native_batch_norm_legit.py`` and
# ``slow_conv2d_backward.py``).
_generic_module = _sys.modules.get("flag_gems.ops.slow_conv_transpose2d")
if _generic_module is not None:
    _generic_module.slow_conv_transpose2d = slow_conv_transpose2d
_ops_package = _sys.modules.get("flag_gems.ops")
if _ops_package is not None:
    _ops_package.slow_conv_transpose2d = slow_conv_transpose2d
