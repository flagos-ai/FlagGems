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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Fixed-grid SIP budget.  The launch grid is ``cdiv(MAX_GRID_DIM, num_warps)``;
# with the hard limit ``num_warps == 4`` this is 6, and 6 programs x 4 warps ==
# 24 warps exactly saturates the 24 SIPs.  A direct sweep of grid strides
# {6,12,24,48,96,128} confirmed 6 is optimal on every shape -- larger grids
# oversubscribe the SIPs and run 15-40% slower -- so MAX_GRID_DIM stays 24.
MAX_GRID_DIM = 24
# Upper bound on the (rare, tiny) left-padding edge width this kernel handles.
# The edge is computed by a matrix-vector reduction (no ``tl.dot``) over
# ``OL0c = ceil(pad_left/S)`` output columns, unrolled by ``tl.static_range``.
# The host guards ``OL0c <= BLOCK_EDGE`` and falls back to the generic path
# above it, so one tile always covers the edge and the unroll stays bounded for
# any realistic padding.  This is only the *guard*: the ``BLOCK_E`` constexpr
# actually passed to the kernel is ``next_power_of_2(edge_len)``, i.e. tight to
# the columns really stored, because sizing the ``(BLOCK_M, BLOCK_E)``
# accumulator at the 64 worst case wastes 8-64x of the per-tap where+add vector
# work (measured 1.01-1.72x slower on pad>0 shapes).
BLOCK_EDGE = 64


# ---------------------------------------------------------------------------
# im2col-free 1D convolution written as a batched GEMM over the *stride-1*
# ("full") output axis ``f``::
#
#     full[oc, f] = sum_{k, ci} w2[oc, k, ci] * x[ci, f + k*D - pad_left]
#     out[oc, ol] = full[oc, ol*S]
#
# M = OCg, N = f (the long contiguous axis), reduction = (k, ci) walked as an
# outer loop over taps ``k`` and an inner loop over input-channel blocks.
#
# Three measured GCU300 facts shape this kernel:
#
#   1. ``tl.make_block_ptr`` rides the DMA fast path; an explicit-pointer load
#      of the same tile is ~15-20x slower.  So the input (B), the weight (A)
#      and the stride-1 output all use block_ptr.
#   2. A block_ptr offset that goes *negative* is unreliable: ``boundary_check``
#      only guards the upper bound, small tensors silently read garbage and
#      large ones raise a hardware ``Sdte`` fault.  Negative offsets are
#      therefore eliminated *by construction* rather than masked.
#   3. A block_ptr whose fast dimension has stride > 1 corrupts memory, so the
#      requested stride ``S`` can not be folded into the input load.
#
# Fact (2) is handled by **shifted tiling**: main tile ``j`` covers
# ``f in [pad_left + j*BLOCK_N, ...)`` so its load offset is exactly
# ``j*BLOCK_N + k*D >= 0``.  Right padding needs no help -- it is an *upper*
# bound overflow which ``boundary_check`` zero-fills correctly.  The consequence
# is that the leading ``OL0 = ceil(pad_left / S)`` output columns are not
# covered; they form a tiny edge region (``OL0 <= pad_left``, i.e. a handful of
# columns for every realistic padding) which a second, small masked-gather loop
# computes.  ``HAS_EDGE`` is a constexpr, so a padding-free call compiles to a
# pure block_ptr kernel with no gather code at all.
#
# Fact (3) is handled by computing at stride 1 and **down-sampling in the
# store** (``out[ol] = full[ol*S]``), which keeps the input load unit-stride for
# every ``S``.  ``S`` is a constexpr, so ``S == 1`` keeps a pure block_ptr store
# while ``S > 1`` compiles a masked down-sample store -- no runtime branch.
#
# Both loops live in one kernel: a single launch per ``conv1d`` call, and no
# large-tensor PyTorch work on the forward path (only the tiny weight is
# reshaped, ``OC*Cg*K`` elements, to make the A tile row-major for the MMA).
# ---------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conv1d"),
    # S/K/D/OL0 belong in the key: each one changes the code the kernel actually
    # runs (S selects the whole main-loop branch, K the tap-loop trip count, D
    # the tap spacing, OL0 the edge width and hence BLOCK_E), yet log-bucketed
    # N/OF/OCg/Cg alone does not separate them.  OF=2042/2048/2054 all bucket to
    # 2048, so the pad=0, pad=3 and pad=6 shapes at N=16/OCg=48/Cg=12 shared ONE
    # tuned config and pad=0 was handed the pad=6 winner -- BM128 at 0.517ms
    # instead of its own BM64 at 0.324ms, a 1.6x loss on a shape with no edge at
    # all.  These four use "default" (exact) rather than "log" because OL0 and
    # pad_left are 0 on the common no-padding path and log2(0) is undefined.
    key=["N", "OF", "OCg", "Cg", "S", "K", "D", "OL0"],
    strategy=["log", "log", "log", "log", "default", "default", "default", "default"],
)
@triton.jit
def conv1d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    OF,
    OL0,
    OCg,
    G,
    L,
    pad_left: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oc: tl.constexpr,
    Cg: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_EDGE: tl.constexpr,
    GRID_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_num = tl.cdiv(GRID_DIM, num_warps)
    grid_m = tl.cdiv(OCg, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n_kblocks = tl.cdiv(Cg, BLOCK_K)
    # number of leading output columns the shifted main tiling can not reach
    # (== ceil(pad_left / S)); covered by the edge loop.  Defined once because a
    # constexpr can not be reassigned, and both the S>1 main loop and the edge
    # need it.
    OL0c: tl.constexpr = (pad_left + S - 1) // S

    # ===================== main region (output columns) =====================
    # S == 1: tile the stride-1 "full" axis f directly; every computed column is
    #         an output column, so the store is a plain contiguous block_ptr.
    # S > 1 : tile the OUTPUT axis ol.  For output tile [ol0, ol0+BLOCK_OL) the
    #         input span needed by tap k is the *contiguous* run of width
    #         BLOCK_NP = BLOCK_OL*S starting at base_k = ol0*S + k*D - pad_left
    #         (>= 0 by shifted tiling).  Load it with a unit-stride block_ptr
    #         (DMA fast path) and dot the FULL span into acc (BM, BLOCK_NP): the
    #         needed output column ol0+i always sits at acc[:, i*S] for every tap
    #         (base_k already absorbs k*D), so accumulating the full span is
    #         valid.  After the k-loop, down-sample acc[:, i*S] ONCE with a
    #         tl.split layout op (no arithmetic) and store the contiguous
    #         (BM, BLOCK_OL) tile by block_ptr.  This does S x the minimal MMA
    #         work, but every load/store stays on the DMA fast path and the
    #         strided select is a single cheap layout op per tile.  Measured far
    #         faster than the alternatives on GCU300: an explicit-pointer masked
    #         down-sample store scattered a (BM, BLOCK_N) tile (~8x slower) and
    #         select-BEFORE-dot repeated the reshape+sum select K*n_kblocks times
    #         inside the loop (~5x slower); a single tl.split on the accumulated
    #         tile is the sweet spot.
    if S == 1:
        grid_n = tl.cdiv(OF - pad_left, BLOCK_N)
        total = N * G * grid_m * grid_n
        for tile in tl.range(pid, total, grid_num):
            pid_b = tile // (grid_m * grid_n)
            pid_mn = tile % (grid_m * grid_n)
            pid_m = pid_mn // grid_n
            pid_n = pid_mn % grid_n
            n = pid_b // G
            g = pid_b % G

            x_base = x_ptr + n * stride_xn + g * Cg * stride_xc
            w_base = w_ptr + g * OCg * stride_wm
            o_base = o_ptr + n * stride_on + g * OCg * stride_oc

            # load offset == fol0 + k*D - pad_left == pid_n*BLOCK_N + k*D  (>= 0)
            fol0 = pad_left + pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K):
                xoff = pid_n * BLOCK_N + k * D
                for ci0 in range(0, n_kblocks):
                    a_block_ptr = tl.make_block_ptr(
                        base=w_base + k * Cg,
                        shape=(OCg, Cg),
                        strides=(stride_wm, 1),
                        offsets=(pid_m * BLOCK_M, ci0 * BLOCK_K),
                        block_shape=(BLOCK_M, BLOCK_K),
                        order=(1, 0),
                    )
                    a = tl.load(
                        a_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    b_block_ptr = tl.make_block_ptr(
                        base=x_base,
                        shape=(Cg, L),
                        strides=(stride_xc, 1),
                        offsets=(ci0 * BLOCK_K, xoff),
                        block_shape=(BLOCK_K, BLOCK_N),
                        order=(1, 0),
                    )
                    b = tl.load(
                        b_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

            if HAS_BIAS:
                bias_block_ptr = tl.make_block_ptr(
                    base=b_ptr + g * OCg,
                    shape=(OCg,),
                    strides=(1,),
                    offsets=(pid_m * BLOCK_M,),
                    block_shape=(BLOCK_M,),
                    order=(0,),
                )
                bias = tl.load(
                    bias_block_ptr, boundary_check=(0,), padding_option="zero"
                )
                acc += bias.to(tl.float32)[:, None]

            o_block_ptr = tl.make_block_ptr(
                base=o_base,
                shape=(OCg, OF),
                strides=(stride_oc, 1),
                offsets=(pid_m * BLOCK_M, fol0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    else:
        out_len = (OF - 1) // S + 1
        BLOCK_OL: tl.constexpr = BLOCK_N // S
        BLOCK_NP: tl.constexpr = BLOCK_OL * S
        grid_n = tl.cdiv(out_len - OL0c, BLOCK_OL)
        total = N * G * grid_m * grid_n
        for tile in tl.range(pid, total, grid_num):
            pid_b = tile // (grid_m * grid_n)
            pid_mn = tile % (grid_m * grid_n)
            pid_m = pid_mn // grid_n
            pid_n = pid_mn % grid_n
            n = pid_b // G
            g = pid_b % G

            x_base = x_ptr + n * stride_xn + g * Cg * stride_xc
            w_base = w_ptr + g * OCg * stride_wm
            o_base = o_ptr + n * stride_on + g * OCg * stride_oc

            ol0 = OL0c + pid_n * BLOCK_OL
            acc = tl.zeros((BLOCK_M, BLOCK_NP), dtype=tl.float32)
            for k in range(0, K):
                base_k = ol0 * S + k * D - pad_left  # >= 0 (shifted tiling)
                for ci0 in range(0, n_kblocks):
                    a_block_ptr = tl.make_block_ptr(
                        base=w_base + k * Cg,
                        shape=(OCg, Cg),
                        strides=(stride_wm, 1),
                        offsets=(pid_m * BLOCK_M, ci0 * BLOCK_K),
                        block_shape=(BLOCK_M, BLOCK_K),
                        order=(1, 0),
                    )
                    a = tl.load(
                        a_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    b_block_ptr = tl.make_block_ptr(
                        base=x_base,
                        shape=(Cg, L),
                        strides=(stride_xc, 1),
                        offsets=(ci0 * BLOCK_K, base_k),
                        block_shape=(BLOCK_K, BLOCK_NP),
                        order=(1, 0),
                    )
                    b = tl.load(
                        b_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

            if HAS_BIAS:
                bias_block_ptr = tl.make_block_ptr(
                    base=b_ptr + g * OCg,
                    shape=(OCg,),
                    strides=(1,),
                    offsets=(pid_m * BLOCK_M,),
                    block_shape=(BLOCK_M,),
                    order=(0,),
                )
                bias = tl.load(
                    bias_block_ptr, boundary_check=(0,), padding_option="zero"
                )
                acc += bias.to(tl.float32)[:, None]

            # down-sample acc[:, i*S] -> (BM, BLOCK_OL) with tl.split, a pure
            # layout op (no arithmetic).  S is a host-guarded power of 2; S==2/4
            # use one/two splits, larger powers fall back to a masked sum.
            if S == 2:
                acc_ds, _ = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_OL, 2)))
            elif S == 4:
                acc_q, _ = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_OL, 2, 2)))
                acc_ds, _ = tl.split(acc_q)
            else:
                esel = (tl.arange(0, S)[None, None, :] == 0).to(tl.float32)
                acc_ds = tl.sum(tl.reshape(acc, (BLOCK_M, BLOCK_OL, S)) * esel, axis=2)
            o_block_ptr = tl.make_block_ptr(
                base=o_base,
                shape=(OCg, out_len),
                strides=(stride_oc, 1),
                offsets=(pid_m * BLOCK_M, ol0),
                block_shape=(BLOCK_M, BLOCK_OL),
                order=(1, 0),
            )
            tl.store(
                o_block_ptr, acc_ds.to(o_ptr.dtype.element_ty), boundary_check=(0, 1)
            )

    # ------- edge region: out columns [0, OL0) needing negative positions ---
    # Matrix-vector edge (no ``tl.dot``).  The edge only reads the first few
    # input columns; a tap contributes only when ``pos = ol*S + k*D - pad_left``
    # lands in ``[0, L)``.  Each contributing tap is a strided vector load of
    # ``x[ci, pos]`` broadcast against the weight tile and reduced with
    # ``tl.sum`` -- replacing the big masked gather tile + an N=BLOCK_E
    # ``tl.dot`` (whose N dim is always BLOCK_E even when OL0 is 1, and N<=64
    # dots are "poisoned" on GCU300).  Measured 1.1-2.6x faster than gather+dot
    # on every pad>0 shape.
    #
    # Loop order is ``ol`` (unrolled) OUTSIDE a runtime ``k`` loop whose start
    # ``_k0`` is constexpr.  A tap only contributes when
    # ``pos = ol*S + k*D - pad_left >= 0``, i.e. ``k >= ceil((pad_left-ol*S)/D)``,
    # and every term of that bound is constexpr -- so starting the k loop at _k0
    # means the non-contributing taps are never generated at all.  That
    # elimination, NOT the unrolling, is where the speed is: on the K=7/pad=6
    # shape only 21 of the 42 (k, ol) bodies contribute, and on K=11/pad=5 only
    # 40 of 55.
    #
    # Measured on S60G (BM64/BN1024/BK32, fp16, best of 6 configs):
    #   K11 pad5  runtime range(0,K) 1.294ms/3.6-4.6s compile | static_range(K)
    #             0.967ms/28-47s | this form 0.780ms/4.2-5.2s
    #   K7  pad6  runtime 2.741ms/4.3-5.7s | static_range 1.430ms/13-21s |
    #             this form 1.095ms/5.0-6.1s
    # So the constexpr start beats full unrolling by 1.24-1.31x AND compiles
    # 3-8x faster.  The earlier "rolling the loop gives up almost nothing" claim
    # was wrong -- it cost 1.34-1.92x -- because it kept executing every masked
    # -off tap: a full (BLOCK_M, BLOCK_K) fp32 multiply plus a horizontal
    # ``tl.sum`` per body, which is the expensive part, not the load.
    #
    # ``pos`` stays a runtime value, so the address is clamped into ``[0, L)``
    # and the tap masked off, because it is a negative *pointer* that faults on
    # GCU300, not a masked-off lane.  ``_k0`` already guarantees ``pos >= 0``, so
    # only the upper bound needs guarding -- one ``tl.minimum`` and one mask term
    # instead of two of each.
    #
    # ``OL0c`` stays unrolled -- it is small (<= BLOCK_EDGE) and each ``ol``
    # needs its own scatter into acc_e.  BLOCK_E is ``next_pow2(edge_len)``
    # (host-computed) so the ``(BLOCK_M, BLOCK_E)`` accumulator is tight to the
    # columns actually stored and at most half its lanes are masked surplus;
    # sizing it at the BLOCK_EDGE=64 worst case instead costs 1.01-1.72x.  A
    # single tile covers the whole edge, so there is no edge sub-grid.
    if HAS_EDGE:
        offs_e = tl.arange(0, BLOCK_E)
        total_e = N * G * grid_m
        for tile in tl.range(pid, total_e, grid_num):
            pid_b = tile // grid_m
            pid_m = tile % grid_m
            n = pid_b // G
            g = pid_b % G

            x_base = x_ptr + n * stride_xn + g * Cg * stride_xc
            w_base = w_ptr + g * OCg * stride_wm
            o_base = o_ptr + n * stride_on + g * OCg * stride_oc

            oc = pid_m * BLOCK_M + offs_m
            oc_ok = oc < OCg
            acc_e = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
            for ol in tl.static_range(OL0c):
                # first tap that can land inside x (see the block comment above).
                # Plain assignment, not ``: tl.constexpr`` -- a constexpr-annotated
                # name can not be re-bound on the next static_range iteration.
                _kraw = (pad_left - ol * S + D - 1) // D
                _k0 = _kraw if _kraw > 0 else 0
                for k in range(_k0, K):
                    for ci0 in range(0, n_kblocks):
                        a_block_ptr = tl.make_block_ptr(
                            base=w_base + k * Cg,
                            shape=(OCg, Cg),
                            strides=(stride_wm, 1),
                            offsets=(pid_m * BLOCK_M, ci0 * BLOCK_K),
                            block_shape=(BLOCK_M, BLOCK_K),
                            order=(1, 0),
                        )
                        a = tl.load(
                            a_block_ptr, boundary_check=(0, 1), padding_option="zero"
                        )
                        ci = ci0 * BLOCK_K + offs_k
                        ci_ok = ci < Cg
                        # pos >= 0 is guaranteed by _k0, so only clamp/guard the
                        # upper bound.
                        pos = ol * S + k * D - pad_left
                        pos_c = tl.minimum(pos, L - 1)
                        xv = tl.load(
                            x_base + ci * stride_xc + pos_c,
                            mask=ci_ok & (pos < L),
                            other=0.0,
                        )
                        # reduce in fp32 (matches the old tl.dot out_dtype);
                        # an fp16 tl.sum loses precision for large Cg.
                        col = tl.sum(
                            a.to(tl.float32) * xv[None, :].to(tl.float32), axis=1
                        )
                        acc_e += tl.where(offs_e[None, :] == ol, col[:, None], 0.0)

            if HAS_BIAS:
                bias = tl.load(b_ptr + g * OCg + oc, mask=oc_ok, other=0.0)
                acc_e += bias.to(tl.float32)[:, None]

            o_ptrs = o_base + oc[:, None] * stride_oc + offs_e[None, :]
            tl.store(
                o_ptrs,
                acc_e.to(o_ptr.dtype.element_ty),
                mask=oc_ok[:, None] & (offs_e < OL0)[None, :],
            )


def _resolve_padding(padding, stride, dilation, in_len, kernel_size):
    """Return ``(pad_left, pad_right)`` so no output slicing is needed."""
    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, (
                "Doesn't support any stride values other than 1 "
                f"in padding = 'same' mode, received stride value {stride}"
            )
            sym = math.ceil(
                (stride * (in_len - 1) + 1 + dilation * (kernel_size - 1) - in_len) / 2
            )
            out_full = (
                in_len + 2 * sym - dilation * (kernel_size - 1) - 1
            ) // stride + 1
            trim = out_full - in_len
            pad_left = sym - trim * stride
            pad_right = dilation * (kernel_size - 1) - pad_left
            return pad_left, pad_right
        elif padding == "valid":
            return 0, 0
        raise ValueError(
            f"Unsupported padding mode: {padding}, only 'valid' or 'same' are allowed."
        )
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    return padding, padding


def _conv1d_forward(input, weight, bias, stride, pad_left, pad_right, dilation, groups):
    batch, _, in_len = input.shape
    out_c, in_c_g, kernel_size = weight.shape
    out_c_g = out_c // groups
    # "Full" stride-1 output length; the requested stride is applied by
    # down-sampling in the kernel store (out[ol] = full[ol*stride]).
    out_full = in_len + pad_left + pad_right - dilation * (kernel_size - 1)
    out_len = (out_full - 1) // stride + 1
    # Output columns whose input window starts before x[0]; the shifted tiling
    # of the main loop can not reach them, so the edge loop covers exactly these.
    edge_len = min(-(-pad_left // stride), out_len)

    # (OC, K, Cg) makes the weight A-tile row-major (ci contiguous) for the MMA.
    # The weight is tiny (out_c*in_c_g*kernel_size), so this host-side reshape is
    # not a large-tensor op.
    w = weight.permute(0, 2, 1).contiguous()
    out = torch.empty((batch, out_c, out_len), dtype=input.dtype, device=input.device)
    has_bias = bias is not None
    b = bias.contiguous() if has_bias else weight

    def _total_tiles(meta):
        grid_m = triton.cdiv(out_c_g, meta["BLOCK_M"])
        bn = meta["BLOCK_N"]
        if stride == 1:
            # main tiles the stride-1 "full" axis f in [pad_left, out_full)
            grid_n = triton.cdiv(max(out_full - pad_left, 0), bn)
        else:
            # main tiles the OUTPUT axis in [edge_len, out_len), BLOCK_OL=bn//S wide
            grid_n = triton.cdiv(max(out_len - edge_len, 0), max(bn // stride, 1))
        # the matrix-vector edge is a single tile per (n, g, m-tile)
        grid_e = 1 if edge_len > 0 else 0
        return batch * groups * grid_m * (grid_n + grid_e)

    grid = lambda META: (  # noqa: E731
        min(triton.cdiv(MAX_GRID_DIM, META["num_warps"]), _total_tiles(META)),
    )
    with torch_device_fn.device(input.device):
        conv1d_fwd_kernel[grid](
            input,
            w,
            b,
            out,
            batch,
            out_full,
            edge_len,
            out_c_g,
            groups,
            in_len,
            pad_left,
            stride_xn=input.stride(0),
            stride_xc=input.stride(1),
            stride_wm=w.stride(0),
            stride_on=out.stride(0),
            stride_oc=out.stride(1),
            Cg=in_c_g,
            K=kernel_size,
            S=stride,
            D=dilation,
            HAS_BIAS=has_bias,
            HAS_EDGE=edge_len > 0,
            GRID_DIM=MAX_GRID_DIM,
            # Size the edge accumulator to the edge width actually needed rather
            # than the BLOCK_EDGE worst case.  Only ``edge_len`` columns are ever
            # stored (mask ``offs_e < OL0``), and ``BLOCK_E >= edge_len`` is
            # exactly what the ``tl.where(offs_e == ol, ...)`` scatter needs to
            # land every live column, so next_pow2(edge_len) is the safe minimum.
            # Measured 1.01-1.72x faster on every pad>0 shape with bit-identical
            # results (same fp32 reduction order): acc_e shrinks from
            # (BLOCK_M, 64) to (BLOCK_M, 1..8), cutting the per-tap where+add
            # vector work by 8-64x.  Largest gain where the edge is narrowest
            # relative to BLOCK_EDGE, e.g. OL0c=2 fp32 0.599 -> 0.411ms (1.46x).
            BLOCK_E=max(triton.next_power_of_2(edge_len), 1),
        )
    return out


class _Conv1dFn(torch.autograd.Function):
    """Fast Triton forward; backward reuses the generic differentiable path."""

    @staticmethod
    def forward(
        ctx, input, weight, bias, stride, pad_left, pad_right, dilation, groups
    ):
        out = _conv1d_forward(
            input, weight, bias, stride, pad_left, pad_right, dilation, groups
        )
        ctx.save_for_backward(input, weight, bias)
        ctx.meta = (stride, pad_left, pad_right, dilation, groups)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from flag_gems.ops.conv1d import conv1d as conv1d_default

        input, weight, bias = ctx.saved_tensors
        stride, pad_left, pad_right, dilation, groups = ctx.meta
        with torch.enable_grad():
            i2 = input.detach().requires_grad_(True)
            w2 = weight.detach().requires_grad_(True)
            b2 = bias.detach().requires_grad_(True) if bias is not None else None
            ref = conv1d_default(i2, w2, b2, stride, pad_left, dilation, groups)
            inputs = [i2, w2] if b2 is None else [i2, w2, b2]
            grads = torch.autograd.grad(ref, inputs, grad_output)
        grad_bias = None if bias is None else grads[2]
        return grads[0], grads[1], grad_bias, None, None, None, None, None


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_ENFLAME_GCU300 CONV1D")
    stride_w = stride[0] if isinstance(stride, (list, tuple)) else stride
    dilation_w = dilation[0] if isinstance(dilation, (list, tuple)) else dilation

    if isinstance(padding, str) and padding == "same":
        # keep the generic assertion semantics (compares the raw argument)
        assert stride == 1, (
            "Doesn't support any stride values other than 1 "
            f"in padding = 'same' mode, received stride value {stride}"
        )

    if input.dim() == 2:
        out = conv1d(
            input.unsqueeze(0), weight, bias, stride, padding, dilation, groups
        )
        return out.squeeze(0)

    pad_left, pad_right = _resolve_padding(
        padding, stride_w, dilation_w, input.shape[-1], weight.shape[-1]
    )

    out_c, in_c_g, kernel_size = weight.shape
    in_len = input.shape[-1]
    out_full = in_len + pad_left + pad_right - dilation_w * (kernel_size - 1)
    out_len = (out_full - 1) // stride_w + 1
    supported = (
        input.dim() == 3
        and weight.dim() == 3
        and out_len > 0
        and in_c_g * groups == input.shape[1]
        and out_c % groups == 0
        and pad_left >= 0
        and pad_right >= 0
        # the matrix-vector edge covers [0, ceil(pad_left/stride)) in ONE tile
        # unrolled by ``tl.static_range``; keep it small so the tile fits in
        # BLOCK_E and the unroll does not explode.  Larger padding is rare and
        # falls back to the generic path.
        and -(-pad_left // stride_w) <= BLOCK_EDGE
        # the stride>1 down-sample uses tl.split / tl.reshape over the stride
        # axis, which require a power-of-2 stride; other strides (3,5,...) fall
        # back to the generic path (benchmark/tests only use S in {1,2,4}).
        and (stride_w & (stride_w - 1)) == 0
        # the DMA fast path needs a unit-stride spatial axis
        and input.stride(2) == 1
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not supported:
        from flag_gems.ops.conv1d import conv1d as conv1d_default

        return conv1d_default(input, weight, bias, stride, padding, dilation, groups)

    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return _Conv1dFn.apply(
            input, weight, bias, stride_w, pad_left, pad_right, dilation_w, groups
        )
    return _conv1d_forward(
        input, weight, bias, stride_w, pad_left, pad_right, dilation_w, groups
    )
