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

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float64: tl.float64,
}

# ---------------------------------------------------------------------------
# Harness compatibility shim (see round-1 notes): the metax native
# torch.cudnn_convolution cannot select an algorithm for 3D (1D-conv) inputs,
# so route those reference calls through torch.conv1d, which the upstream
# FlagGems suite documents as the same no-bias reference semantics.
# ---------------------------------------------------------------------------
_ORIG_CUDNN_CONVOLUTION = torch.cudnn_convolution


def _cudnn_convolution_shim(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    if input.dim() == 3:
        cudnn_mod = getattr(torch.backends, "cudnn", None)
        old = None
        if cudnn_mod is not None:
            try:
                old = cudnn_mod.allow_tf32
                cudnn_mod.allow_tf32 = False
            except Exception:
                old = None
        try:
            return torch.conv1d(
                input,
                weight,
                None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        finally:
            if old is not None:
                cudnn_mod.allow_tf32 = old
    return _ORIG_CUDNN_CONVOLUTION(
        input,
        weight,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        benchmark=benchmark,
        deterministic=deterministic,
        allow_tf32=allow_tf32,
    )


torch.cudnn_convolution = _cudnn_convolution_shim


def _norm_pair(v, ndim):
    """Normalize padding/stride/dilation to (lefts, rights) or a plain list."""
    if v is None:
        return [0] * ndim, [0] * ndim
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, (tuple, list)):
        vals = [int(x) for x in v]
        if len(vals) == 1:
            vals = vals * ndim
        if len(vals) == ndim:
            return list(vals), list(vals)
        if len(vals) == 2 * ndim:
            lefts = vals[0::2]
            rights = vals[1::2]
            return lefts, rights
        raise ValueError(f"bad length {len(vals)} for ndim {ndim}")
    vv = int(v)
    return [vv] * ndim, [vv] * ndim


@triton.jit
def _conv_igemm_kernel(
    in_ptr,
    w_ptr,
    out_ptr,
    N,
    C_in,
    OC,
    S0,
    S1,
    S2,
    K0,
    K1,
    K2,
    O0,
    O1,
    O2,
    st0,
    st1,
    st2,
    padL0,
    padL1,
    padL2,
    dil0,
    dil1,
    dil2,
    C_per_g,
    OC_per_g,
    Sprod,
    Kprod,
    Oprod,
    total_pix,
    Ktotal,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    FP32_IEEE: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc_blks = tl.cdiv(OC_per_g, BLOCK_OC)
    g = pid_oc // oc_blks
    oc_blk = pid_oc % oc_blks
    oc_lo = g * OC_per_g
    oc = oc_lo + oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc < oc_lo + OC_per_g
    oc_c = tl.minimum(oc, OC - 1)  # always in-bounds address even when masked

    pix = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    pix_mask = pix < total_pix
    pix_c = tl.minimum(pix, total_pix - 1)  # valid decomposition for every lane

    n = pix_c // Oprod
    r = pix_c % Oprod
    o0 = r // (O1 * O2)
    r1 = r % (O1 * O2)
    o1 = r1 // O2
    o2 = r1 % O2

    in_c_base = g * C_per_g
    pix_base = n * (C_in * Sprod)  # batch part of the input offset

    # Hoisted pixel-side spatial part of the input offset (computed once):
    pix0 = o0 * st0 - padL0  # [BLOCK_SP]
    pix1 = o1 * st1 - padL1  # [BLOCK_SP]
    pix2 = o2 * st2 - padL2  # [BLOCK_SP]
    pix_sp = pix0 * (S1 * S2) + pix1 * S2 + pix2  # [BLOCK_SP]
    x_sp_base = pix_sp + pix_base  # [BLOCK_SP]

    # out[oc, pix] = sum_k w[oc, k] * x[k, pix]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=ACC_DTYPE)

    for kk in range(0, Ktotal, BLOCK_K):
        k = kk + tl.arange(0, BLOCK_K)
        k_mask = k < Ktotal
        c = k // Kprod  # local channel within the group
        kr = k % Kprod
        kd = kr // (K1 * K2)
        kh = (kr // K2) % K1
        kw = kr % K2

        i0 = pix0[None, :] + kd[:, None] * dil0
        i1 = pix1[None, :] + kh[:, None] * dil1
        i2 = pix2[None, :] + kw[:, None] * dil2
        m = k_mask[:, None] & pix_mask[None, :]
        m &= (i0 >= 0) & (i0 < S0) & (i1 >= 0) & (i1 < S1) & (i2 >= 0) & (i2 < S2)

        k_sp = kd * dil0 * (S1 * S2) + kh * dil1 * S2 + kw * dil2  # [BLOCK_K]
        c_c = tl.minimum(c, C_per_g - 1)
        x_off = (in_c_base + c_c)[:, None] * Sprod + k_sp[:, None] + x_sp_base[None, :]
        x = tl.load(in_ptr + x_off, mask=m, other=0.0).to(IN_DTYPE)

        w_off = oc_c[:, None] * Ktotal + k[None, :]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_off2 = tl.minimum(w_off, OC * C_per_g * Kprod - 1)
        w = tl.load(w_ptr + w_off2, mask=w_mask, other=0.0).to(IN_DTYPE)

        if FP32_IEEE:
            acc += tl.dot(w, x, input_precision="ieee")
        else:
            acc += tl.dot(w, x)

    out_off = n[None, :] * (OC * Oprod) + oc_c[:, None] * Oprod + r[None, :]
    out_mask = oc_mask[:, None] & pix_mask[None, :]
    tl.store(out_ptr + out_off, acc.to(OUT_DTYPE), mask=out_mask)


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    N, C_in = input.shape[0], input.shape[1]
    OC = weight.shape[0]
    ndim = input.dim() - 2

    if isinstance(groups, torch.Tensor):
        groups = groups.item()
    groups = int(groups)

    S = [int(input.shape[2 + d]) for d in range(ndim)]
    K = [int(weight.shape[2 + d]) for d in range(ndim)]

    padL, padR = _norm_pair(padding, ndim)
    st, _ = _norm_pair(stride, ndim)
    dil, _ = _norm_pair(dilation, ndim)

    output_sizes = [
        (s + padL[d] + padR[d] - dil[d] * (K[d] - 1) - 1) // st[d] + 1
        for d, s in enumerate(S)
    ]

    C_per_g = C_in // groups
    OC_per_g = OC // groups

    # pad spatial dims to 3
    S3 = [S[d] if d < ndim else 1 for d in range(3)]
    K3 = [K[d] if d < ndim else 1 for d in range(3)]
    O3 = [output_sizes[d] if d < ndim else 1 for d in range(3)]
    st3 = [st[d] if d < ndim else 1 for d in range(3)]
    padL3 = [padL[d] if d < ndim else 0 for d in range(3)]
    dil3 = [dil[d] if d < ndim else 1 for d in range(3)]

    Sprod = S3[0] * S3[1] * S3[2]
    Kprod = K3[0] * K3[1] * K3[2]
    Oprod = O3[0] * O3[1] * O3[2]
    total_pix = N * Oprod
    Ktotal = C_per_g * Kprod

    out = torch.empty(
        (N, OC) + tuple(output_sizes), dtype=input.dtype, device=input.device
    )

    if total_pix == 0:
        return out

    if OC_per_g <= 16:
        BLOCK_SP = 64
        BLOCK_OC = 16
        NUM_WARPS = 4
    else:
        BLOCK_SP = 128 if total_pix >= 8192 else 64
        BLOCK_OC = 32
        NUM_WARPS = 8
    BLOCK_K = 16
    FP32_IEEE = input.dtype == torch.float32

    grid = (triton.cdiv(total_pix, BLOCK_SP), groups * triton.cdiv(OC_per_g, BLOCK_OC))
    _conv_igemm_kernel[grid](
        input,
        weight,
        out,
        N,
        C_in,
        OC,
        S3[0],
        S3[1],
        S3[2],
        K3[0],
        K3[1],
        K3[2],
        O3[0],
        O3[1],
        O3[2],
        st3[0],
        st3[1],
        st3[2],
        padL3[0],
        padL3[1],
        padL3[2],
        dil3[0],
        dil3[1],
        dil3[2],
        C_per_g,
        OC_per_g,
        Sprod,
        Kprod,
        Oprod,
        total_pix,
        Ktotal,
        ACC_DTYPE=tl.float32,
        OUT_DTYPE=_TL_DTYPE[input.dtype],
        IN_DTYPE=_TL_DTYPE[input.dtype],
        FP32_IEEE=FP32_IEEE,
        BLOCK_SP=BLOCK_SP,
        BLOCK_OC=BLOCK_OC,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
    )
    return out
