import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name

# ---------------------------------------------------------------------------
# Separable trilinear upsampling via tl.dot.
#
# 3D trilinear interpolation is separable into three independent 1D linear
# interpolations (along W, H, D). Each 1D interpolation along an axis is a
# small (out, in) weight matrix applied along that axis, i.e. a matmul. The
# naive per-output-element kernel is bound by data-dependent gather throughput
# (~0.017x-0.9x vs torch). Recasting W/H as tl.dot matmuls turns the gather
# into dense contiguous compute on the matmul units; D (tiny extent) stays a
# contiguous 2-neighbor lerp. Everything is a Triton kernel, so it is immune
# to the use_gems aten interception that penalizes torch.matmul internals.
#
# The three interpolation matrices depend only on (in, out, align_corners,
# scale); they are built once (fp32) and cached. All accumulation is fp32.
# ---------------------------------------------------------------------------

_WEIGHT_CACHE = {}


@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[K, N], acc in fp32 (A loaded then upcast).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_m[:, None] * sam + offs_k[None, :] * sak)
    b_ptrs = b_ptr + (offs_k[:, None] * sbk + offs_n[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = offs_k[None, :] < K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & km, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + scm * offs_m[:, None] + scn * offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _lbmm_kernel(
    w_ptr, x_ptr, c_ptr, OUT, K, COLS,
    swo, swk, sxb, sxk, sxc, scb, sco, scc,
    BO: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    # C[b, OUT, COLS] = W[OUT, K] @ X[b, K, COLS], W shared across batch b.
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_o = pid_o * BO + tl.arange(0, BO)
    offs_c = pid_c * BC + tl.arange(0, BC)
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + (offs_o[:, None] * swo + offs_k[None, :] * swk)
    x_ptrs = x_ptr + pid_b * sxb + (offs_k[:, None] * sxk + offs_c[None, :] * sxc)
    acc = tl.zeros((BO, BC), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kk = K - k * BK
        w = tl.load(w_ptrs, mask=(offs_o[:, None] < OUT) & (offs_k[None, :] < kk), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < kk) & (offs_c[None, :] < COLS), other=0.0).to(tl.float32)
        acc += tl.dot(w, x, allow_tf32=False)
        w_ptrs += BK * swk
        x_ptrs += BK * sxk
    c_ptrs = c_ptr + pid_b * scb + sco * offs_o[:, None] + scc * offs_c[None, :]
    tl.store(c_ptrs, acc, mask=(offs_o[:, None] < OUT) & (offs_c[None, :] < COLS))


@triton.jit
def _dlerp_kernel(
    y_ptr, o_ptr, id0_ptr, id1_ptr, w0_ptr, w1_ptr,
    OD, P, ID, BP: tl.constexpr,
):
    # out[n, od, p] = w0[od]*y[n, id0[od], p] + w1[od]*y[n, id1[od], p]
    pid_nod = tl.program_id(0)
    pid_p = tl.program_id(1)
    n = pid_nod // OD
    od = pid_nod % OD
    id0 = tl.load(id0_ptr + od)
    id1 = tl.load(id1_ptr + od)
    w0 = tl.load(w0_ptr + od)
    w1 = tl.load(w1_ptr + od)
    offs = pid_p * BP + tl.arange(0, BP)
    m = offs < P
    base = n * ID * P
    a = tl.load(y_ptr + base + id0 * P + offs, mask=m, other=0.0)
    b = tl.load(y_ptr + base + id1 * P + offs, mask=m, other=0.0)
    out = a * w0 + b * w1
    tl.store(o_ptr + pid_nod * P + offs, out.to(o_ptr.dtype.element_ty), mask=m)


def _build_weights(in_sz, out_sz, align_corners, scale, dev):
    key = (in_sz, out_sz, align_corners, scale, str(dev))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    o = torch.arange(out_sz, device=dev, dtype=torch.float32)
    if align_corners:
        scale_val = (in_sz - 1.0) / (out_sz - 1.0) if out_sz > 1 else 0.0
        src = o * scale_val
    else:
        real_scale = (1.0 / scale) if scale is not None else (in_sz / out_sz)
        src = o * real_scale + (0.5 * real_scale - 0.5)
    src = torch.clamp(src, min=0.0, max=in_sz - 1.0)
    i0 = src.to(torch.int32)
    i1 = torch.clamp(i0 + 1, max=in_sz - 1)
    t = src - i0.to(torch.float32)
    w = torch.zeros(out_sz, in_sz, device=dev, dtype=torch.float32)
    rows = torch.arange(out_sz, device=dev)
    w[rows, i0.long()] += 1.0 - t
    w[rows, i1.long()] += t
    entry = (w, i0.contiguous(), i1.contiguous(), (1.0 - t).contiguous(), t.contiguous())
    _WEIGHT_CACHE[key] = entry
    return entry


def upsample_trilinear3d(
    self: torch.Tensor,
    output_size: Tuple[int, int, int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_TRILINEAR3D")
    assert self.device.type == device
    assert self.ndim == 5, f"Input must be 5D (NCDHW), got {self.ndim}D"

    N, C, ID, IH, IW = self.shape
    OD, OH, OW = output_size
    NC = N * C

    out = torch.empty((N, C, OD, OH, OW), device=self.device, dtype=self.dtype)
    if out.numel() == 0:
        return out

    dev = self.device
    wd, id0, id1, wd0, wd1 = _build_weights(ID, OD, align_corners, scales_d, dev)
    wh = _build_weights(IH, OH, align_corners, scales_h, dev)[0]
    ww = _build_weights(IW, OW, align_corners, scales_w, dev)[0]

    with torch_device_fn.device(dev):
        # W-pass: y1[NC*ID*IH, OW] = x[NC*ID*IH, IW] @ Ww^T[IW, OW]
        xin = self.reshape(NC * ID * IH, IW).contiguous()
        wwt = ww.t().contiguous()
        M1 = NC * ID * IH
        y1 = torch.empty((M1, OW), device=dev, dtype=torch.float32)
        BM, BN, BK = 64, 64, 32
        _mm_kernel[(triton.cdiv(M1, BM), triton.cdiv(OW, BN))](
            xin, wwt, y1, M1, OW, IW,
            xin.stride(0), xin.stride(1), wwt.stride(0), wwt.stride(1),
            y1.stride(0), y1.stride(1), BM=BM, BN=BN, BK=BK,
        )

        # H-pass: y2[NC*ID, OH, OW] = Wh[OH, IH] @ y1[NC*ID, IH, OW]
        y1b = y1.reshape(NC * ID, IH, OW)
        y2 = torch.empty((NC * ID, OH, OW), device=dev, dtype=torch.float32)
        BO, BC, BKh = 64, 64, 32
        _lbmm_kernel[(NC * ID, triton.cdiv(OH, BO), triton.cdiv(OW, BC))](
            wh, y1b, y2, OH, IH, OW,
            wh.stride(0), wh.stride(1),
            y1b.stride(0), y1b.stride(1), y1b.stride(2),
            y2.stride(0), y2.stride(1), y2.stride(2),
            BO=BO, BC=BC, BK=BKh,
        )

        # D-pass: out[N, C, OD, OH, OW] = lerp over ID of y2[NC, ID, OH*OW]
        y2p = y2.reshape(NC, ID, OH * OW)
        P = OH * OW
        BP = 1024
        _dlerp_kernel[(NC * OD, triton.cdiv(P, BP))](
            y2p, out, id0, id1, wd0, wd1, OD, P, ID, BP=BP,
        )

    return out
