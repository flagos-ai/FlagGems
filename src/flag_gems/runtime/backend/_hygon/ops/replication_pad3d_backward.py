import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single fused kernel: every program computes a (BLOCK_H x BLOCK_W) tile of
# one output plane.
#   - Fast interior path: rows in [1, H-2] via a vectorized 2D tile; W-boundary
#     box sums from two tiny vector loads (LSP/RSP lanes) instead of a full
#     next-pow2(W_out) tile (avoids ~45% masked-lane waste).
#   - Boundary paths: rows h==0 and h==H-1 via the same tiny-load scheme,
#     vectorized over the H-box span (HSP rows): per dd plane, [HSP x LSP] and
#     [HSP x RSP] loads for the row/col corner sums plus a [HSP x BLOCK_W]
#     shifted load reduced over the H span. Uniform scalar branches select
#     them only for the first/last h-tile.
# All accumulation in fp32; single kernel launch. NO_THM is a host-computed
# compile-time flag: when pad_top/pad_bottom are non-negative, interior rows
# h in [1, H-2] always map into [0, H_out), so the per-row H validity mask is
# compiled out of the fast path.
# ---------------------------------------------------------------------------
@triton.jit
def _rep_pad3d_bwd_fused(
    go,
    out,
    W,
    H,
    D,
    W_out,
    H_out,
    D_out,
    left,
    top,
    front,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    LSP: tl.constexpr,
    RSP: tl.constexpr,
    HSP: tl.constexpr,
    NO_THM: tl.constexpr,
    NO_WM: tl.constexpr,
    DT: tl.constexpr,
    ODT: tl.constexpr,
):
    pid_plane = tl.program_id(0)
    d = tl.program_id(1)
    h0 = tl.program_id(2) * BLOCK_H

    w = tl.arange(0, BLOCK_W)
    wmask = w < W
    ls_idx = tl.arange(0, LSP)
    rs_idx = tl.arange(0, RSP)
    hh_idx = tl.arange(0, HSP)

    if NO_WM:
        wmask_t = tl.full([BLOCK_W], 1, tl.int1)  # all-true; W pow2 and pads >= 0
    else:
        wmask_t = wmask

    d_lo = tl.maximum(tl.where(d == 0, 0, d + front), 0)
    d_hi = tl.minimum(tl.where(d == D - 1, D_out - 1, d + front), D_out - 1)
    n_ls = tl.maximum(tl.minimum(left + 1, W_out), 0)
    rs_start = tl.maximum(left + W - 1, 0)
    n_rs = tl.maximum(W_out - rs_start, 0)

    plane_stride = D_out * H_out * W_out
    out_plane_stride = D * H * W

    # ---------------- fast interior path: rows in [1, H-2] ----------------
    h = h0 + tl.arange(0, BLOCK_H)
    hm = (h >= 1) & (h < H - 1)
    th = h + top
    if NO_THM:
        hmask = hm[:, None]
    else:
        thm = (th >= 0) & (th < H_out)
        hmask = hm[:, None] & thm[:, None]

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=DT)
    for dd in tl.range(d_lo, d_hi + 1):
        base = go + pid_plane * plane_stride + dd * (H_out * W_out)
        rowb = base + th[:, None] * W_out
        lt = tl.load(
            rowb + ls_idx[None, :], mask=hmask & (ls_idx[None, :] < n_ls), other=0.0
        ).to(DT)
        ls = tl.sum(lt, axis=1)  # [BLOCK_H]
        rt = tl.load(
            rowb + (rs_start + rs_idx)[None, :],
            mask=hmask & (rs_idx[None, :] < n_rs),
            other=0.0,
        ).to(DT)
        rs = tl.sum(rt, axis=1)  # [BLOCK_H]
        sv = tl.load(
            rowb + (w + left)[None, :],
            mask=hmask
            & wmask_t[None, :]
            & ((w + left)[None, :] >= 0)
            & ((w + left)[None, :] < W_out),
            other=0.0,
        ).to(DT)
        c = tl.where(
            w[None, :] == 0, ls[:, None], tl.where(w[None, :] == W - 1, rs[:, None], sv)
        )
        c = tl.where((W == 1) & (w[None, :] == 0), ls[:, None] + rs[:, None] - sv, c)
        acc += c

    outb = out + pid_plane * out_plane_stride + d * (H * W)
    tl.store(
        outb + h[:, None] * W + w[None, :], acc.to(ODT), mask=hmask & wmask_t[None, :]
    )

    # ---------------- boundary row h == 0 (first tile only) ----------------
    if h0 == 0:
        # box: h' in [0, h_hi0]; general formula handles H == 1 (full box)
        h_hi0 = tl.minimum(H_out - 1, tl.where(H == 1, H_out - 1, top))
        n_hh0 = h_hi0 + 1
        bacc = tl.zeros([BLOCK_W], dtype=DT)
        for dd in tl.range(d_lo, d_hi + 1):
            base = go + pid_plane * plane_stride + dd * (H_out * W_out)
            rowb = base + hh_idx[:, None] * W_out
            hhm = hh_idx[:, None] < n_hh0
            lt = tl.load(
                rowb + ls_idx[None, :], mask=hhm & (ls_idx[None, :] < n_ls), other=0.0
            ).to(DT)
            ls0 = tl.sum(lt)
            rt = tl.load(
                rowb + (rs_start + rs_idx)[None, :],
                mask=hhm & (rs_idx[None, :] < n_rs),
                other=0.0,
            ).to(DT)
            rs0 = tl.sum(rt)
            svt = tl.load(
                rowb + (w + left)[None, :],
                mask=hhm
                & wmask_t[None, :]
                & ((w + left)[None, :] >= 0)
                & ((w + left)[None, :] < W_out),
                other=0.0,
            ).to(DT)
            sv0 = tl.sum(svt, axis=0)  # [BLOCK_W]
            c0 = tl.where(w == 0, ls0, tl.where(w == W - 1, rs0, sv0))
            c0 = tl.where((W == 1) & (w == 0), ls0 + rs0 - sv0, c0)
            bacc += c0
        tl.store(outb + w, bacc.to(ODT), mask=wmask_t)

    # ---------------- boundary row h == H-1 (last tile only) ----------------
    if h0 + BLOCK_H >= H:
        h_lo1 = tl.maximum(0, tl.where(H == 1, 0, H - 1 + top))
        n_hh1 = H_out - h_lo1
        bacc1 = tl.zeros([BLOCK_W], dtype=DT)
        for dd in tl.range(d_lo, d_hi + 1):
            base = go + pid_plane * plane_stride + dd * (H_out * W_out)
            rowb = base + (h_lo1 + hh_idx)[:, None] * W_out
            hhm = hh_idx[:, None] < n_hh1
            lt = tl.load(
                rowb + ls_idx[None, :], mask=hhm & (ls_idx[None, :] < n_ls), other=0.0
            ).to(DT)
            ls1 = tl.sum(lt)
            rt = tl.load(
                rowb + (rs_start + rs_idx)[None, :],
                mask=hhm & (rs_idx[None, :] < n_rs),
                other=0.0,
            ).to(DT)
            rs1 = tl.sum(rt)
            svt = tl.load(
                rowb + (w + left)[None, :],
                mask=hhm
                & wmask_t[None, :]
                & ((w + left)[None, :] >= 0)
                & ((w + left)[None, :] < W_out),
                other=0.0,
            ).to(DT)
            sv1 = tl.sum(svt, axis=0)  # [BLOCK_W]
            c1 = tl.where(w == 0, ls1, tl.where(w == W - 1, rs1, sv1))
            c1 = tl.where((W == 1) & (w == 0), ls1 + rs1 - sv1, c1)
            bacc1 += c1
        tl.store(outb + (H - 1) * W + w, bacc1.to(ODT), mask=wmask_t)


def run(grad_output, self, padding):
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()
    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError(
            "padding must be a sequence of length 6: "
            "(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)"
        )
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = map(int, padding)
    if self.dim() < 3:
        raise ValueError("self must have at least 3 dimensions (D, H, W)")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "replication_pad3d_backward supports float16, bfloat16, and float32"
        )

    go = grad_output.contiguous()
    x = self.contiguous()

    D_in = int(x.shape[-3])
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if D_in <= 0 or H_in <= 0 or W_in <= 0:
        raise ValueError("input volumetric dims (D, H, W) must all be > 0")

    D_out = D_in + pad_front + pad_back
    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right
    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        raise ValueError("padding results in a non-positive output dimension")

    expected = (D_out, H_out, W_out)
    actual = (int(go.shape[-3]), int(go.shape[-2]), int(go.shape[-1]))
    if actual != expected:
        raise ValueError(
            f"grad_output spatial shape {actual} does not match expected {expected}."
        )
    if tuple(go.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    leading = x.shape[:-3]
    B = math.prod(leading) if len(leading) > 0 else 1

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    BLOCK_W = triton.next_power_of_2(max(W_in, 1))
    if W_in <= 16:
        BLOCK_H = 8
        num_warps = 8
    elif D_in <= 16:
        BLOCK_H = 16
        num_warps = 1
    else:
        BLOCK_H = 16
        num_warps = 2
    n_ls = max(min(pad_left + 1, W_out), 0)
    rs_start = max(pad_left + W_in - 1, 0)
    n_rs = max(W_out - rs_start, 0)
    LSP = triton.next_power_of_2(max(n_ls, 1))
    RSP = triton.next_power_of_2(max(n_rs, 1))
    h_hi0 = min(H_out - 1, H_out - 1 if H_in == 1 else pad_top)
    h_lo1 = max(0, 0 if H_in == 1 else (H_in - 1) + pad_top)
    n_hh = max(h_hi0 + 1, H_out - h_lo1, 1)
    HSP = triton.next_power_of_2(n_hh)
    NO_THM = (pad_top >= 0) and (pad_bottom >= 0)
    NO_WM = (pad_left >= 0) and (pad_right >= 0) and (W_in & (W_in - 1) == 0)
    DT = tl.float32
    ODT = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[x.dtype]

    grid = (B, D_in, (H_in + BLOCK_H - 1) // BLOCK_H)
    _rep_pad3d_bwd_fused[grid](
        go,
        out,
        W_in,
        H_in,
        D_in,
        W_out,
        H_out,
        D_out,
        pad_left,
        pad_top,
        pad_front,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        LSP=LSP,
        RSP=RSP,
        HSP=HSP,
        NO_THM=NO_THM,
        NO_WM=NO_WM,
        DT=DT,
        ODT=ODT,
        num_warps=num_warps,
    )
    return out


# Alias for FlagGems import convention
replication_pad3d_backward = run
