import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _rep_pad3d_bwd_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total_in,        # D_in * H_in * W_in (per batch volume)
    D_in,
    H_in,
    W_in,
    pad_left,
    pad_right,
    pad_top,
    pad_bottom,
    pad_front,
    pad_back,
    D_out,
    H_out,
    W_out,
    BLOCK: tl.constexpr,
    MAXD: tl.constexpr,
    MAXH: tl.constexpr,
    MAXW_L: tl.constexpr,
    MAXW_R: tl.constexpr,
):
    """Atomic-free replication_pad3d_backward (flat gather, wide blocks).

    The forward maps each output voxel (d', h', w') to the input voxel
    ``(clamp(d' - pad_front), clamp(h' - pad_top), clamp(w' - pad_left))``.
    In the backward, each *input* voxel (d, h, w) is the sum of grad_output
    over the box of output voxels that clamp to it::

        grad_input[d, h, w] = sum_{d' in [dlo,dhi], h' in [hlo,hhi],
                                     w' in [wlo(w), whi(w)]} grad_output[d', h', w']

    Each program owns ``BLOCK`` contiguous input voxels, decomposes them into
    (d, h, w), and gathers + reduces their source boxes locally.  Every input
    voxel is written by exactly one program, so no atomic and no lost updates.

    The per-axis source intervals are (clamp-inverse):
      - interior: single source at ``coord + pad``
      - low boundary (coord == 0, pad > 0): ``[0, pad]``
      - high boundary (coord == C_in-1): ``[pad + C_in - 1, L_out - 1]``
      - negative pad that empties the interval -> zero contribution

    XPU backend notes (mirrors replication_pad2d_backward):
      - ``tl.atomic_add`` silently drops updates -> atomic-free by design.
      - masked loads with ``other=0.0`` read REAL memory for masked lanes,
        so a masked lane value can leak into a sum.
      - vector reductions (``tl.sum``) inside a runtime loop can drop lanes.
      - memory throughput collapses for blocks narrower than ~256 lanes
        (measured ~3 GB/s at BLOCK=64 vs ~148 GB/s at BLOCK=1024) and
        sub-64-wide stores are also unreliable (program_id / lane scramble).
      Hence this kernel always uses a wide flat block (BLOCK=1024), every load
      uses a clamped in-bounds address, and every invalid contribution is
      zeroed with a register-level ``tl.where`` BEFORE it feeds the
      accumulator.  No masked-load result, no vector-reduction result, and no
      narrow-block store ever contributes to a result.
    """
    pid_b = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    offs = pid_chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_in

    # ---- decompose flat input index -> (d, h, w) ----
    w = offs % W_in
    h = (offs // W_in) % H_in
    d = offs // (H_in * W_in)

    # ---- d source interval [dlo, dhi] ----
    dlo_raw = tl.where((d == 0) & (pad_front > 0), 0, d + pad_front)
    dhi_raw = tl.where(d == D_in - 1, D_out - 1, d + pad_front)
    dlo = tl.maximum(dlo_raw, 0)
    dhi = tl.minimum(dhi_raw, D_out - 1)
    cd = dhi - dlo + 1

    # ---- h source interval [hlo, hhi] ----
    hlo_raw = tl.where((h == 0) & (pad_top > 0), 0, h + pad_top)
    hhi_raw = tl.where(h == H_in - 1, H_out - 1, h + pad_top)
    hlo = tl.maximum(hlo_raw, 0)
    hhi = tl.minimum(hhi_raw, H_out - 1)
    ch = hhi - hlo + 1

    # ---- w sources: main + boundary corrections ----
    wmain = w + pad_left
    wmain_c = tl.minimum(tl.maximum(wmain, 0), tl.maximum(W_out - 1, 0))
    wmain_ok = (wmain >= 0) & (wmain < W_out)
    wfirst = w == 0
    wlast = w == W_in - 1

    out_base = pid_b * D_out * H_out * W_out
    in_base = pid_b * total_in

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for di in tl.static_range(MAXD):
        dd = dlo + di
        dd_c = tl.minimum(dd, D_out - 1)
        dd_ok = di < cd
        for hi in tl.static_range(MAXH):
            hh = hlo + hi
            hh_c = tl.minimum(hh, H_out - 1)
            hh_ok = hi < ch
            row_ok = mask & dd_ok & hh_ok
            row_base = dd_c * H_out * W_out + hh_c * W_out

            # main: row[w + pad_left]
            v = tl.load(grad_output_ptr + out_base + row_base + wmain_c)
            acc += tl.where(row_ok & wmain_ok, v.to(tl.float32), 0.0)

            # left boundary: row[0 .. pad_left-1] feeds w == 0
            for wl in tl.static_range(MAXW_L):
                v = tl.load(grad_output_ptr + out_base + row_base + tl.minimum(wl, tl.maximum(W_out - 1, 0)))
                sel = row_ok & wfirst & (wl < pad_left) & (wl < W_out)
                acc += tl.where(sel, v.to(tl.float32), 0.0)

            # right boundary: row[W_out-pad_right .. W_out-1] feeds w == W_in-1
            for wr in tl.static_range(MAXW_R):
                rpos = W_out - pad_right + wr
                rpos_c = tl.minimum(tl.maximum(rpos, 0), tl.maximum(W_out - 1, 0))
                v = tl.load(grad_output_ptr + out_base + row_base + rpos_c)
                sel = row_ok & wlast & (wr < pad_right) & (rpos >= 0) & (rpos < W_out)
                acc += tl.where(sel, v.to(tl.float32), 0.0)

    tl.store(
        grad_input_ptr + in_base + offs,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _axis_interval(coord, cin, cout, pad_lo, pad_hi):
    """Source interval for one input coord on one axis (host-side, matches kernel)."""
    lo_raw = 0 if (coord == 0 and pad_lo > 0) else coord + pad_lo
    hi_raw = cout - 1 if coord == cin - 1 else coord + pad_lo
    lo = max(lo_raw, 0)
    hi = min(hi_raw, cout - 1)
    return hi - lo + 1


def _max_axis(cin, cout, pad_lo, pad_hi):
    """Max source-interval length over all coords on one axis."""
    if cin < 1:
        return 1
    return max(1, _axis_interval(0, cin, cout, pad_lo, pad_hi),
               _axis_interval(cin - 1, cin, cout, pad_lo, pad_hi))


def replication_pad3d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    if not isinstance(padding, (list, tuple)) or len(padding) != 6:
        raise ValueError("padding must contain six values")
    if self.dim() < 3:
        raise ValueError("self must have at least three dimensions")
    if grad_output.device != self.device or grad_output.dtype != self.dtype:
        raise ValueError("grad_output and self must have the same device and dtype")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("replication_pad3d_backward supports floating point dtypes")

    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = map(int, padding)
    x = self.contiguous()
    grad_output = grad_output.contiguous()
    d_in, h_in, w_in = (int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1]))
    d_out = d_in + pad_front + pad_back
    h_out = h_in + pad_top + pad_bottom
    w_out = w_in + pad_left + pad_right
    expected_spatial = (d_out, h_out, w_out)
    if tuple(grad_output.shape[-3:]) != expected_spatial:
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-3:])} does not match {expected_spatial}"
        )
    if tuple(grad_output.shape[:-3]) != tuple(x.shape[:-3]):
        raise ValueError("grad_output and self must have matching leading dimensions")

    batch = math.prod(x.shape[:-3]) if x.dim() > 3 else 1
    total_in = d_in * h_in * w_in
    grad_output = grad_output.reshape(-1)

    if all(
        value == 0
        for value in (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    ):
        return grad_output.reshape(x.shape)

    grad_input = torch.empty_like(x).reshape(-1)
    block = 1024
    max_d = _max_axis(d_in, d_out, pad_front, pad_back)
    max_h = _max_axis(h_in, h_out, pad_top, pad_bottom)
    max_wl = max(1, triton.next_power_of_2(max(pad_left, 0)))
    max_wr = max(1, triton.next_power_of_2(max(pad_right, 0)))
    with torch_device_fn.device(x.device):
        _rep_pad3d_bwd_kernel[(batch, triton.cdiv(total_in, block))](
            grad_output,
            grad_input,
            total_in,
            d_in,
            h_in,
            w_in,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            pad_front,
            pad_back,
            d_out,
            h_out,
            w_out,
            BLOCK=block,
            MAXD=max_d,
            MAXH=max_h,
            MAXW_L=max_wl,
            MAXW_R=max_wr,
        )
    return grad_input.reshape(x.shape)
