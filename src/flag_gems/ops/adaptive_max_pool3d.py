import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

IS_METAX = runtime.device.vendor_name == "metax"


# ==============================================================================
# Kernel 1 — Direct scan  (M, K)
#
#   M = OUT_PER_BLOCK   — output elements per block
#   K = CHAN_PER_BLOCK  — channels per block (1 = standard, >1 = chunked)
#
# Autotuner selects the best (M, K).  Main function never passes them.
# ==============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("adaptive_max_pool3d"),
    key=[
        "MAX_WIN_D",
        "MAX_WIN_H",
        "MAX_WIN_W",
        "in_c",
        "in_n",
        "out_d",
        "out_h",
        "out_w",
    ],
)
@triton.jit
def _kernel_direct(
    in_ptr,
    out_ptr,
    idx_ptr,
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    MAX_WIN_D,
    MAX_WIN_H,
    MAX_WIN_W,
    OUT_PER_BLOCK: tl.constexpr,
    CHAN_PER_BLOCK: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
):
    """Direct-scan: each thread independently scans one output element."""
    pid = tl.program_id(0)
    block_start = pid * OUT_PER_BLOCK
    c_groups = (in_c + CHAN_PER_BLOCK - 1) // CHAN_PER_BLOCK
    flat_elems = in_n * c_groups * out_d * out_h * out_w

    tid = tl.arange(0, OUT_PER_BLOCK)
    flat_idx = block_start + tid
    valid = flat_idx < flat_elems

    tmp = flat_idx
    w_out_pos = tmp % out_w
    tmp //= out_w
    h_out_pos = tmp % out_h
    tmp //= out_h
    d_out_pos = tmp % out_d
    tmp //= out_d
    c_group = tmp % c_groups
    tmp //= c_groups
    n_idx = tmp

    n_idx = tl.where(valid, n_idx, 0)
    c_group = tl.where(valid, c_group, 0)
    d_out_pos = tl.where(valid, d_out_pos, 0)
    h_out_pos = tl.where(valid, h_out_pos, 0)
    w_out_pos = tl.where(valid, w_out_pos, 0)
    c_base = c_group * CHAN_PER_BLOCK

    d_start = d_out_pos * in_d // out_d
    win_d = ((d_out_pos + 1) * in_d + out_d - 1) // out_d - d_start
    h_start = h_out_pos * in_h // out_h
    win_h = ((h_out_pos + 1) * in_h + out_h - 1) // out_h - h_start
    w_start = w_out_pos * in_w // out_w
    win_w = ((w_out_pos + 1) * in_w + out_w - 1) // out_w - w_start

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    out_dhw = out_d * out_h * out_w

    for c_off in range(CHAN_PER_BLOCK):
        c_idx = c_base + c_off
        chan_valid = (c_idx < in_c) & valid
        in_base = (
            in_ptr + n_idx * in_c * in_d * in_h * in_w + c_idx * in_d * in_h * in_w
        )

        acc_val = tl.full((OUT_PER_BLOCK,), min_val, dtype=dtype)
        acc_idx = (
            tl.full((OUT_PER_BLOCK,), -1, dtype=tl.int64) if RETURN_INDICES else acc_val
        )

        for kd in range(MAX_WIN_D):
            d_in_raw = d_start + kd
            d_valid = (kd < win_d) & (d_in_raw < in_d) & chan_valid
            d_s = tl.where(d_valid, d_in_raw, 0)
            d_off = d_s * in_h * in_w
            for kh in range(MAX_WIN_H):
                h_in_raw = h_start + kh
                h_valid = (kh < win_h) & (h_in_raw < in_h) & chan_valid
                h_s = tl.where(h_valid, h_in_raw, 0)
                dh_off = d_off + h_s * in_w
                for kw in range(MAX_WIN_W):
                    w_in_raw = w_start + kw
                    w_valid = (kw < win_w) & (w_in_raw < in_w) & chan_valid
                    in_mask = d_valid & h_valid & w_valid
                    w_s = tl.where(w_valid, w_in_raw, 0)
                    off = dh_off + w_s
                    cv = tl.load(
                        in_base + off, mask=in_mask, other=min_val, cache_modifier=".ca"
                    )
                    # Align with PyTorch: compare in input dtype directly.
                    is_new = (cv > acc_val) | (cv != cv)
                    acc_val = tl.where(is_new, cv, acc_val)
                    if RETURN_INDICES:
                        ci = d_s * in_h * in_w + h_s * in_w + w_s
                        acc_idx = tl.where(is_new & in_mask, ci, acc_idx)

        out_off = (
            n_idx * in_c * out_dhw
            + c_idx * out_dhw
            + d_out_pos * out_h * out_w
            + h_out_pos * out_w
            + w_out_pos
        )
        tl.store(out_ptr + out_off, acc_val, mask=valid & chan_valid)
        if RETURN_INDICES:
            tl.store(idx_ptr + out_off, acc_idx, mask=valid & chan_valid)


# ==============================================================================
# Kernel 2 — Cooperative  (T threads scan one window together)
# ==============================================================================

# Threads per window block.  The scratch buffers below are allocated as
# num_windows * _COOP_THREADS, so this must stay in sync with the explicit
# COOP_THREADS passed to _kernel_cooperative.  (No autotune config exists for
# the cooperative kernel, so COOP_THREADS is bound explicitly at the call site.)
_COOP_THREADS = 256


@libentry()
@triton.jit
def _kernel_cooperative(
    in_ptr,
    scratch_vals_ptr,
    scratch_idxs_ptr,
    output_ptr,
    indices_ptr,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    FLAT_ELEMS: tl.constexpr,
    COOP_THREADS: tl.constexpr,
    MAX_WIN_D,
    MAX_WIN_H,
    MAX_WIN_W,
    RETURN_INDICES: tl.constexpr,
    REDUCE_IN_F32: tl.constexpr = False,
):
    """Cooperative: one block per output element, T threads scan via chunks."""
    pid = tl.program_id(0)
    valid = pid < FLAT_ELEMS

    out_vol = out_d * out_h * out_w
    hw_out = out_h * out_w
    nc_out = pid // out_vol
    rem = pid % out_vol
    d_out_idx = rem // hw_out
    rem2 = rem % hw_out
    h_out_idx = rem2 // out_w
    w_out_idx = rem2 % out_w
    n_idx = nc_out // in_c
    c_idx = nc_out % in_c

    n_idx = tl.where(valid, n_idx, 0)
    c_idx = tl.where(valid, c_idx, 0)
    d_out_idx = tl.where(valid, d_out_idx, 0)
    h_out_idx = tl.where(valid, h_out_idx, 0)
    w_out_idx = tl.where(valid, w_out_idx, 0)

    d_start = d_out_idx * in_d // out_d
    d_end = ((d_out_idx + 1) * in_d + out_d - 1) // out_d
    d_end = tl.where(d_out_idx == out_d - 1, in_d, d_end)
    win_d = d_end - d_start
    h_start = h_out_idx * in_h // out_h
    h_end = ((h_out_idx + 1) * in_h + out_h - 1) // out_h
    h_end = tl.where(h_out_idx == out_h - 1, in_h, h_end)
    win_h = h_end - h_start
    w_start = w_out_idx * in_w // out_w
    w_end = ((w_out_idx + 1) * in_w + out_w - 1) // out_w
    w_end = tl.where(w_out_idx == out_w - 1, in_w, w_end)
    win_w = w_end - w_start

    win_elems = win_d * win_h * win_w
    hw_win = win_h * win_w
    MAX_WIN_ELEMS = MAX_WIN_D * MAX_WIN_H * MAX_WIN_W
    MAX_CHUNK = tl.cdiv(MAX_WIN_ELEMS, COOP_THREADS)

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    tid = tl.arange(0, COOP_THREADS)
    in_base = in_ptr + n_idx * in_c * in_d * in_h * in_w + c_idx * in_d * in_h * in_w
    chunk_size = (win_elems + COOP_THREADS - 1) // COOP_THREADS

    maxv = tl.full((COOP_THREADS,), min_val, dtype=dtype)
    maxi = tl.full((COOP_THREADS,), 0, dtype=tl.int64)
    for i in range(MAX_CHUNK):
        flat_pos = tid * chunk_size + i
        in_chunk = i < chunk_size
        kd = flat_pos // hw_win
        rem_k = flat_pos % hw_win
        kh = rem_k // win_w
        kw = rem_k % win_w
        d_in = d_start + kd
        h_in = h_start + kh
        w_in = w_start + kw
        d_ok = (kd < win_d) & (d_in < in_d)
        h_ok = (kh < win_h) & (h_in < in_h)
        w_ok = (kw < win_w) & (w_in < in_w)
        elem_ok = d_ok & h_ok & w_ok
        # Use raw coords — the load mask (in_chunk & elem_ok) prevents
        # actual memory access for invalid positions.  Safe-clamping to 0
        # (tl.where) would produce addresses that alias real memory at d=0,
        # leaking window-external values into the reduction.
        ptr = in_base + d_in * in_h * in_w + h_in * in_w + w_in
        load_mask = in_chunk & elem_ok
        v = tl.load(ptr, mask=load_mask, other=min_val, cache_modifier=".ca")
        f_idx = d_in * in_h * in_w + h_in * in_w + w_in
        better = (v > maxv) | (v != v)
        maxv = tl.where(
            in_chunk, tl.where(elem_ok, tl.where(better, v, maxv), maxv), maxv
        )
        maxi = tl.where(
            in_chunk, tl.where(elem_ok, tl.where(better, f_idx, maxi), maxi), maxi
        )

    tl.store(scratch_vals_ptr + pid * COOP_THREADS + tid, maxv)
    if RETURN_INDICES:
        tl.store(scratch_idxs_ptr + pid * COOP_THREADS + tid, maxi)

    tl.debug_barrier()

    lane0 = tid == 0
    # Vectorized reduction over the COOP_THREADS per-thread partials — this
    # replaces the old `for i in range(COOP_THREADS)` serial loop (a
    # constexpr-trip-count loop Triton warns about).
    scr = tl.arange(0, COOP_THREADS)
    v_all = tl.load(scratch_vals_ptr + pid * COOP_THREADS + scr)
    # NaN must win (v != v), same as the per-chunk scan above; otherwise a
    # NaN in the window is dropped and the pooled output silently loses it.
    # Map NaN → +inf so tl.max picks it (and its index).
    v_eff = tl.where(v_all != v_all, float("inf"), v_all)
    max_v = tl.max(v_eff, axis=0)
    # First occurrence (smallest scan index) among the maxima — matches the
    # serial loop's strict-> update (ties keep the earlier element).
    is_max = v_eff == max_v
    first_i = tl.min(tl.where(is_max, scr, COOP_THREADS), axis=0)
    # Fill with explicit zeros (NOT 0 * v_all): the masked-out scratch slots
    # hold -inf, and 0 * -inf = NaN would poison the reduction sum.
    if REDUCE_IN_F32:
        final_v = tl.sum(
            tl.where(
                scr == first_i,
                v_all.to(tl.float32),
                tl.zeros_like(v_all).to(tl.float32),
            ),
            axis=0,
        ).to(v_all.dtype)
    else:
        final_v = tl.sum(tl.where(scr == first_i, v_all, tl.zeros_like(v_all)), axis=0)
    tl.store(output_ptr + pid + 0 * tid, final_v, mask=lane0 & valid)
    if RETURN_INDICES:
        idx_all = tl.load(scratch_idxs_ptr + pid * COOP_THREADS + scr)
        final_i = tl.sum(
            tl.where(scr == first_i, idx_all, tl.zeros_like(idx_all)), axis=0
        )
        tl.store(indices_ptr + pid + 0 * tid, final_i, mask=lane0 & valid)


# ==============================================================================
# Helpers
# ==============================================================================


@triton.jit
def _merge_outd1_indices_kernel(
    spatial_indices_ptr,
    d_argmax_ptr,
    output_indices_ptr,
    n_elements,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    out_spatial = out_h * out_w
    nc_idx = offsets // out_spatial
    rem = offsets % out_spatial
    h_out_pos = rem // out_w
    w_out_pos = rem % out_w
    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    n_idx = tl.where(mask, n_idx, 0)
    c_idx = tl.where(mask, c_idx, 0)
    h_out_pos = tl.where(mask, h_out_pos, 0)
    w_out_pos = tl.where(mask, w_out_pos, 0)
    spatial_idx = tl.load(spatial_indices_ptr + offsets, mask=mask, other=0)
    h_best = spatial_idx // in_w
    w_best = spatial_idx % in_w
    d_off = n_idx * in_c * in_h * in_w + c_idx * in_h * in_w + h_best * in_w + w_best
    d_best = tl.load(d_argmax_ptr + d_off, mask=mask, other=0)
    tl.store(
        output_indices_ptr + offsets, d_best * in_h * in_w + spatial_idx, mask=mask
    )


@triton.jit
def _fill_identity_indices_kernel(
    indices_ptr,
    spatial_total,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    tl.store(indices_ptr + offsets, offsets % spatial_total, mask=mask)


def _compute_max_win(in_size: int, out_size: int) -> int:
    return (in_size + out_size - 1) // out_size + 1


# ==============================================================================
# Main operator
# ==============================================================================


def adaptive_max_pool3d(input: torch.Tensor, output_size, return_indices=None):
    """3D adaptive max pooling.

    Mirrors ``torch.nn.functional.adaptive_max_pool3d``:
      ``return_indices=True``  → ``(output, indices)``
      ``return_indices=False`` → pooled output tensor only (kernels skip ALL
          argmax/index computation)
      ``return_indices=None``  (default; also what the aten dispatcher passes,
      whose schema is ``(Tensor, int[3]) -> (Tensor, Tensor)``) → the 2-tuple
      with real indices, since ``torch.adaptive_max_pool3d_with_indices``
      consumes them.
    """
    if return_indices is None:
        return_indices = True  # aten 2-arg schema: real indices required
    want_indices = bool(return_indices)

    logger.debug("GEMS ADAPTIVE_MAX_POOL3D")

    input = input.contiguous()
    if isinstance(output_size, int):
        output_size = [output_size, output_size, output_size]

    out_d, out_h, out_w = output_size
    in_n, in_c, in_d, in_h, in_w = input.shape

    # --- empty ---
    if out_d == 0 or out_h == 0 or out_w == 0 or in_d == 0 or in_h == 0 or in_w == 0:
        output = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
        )
        if want_indices:
            indices = torch.empty_like(output, dtype=torch.int64)
            return output, indices
        return output

    # --- identity ---
    if in_d == out_d and in_h == out_h and in_w == out_w:
        if want_indices:
            spatial_total = in_d * in_h * in_w
            indices = torch.empty(
                (in_n, in_c, out_d, out_h, out_w),
                device=input.device,
                dtype=torch.int64,
            )
            _fill_identity_indices_kernel[(triton.cdiv(indices.numel(), 1024),)](
                indices, spatial_total, indices.numel(), BLOCK_SIZE=1024
            )
            return input, indices
        return input

    # --- global pool (1,1,1) → cooperative kernel ---
    if out_d == 1 and out_h == 1 and out_w == 1:
        spatial_volume = in_d * in_h * in_w
        if spatial_volume >= 4:
            nc = in_n * in_c
            output = torch.empty(
                (in_n, in_c, 1, 1, 1), device=input.device, dtype=input.dtype
            )
            T = _COOP_THREADS
            sv = torch.empty(nc * T, device=input.device, dtype=input.dtype)
            if want_indices:
                si = torch.empty(nc * T, device=input.device, dtype=torch.int64)
                indices = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=torch.int64
                )
            else:
                si = sv  # unused when RETURN_INDICES=False
                indices = sv
            flat_in = input.reshape(-1, spatial_volume)
            _kernel_cooperative[(nc,)](
                flat_in,
                sv,
                si,
                output.view(-1),
                indices.view(-1),
                in_c=1,
                in_d=spatial_volume,
                in_h=1,
                in_w=1,
                out_d=1,
                out_h=1,
                out_w=1,
                FLAT_ELEMS=nc,
                COOP_THREADS=_COOP_THREADS,
                MAX_WIN_D=spatial_volume + 1,
                MAX_WIN_H=1,
                MAX_WIN_W=1,
                RETURN_INDICES=want_indices,
                REDUCE_IN_F32=IS_METAX,
            )
            if want_indices:
                return output, indices
            return output

    # --- out_d=1 D-reduce ---
    # For out_d=1, we have two strategies:
    #   A) _kernel_direct(out_d=1):   one kernel, MAX_WIN_D = in_d+1
    #   B) torch.max(dim=2) + _kernel_direct(in_d=1):  two kernels, MAX_WIN_D = 2
    # Strategy A is better for small in_d (avoid extra kernel launches).
    # Strategy B is better for large in_d (MAX_WIN_D dominates loop cost).
    # Threshold: in_d > 16 → fall back to D-reduce first.
    # For in_d <= 16, _kernel_direct handles it efficiently in one pass.
    d_argmax_saved = None
    if out_d == 1 and in_d > 1:
        if in_d <= 16:
            # Single-kernel path: _kernel_direct handles D-reduce + (H,W) pool
            total = in_n * in_c * 1 * out_h * out_w
            output = torch.empty(
                (in_n, in_c, 1, out_h, out_w), device=input.device, dtype=input.dtype
            )
            indices = (
                torch.empty_like(output, dtype=torch.int64) if want_indices else output
            )
            # Use CB=1 for small total to avoid wasted blocks from channel grouping
            grid = lambda meta: (triton.cdiv(total, meta["OUT_PER_BLOCK"]),)
            _kernel_direct[grid](
                input,
                output,
                indices,
                in_n,
                in_c,
                in_d,
                in_h,
                in_w,
                1,
                out_h,
                out_w,
                MAX_WIN_D=_compute_max_win(in_d, 1),
                MAX_WIN_H=_compute_max_win(in_h, out_h),
                MAX_WIN_W=_compute_max_win(in_w, out_w),
                RETURN_INDICES=want_indices,
            )
            if want_indices:
                return output, indices
            return output
        else:
            # Two-kernel path: reduce D via torch.max first, then pool (H,W)
            if want_indices:
                d_reduced, d_argmax_saved = input.max(dim=2)
            else:
                d_reduced = input.max(dim=2).values
            input = d_reduced.unsqueeze(2)
            in_d = 1

            # Fast path: if H,W are identity after D-reduce, values are already
            # correct (max over D gave the right per-(h,w) value).  Just build
            # indices from d_argmax and return — no kernel needed.
            if in_h == out_h and in_w == out_w:
                output = input  # already (N,C,1,H,W), correct values
                if want_indices:
                    hw = in_h * in_w
                    # Build flat 3D index: d_best * H*W + h*W + w
                    base_spatial = torch.arange(
                        hw, device=input.device, dtype=torch.int64
                    ).view(1, 1, 1, out_h, out_w)
                    indices = d_argmax_saved.unsqueeze(2) * hw + base_spatial
                    return output, indices
                return output

    # --- main dispatch ---
    max_win_d = _compute_max_win(in_d, out_d)
    max_win_h = _compute_max_win(in_h, out_h)
    max_win_w = _compute_max_win(in_w, out_w)
    total_output = in_n * in_c * out_d * out_h * out_w
    win_size = max_win_d * max_win_h * max_win_w

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty_like(output, dtype=torch.int64) if want_indices else output
    flat_k1 = total_output

    # Cooperative path for very small total_output with large windows.
    # One block per output element; T threads split the window scan and then
    # reduce their per-thread maxima.  The kernel decodes the full (n,c,d,h,w)
    # coordinates from the flat output index, so it works for any output shape.
    if total_output <= 64 and win_size >= 1024:
        T = _COOP_THREADS  # scratch sized to match COOP_THREADS
        sv = torch.empty(total_output * T, device=input.device, dtype=input.dtype)
        if want_indices:
            si = torch.empty(total_output * T, device=input.device, dtype=torch.int64)
        else:
            si = sv
        _kernel_cooperative[(total_output,)](
            input,
            sv,
            si,
            output.view(-1),
            indices.view(-1),
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            FLAT_ELEMS=total_output,
            COOP_THREADS=_COOP_THREADS,
            MAX_WIN_D=max_win_d,
            MAX_WIN_H=max_win_h,
            MAX_WIN_W=max_win_w,
            RETURN_INDICES=want_indices,
            REDUCE_IN_F32=IS_METAX,
        )
    else:
        # Direct scan — autotuner picks (M, K)
        grid = lambda meta: (triton.cdiv(flat_k1, meta["OUT_PER_BLOCK"]),)
        _kernel_direct[grid](
            input,
            output,
            indices,
            in_n,
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            MAX_WIN_D=max_win_d,
            MAX_WIN_H=max_win_h,
            MAX_WIN_W=max_win_w,
            RETURN_INDICES=want_indices,
        )

    if d_argmax_saved is not None and want_indices:
        _merge_outd1_indices_kernel[(triton.cdiv(indices.numel(), 256),)](
            indices.view(-1),
            d_argmax_saved,
            indices.view(-1),
            indices.numel(),
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            BLOCK_SIZE=256,
        )
    if want_indices:
        return output, indices
    return output
