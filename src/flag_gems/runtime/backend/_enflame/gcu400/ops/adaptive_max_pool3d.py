# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gcu400 adaptive_max_pool3d forward specialisation.

Differences from the generic ``flag_gems.ops.adaptive_max_pool3d``:

All argmax/index bookkeeping is staged through **int32** tensors.  On this
torch_gcu build, without ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` a Triton
kernel cannot store int32-computed values into an int64 tensor parameter
(the runtime silently replaces the long storage with int, so the 8-byte
strides the kernel computes land on the wrong bytes and the indices come
out interleaved garbage).  Turning that env var on is not an option here
because aten advanced indexing (used by the tests to validate indices)
aborts the process with it enabled.  Every kernel therefore writes its
partial/full indices into int32 tensors, and the host converts the final
index tensor to the API-required ``torch.int64`` with a plain aten cast
before returning; aten int64 arithmetic is safe on this target.

Index values stay far below 2**31: they are flat offsets inside one
(n, c) channel slab (``d * H * W + h * W + w``), so int32 is exact for
every shape this operator is expected to pool.
"""

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
    """Direct-scan: each thread independently scans one output element.

    gcu400 caps the hardware grid at 65535 along x, so the kernel runs as a
    grid-stride loop: the launch grid is clamped and each program strides
    over the remaining output blocks."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    c_groups = (in_c + CHAN_PER_BLOCK - 1) // CHAN_PER_BLOCK
    flat_elems = in_n * c_groups * out_d * out_h * out_w
    num_blocks = tl.cdiv(flat_elems, OUT_PER_BLOCK)

    tid = tl.arange(0, OUT_PER_BLOCK)
    for block_id in range(pid, num_blocks, num_programs):
        block_start = block_id * OUT_PER_BLOCK
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
            # gcu400: the (n, c) slab base offset is widened to int64 — a
            # >2 GiB input's byte offset overflows 32-bit address math and
            # faults the device (Sip error) otherwise.
            in_base = (
                in_ptr
                + (n_idx * in_c * in_d * in_h * in_w).to(tl.int64)
                + (c_idx * in_d * in_h * in_w).to(tl.int64)
            )

            acc_val = tl.full((OUT_PER_BLOCK,), min_val, dtype=dtype)
            # gcu400: indices are staged as int32 (see module docstring) — the
            # flat input offset inside one (n, c) slab always fits int32.
            acc_idx = (
                tl.full((OUT_PER_BLOCK,), -1, dtype=tl.int32)
                if RETURN_INDICES
                else acc_val
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
                            in_base + off,
                            mask=in_mask,
                            other=min_val,
                            cache_modifier=".ca",
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
    """Cooperative: one block per output element, T threads scan via chunks.

    gcu400 caps the hardware grid at 65535 along x, so the kernel runs as a
    grid-stride loop over output windows (the launch grid is clamped by the
    host)."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for win_id in range(pid, FLAT_ELEMS, num_programs):
        valid = win_id < FLAT_ELEMS

        out_vol = out_d * out_h * out_w
        hw_out = out_h * out_w
        nc_out = win_id // out_vol
        rem = win_id % out_vol
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
        # gcu400: widen the (n, c) slab base offset to int64 — >2 GiB inputs
        # overflow 32-bit byte-address math and fault the device otherwise.
        in_base = (
            in_ptr
            + (n_idx * in_c * in_d * in_h * in_w).to(tl.int64)
            + (c_idx * in_d * in_h * in_w).to(tl.int64)
        )
        chunk_size = (win_elems + COOP_THREADS - 1) // COOP_THREADS

        maxv = tl.full((COOP_THREADS,), min_val, dtype=dtype)
        # gcu400: per-thread argmax staged as int32 (see module docstring).
        maxi = tl.full((COOP_THREADS,), 0, dtype=tl.int32)
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

        tl.store(scratch_vals_ptr + win_id * COOP_THREADS + tid, maxv)
        if RETURN_INDICES:
            tl.store(scratch_idxs_ptr + win_id * COOP_THREADS + tid, maxi)

        tl.debug_barrier()

        lane0 = tid == 0
        # Vectorized reduction over the COOP_THREADS per-thread partials — this
        # replaces the old `for i in range(COOP_THREADS)` serial loop (a
        # constexpr-trip-count loop Triton warns about).
        scr = tl.arange(0, COOP_THREADS)
        v_all = tl.load(scratch_vals_ptr + win_id * COOP_THREADS + scr)
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
            final_v = tl.sum(
                tl.where(scr == first_i, v_all, tl.zeros_like(v_all)), axis=0
            )
        tl.store(output_ptr + win_id + 0 * tid, final_v, mask=lane0 & valid)
        if RETURN_INDICES:
            idx_all = tl.load(scratch_idxs_ptr + win_id * COOP_THREADS + scr)
            final_i = tl.sum(
                tl.where(scr == first_i, idx_all, tl.zeros_like(idx_all)), axis=0
            )
            tl.store(indices_ptr + win_id + 0 * tid, final_i, mask=lane0 & valid)


@libentry()
@triton.jit
def _d_reduce_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
):
    """NaN-aware max-reduce over the N dimension of a (M, N, K) tensor.

    Used for the out_d==1 path: reduce the D dimension of (N*C, in_d, H*W).
    Propagates NaN (a NaN in the window wins, matching torch.max) so no
    separate torch ``isnan/where`` re-mask is needed.

    The (M-blocks, K-blocks) 2D tiling is flattened into a single 1D
    grid-stride loop: gcu400 limits grid.x to 65535 and grid.y to 255, and
    either axis can exceed those caps for large N*C / H*W."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    k_blocks = tl.cdiv(K, BLOCK_K)
    total_blocks = tl.cdiv(M, BLOCK_M) * k_blocks

    dtype = in_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    for block_id in range(pid, total_blocks, num_programs):
        pid_m = block_id // k_blocks
        pid_k = block_id % k_blocks
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        m_ok = m < M
        k_ok = k < K
        mask = m_ok[:, None] & k_ok[None, :]
        acc = tl.full((BLOCK_M, BLOCK_K), min_val, dtype=dtype)
        # gcu400: D argmax staged as int32 (see module docstring).
        acc_idx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
        # int64 row base: (N*C, in_d, H*W) inputs can exceed 2 GiB.
        base = (m[:, None] * (N * K)).to(tl.int64) + k[None, :]
        for d in range(N):
            v = tl.load(in_ptr + base + d * K, mask=mask, other=min_val)
            # NaN-aware max: a NaN in the window must win (matches torch.max).
            better = (v > acc) | (v != v)
            acc = tl.where(better, v, acc)
            if RETURN_INDICES:
                acc_idx = tl.where(better, d, acc_idx)
        out_base = m[:, None] * K + k[None, :]
        tl.store(out_val_ptr + out_base, acc.to(dtype), mask=mask)
        if RETURN_INDICES:
            tl.store(out_idx_ptr + out_base, acc_idx, mask=mask)


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
    # 1D grid-stride loop: gcu400 caps grid.x at 65535.
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
        # int64 (n, c) slab base: d_argmax staging can exceed 2 GiB.
        d_off = (
            (n_idx * in_c * in_h * in_w + c_idx * in_h * in_w).to(tl.int64)
            + h_best * in_w
            + w_best
        )
        d_best = tl.load(d_argmax_ptr + d_off, mask=mask, other=0)
        tl.store(
            output_indices_ptr + offsets, d_best * in_h * in_w + spatial_idx, mask=mask
        )


@triton.jit
def _fill_identity_indices_kernel(
    indices_ptr,
    spatial_total,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid-stride loop: gcu400 caps grid.x at 65535.
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(indices_ptr + offsets, offsets % spatial_total, mask=mask)


def _compute_max_win(in_size: int, out_size: int) -> int:
    return (in_size + out_size - 1) // out_size + 1


# gcu400 hardware launch-grid cap along x.
_MAX_GRID_X = 65535


def _capped_grid_1d(num_blocks: int) -> int:
    return min(num_blocks, _MAX_GRID_X)


def _as_int64_indices(indices_i32: torch.Tensor) -> torch.Tensor:
    """Final aten cast of the staged int32 index tensor to the API dtype.

    aten int64 arithmetic / casts are safe on gcu400 without the 64-bit env
    switch; only Triton stores into int64 tensor parameters are broken (see
    module docstring)."""
    return indices_i32.to(torch.int64)


# ==============================================================================
# Main operator
# ==============================================================================


def adaptive_max_pool3d(input: torch.Tensor, output_size, return_indices=None):
    """3D adaptive max pooling (gcu400 specialisation).

    Mirrors ``torch.nn.functional.adaptive_max_pool3d``:
      ``return_indices=True``  → ``(output, indices)``
      ``return_indices=False`` → pooled output tensor only (kernels skip ALL
          argmax/index computation)
      ``return_indices=None``  (default; also what the aten dispatcher passes,
      whose schema is ``(Tensor, int[3]) -> (Tensor, Tensor)``) → the 2-tuple
      with real indices, since ``torch.adaptive_max_pool3d_with_indices``
      consumes them.

    Indices are computed by the kernels into int32 staging tensors and cast
    to ``torch.int64`` on the host before returning (see module docstring)."""
    if return_indices is None:
        return_indices = True  # aten 2-arg schema: real indices required
    want_indices = bool(return_indices)

    logger.debug("GEMS ENFLAME ADAPTIVE_MAX_POOL3D")

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
            # int32 staging tensor: the Triton fill kernel writes int32 values
            # (int64 stores are broken on this target without the 64-bit env
            # switch, which in turn aborts aten indexing in downstream users).
            indices_i32 = torch.empty(
                (in_n, in_c, out_d, out_h, out_w),
                device=input.device,
                dtype=torch.int32,
            )
            _fill_identity_indices_kernel[
                (_capped_grid_1d(triton.cdiv(indices_i32.numel(), 1024)),)
            ](indices_i32, spatial_total, indices_i32.numel(), BLOCK_SIZE=1024)
            return input, _as_int64_indices(indices_i32)
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
                si = torch.empty(nc * T, device=input.device, dtype=torch.int32)
                indices_i32 = torch.empty(
                    (in_n, in_c, 1, 1, 1), device=input.device, dtype=torch.int32
                )
            else:
                si = sv  # unused when RETURN_INDICES=False
                indices_i32 = sv
            flat_in = input.reshape(-1, spatial_volume)
            _kernel_cooperative[(_capped_grid_1d(nc),)](
                flat_in,
                sv,
                si,
                output.view(-1),
                indices_i32.view(-1),
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
                return output, _as_int64_indices(indices_i32)
            return output

    # --- out_d=1 D-reduce ---
    # For out_d=1, we have two strategies:
    #   A) _kernel_direct(out_d=1):   one kernel, MAX_WIN_D = in_d+1
    #   B) D-reduce kernel + _kernel_direct(in_d=1):  two kernels, MAX_WIN_D = 2
    # Strategy A is better for small in_d (avoid extra kernel launches).
    # Strategy B is better for large in_d (MAX_WIN_D dominates loop cost).
    # Threshold: in_d > 16 → fall back to D-reduce first.
    # For in_d <= 16, _kernel_direct handles it efficiently in one pass.
    d_argmax_i32 = None
    if out_d == 1 and in_d > 1:
        if in_d <= 16:
            # Single-kernel path: _kernel_direct handles D-reduce + (H,W) pool
            total = in_n * in_c * 1 * out_h * out_w
            output = torch.empty(
                (in_n, in_c, 1, out_h, out_w), device=input.device, dtype=input.dtype
            )
            # gcu400: always run the direct kernel with RETURN_INDICES=True.
            # Profiling shows the index computation + int32 store gives the
            # compiler extra ILP that hides the window-load latency: without
            # it the kernel is uniformly ~2.5x slower across every autotune
            # config (156.9 ms vs 61.1 ms on (2,640,64,42,72)->(16,14,14)).
            # When the caller does not want indices the buffer is scratch and
            # its contents are discarded.
            indices_i32 = torch.empty_like(output, dtype=torch.int32)
            # Use CB=1 for small total to avoid wasted blocks from channel grouping
            grid = lambda meta: (
                _capped_grid_1d(triton.cdiv(total, meta["OUT_PER_BLOCK"])),
            )
            _kernel_direct[grid](
                input,
                output,
                indices_i32,
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
                RETURN_INDICES=True,
            )
            if want_indices:
                return output, _as_int64_indices(indices_i32)
            return output
        else:
            # Two-kernel path: NaN-aware Triton D-reduce, then pool (H,W).
            # The in-file _d_reduce_kernel propagates NaN (a NaN in the window
            # wins), so no torch.max / torch.isnan / torch.where is needed.
            m = in_n * in_c
            n = in_d
            k = in_h * in_w
            d_reduced = torch.empty(
                (in_n, in_c, in_h, in_w), device=input.device, dtype=input.dtype
            )
            # gcu400: always compute the D argmax — same ILP effect as the
            # direct kernel (index work hides the load latency of the serial
            # max-select chain); discarded when indices are not requested.
            d_argmax_i32 = torch.empty(
                (in_n, in_c, in_h, in_w),
                device=input.device,
                dtype=torch.int32,
            )
            block_m = 64
            block_k = min(64, triton.next_power_of_2(k))
            # Flattened 1D grid (capped): the kernel decodes (m-block, k-block)
            # from the flat block id, so neither grid.y nor oversized grid.x
            # can hit the gcu400 hardware limits.
            total_blocks = triton.cdiv(m, block_m) * triton.cdiv(k, block_k)
            grid = (_capped_grid_1d(total_blocks),)
            _d_reduce_kernel[grid](
                input,
                d_reduced,
                d_argmax_i32,
                m,
                n,
                k,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                RETURN_INDICES=True,
            )
            input = d_reduced.unsqueeze(2)
            in_d = 1

            # Fast path: if H,W are identity after D-reduce, values are already
            # correct (max over D gave the right per-(h,w) value).  Just build
            # indices from d_argmax and return — no kernel needed.  The index
            # arithmetic runs through aten (int32 → int64 promotion), which is
            # safe on this target.
            if in_h == out_h and in_w == out_w:
                output = input  # already (N,C,1,H,W), correct values
                if want_indices:
                    hw = in_h * in_w
                    # Flat 3D index: d_best * H*W + h*W + w
                    base_spatial = torch.arange(
                        hw, device=input.device, dtype=torch.int64
                    ).view(1, 1, 1, out_h, out_w)
                    indices = (
                        d_argmax_i32.unsqueeze(2).to(torch.int64) * hw + base_spatial
                    )
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
    # gcu400: the direct kernel always runs with RETURN_INDICES=True (see the
    # out_d=1 comment — index work gives the compiler the ILP to hide window
    # load latency, ~2.5x faster).  Discarded when indices are not requested.
    indices_i32 = torch.empty_like(output, dtype=torch.int32)
    flat_k1 = total_output

    # Cooperative path for very small total_output with large windows.
    # One block per output element; T threads split the window scan and then
    # reduce their per-thread maxima.  The kernel decodes the full (n,c,d,h,w)
    # coordinates from the flat output index, so it works for any output shape.
    if total_output <= 64 and win_size >= 1024:
        T = _COOP_THREADS  # scratch sized to match COOP_THREADS
        sv = torch.empty(total_output * T, device=input.device, dtype=input.dtype)
        if want_indices:
            si = torch.empty(total_output * T, device=input.device, dtype=torch.int32)
        else:
            si = sv
        _kernel_cooperative[(_capped_grid_1d(total_output),)](
            input,
            sv,
            si,
            output.view(-1),
            indices_i32.view(-1),
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
        grid = lambda meta: (
            _capped_grid_1d(triton.cdiv(flat_k1, meta["OUT_PER_BLOCK"])),
        )
        _kernel_direct[grid](
            input,
            output,
            indices_i32,
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
            RETURN_INDICES=True,
        )

    if d_argmax_i32 is not None and want_indices:
        _merge_outd1_indices_kernel[
            (_capped_grid_1d(triton.cdiv(indices_i32.numel(), 256)),)
        ](
            indices_i32.view(-1),
            d_argmax_i32,
            indices_i32.view(-1),
            indices_i32.numel(),
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            BLOCK_SIZE=256,
        )
    if want_indices:
        return output, _as_int64_indices(indices_i32)
    return output
