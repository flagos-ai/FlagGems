"""linalg.matrix_power — raises a square matrix to an integer power.

Adopts the same parallel strategy as the PyTorch ATen implementation:
  - Binary exponentiation (O(log n) matrix multiplications).
  - Intra-op parallelism: each matmul is internally parallelised by the
    underlying backend (BLAS / CUDA / FlagGems Triton matmul).
  - Batch parallelism: all batch elements follow the same exponentiation
    path; batched matmul handles them simultaneously.

Optimisation:
  - M <= 32: **single-program fused kernel** — the full MxM matrix fits in
    one tl.dot call.  The entire binary exponentiation runs inside a single
    kernel launch, eliminating per-matmul dispatch overhead.
  - 32 < M <= 64: **tiled multi-launch kernel** — the 64×64 matmul is
    decomposed into a 2×2 grid of 32×32 tl.dot tiles across 4 programs.
    Three kernel launches chain the binary-exponentiation steps through a
    scratch buffer, providing grid-level synchronisation while keeping
    tl.dot in its sweet spot (32×32).
  - M > 64: batched torch.mm / torch.bmm fallback with TF32 disabled
    (matching PyTorch's NoTF32Guard).
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold: matrices up to this size use the fused Triton kernel.
# M=64 uses BLOCK=64 with a single tl.dot call per matmul — slightly
# slower than cuBLAS for 64×64 (speedup ~0.7x) but still far better than
# the multi-launch fallback (~0.06x).  Above this we fall back to torch.mm.
# ---------------------------------------------------------------------------
TRITON_THRESHOLD = 64  # max M for any Triton path (single-tile or tiled)


# ===========================================================================
# Kernel 1  — Single-tile fused  (M <= SINGLE_TILE_THRESHOLD)
# ===========================================================================


@libentry()
@triton.heuristics(
    values={
        "num_warps": lambda args: (4 if args["BLOCK"] <= 16
                                   else 8 if args["BLOCK"] <= 32
                                   else 8),  # 8 warps matches cuBLAS
        "num_stages": lambda args: (2 if args["BLOCK"] <= 32 else 4),
    }
)
@triton.jit(do_not_specialize=["n"])
def _single_tile_kernel(
    A_ptr, out_ptr, M, n, batch_stride, BLOCK: tl.constexpr,
):
    """One program per batch element.  M <= BLOCK, one tl.dot per matmul."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK); offs_n = tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)

    a_base = A_ptr + pid * batch_stride
    out_base = out_ptr + pid * batch_stride

    a = tl.load(a_base + offs_m[:, None] * M + offs_n[None, :], mask=mask, other=0.0)

    acc_dtype = tl.float32
    z = a.to(acc_dtype)
    result = z
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                result = z; has_result = True
            else:
                result = tl.dot(result, z, allow_tf32=False)
            result = tl.where(mask, result, 0.0)
        n_remaining >>= 1
        if n_remaining > 0:
            z = tl.dot(z, z, allow_tf32=False)
            z = tl.where(mask, z, 0.0)

    result = result.to(a.dtype)
    tl.store(out_base + offs_m[:, None] * M + offs_n[None, :], result, mask=mask)


# ===========================================================================
# Kernel 2 — Grid-level sync fused  (M > 32, single kernel, multi-SM)
# ===========================================================================

TILE = 32


@libentry()
@triton.jit(do_not_specialize=["n"])
def _grid_sync_kernel(
    A_ptr, out_ptr, scratch_ptr, barrier_ptr,
    M, n, batch_stride,
    TILE_BLOCK: tl.constexpr, TILES: tl.constexpr,
):
    """Single-kernel binary exponentiation with grid-level sync.

    Grid: ``(batch_size, TILES, TILES)``.
    Each program owns one TILE×TILE output tile and runs the entire
    binary-exponentiation loop.  Between matmul steps all programs
    synchronise via an atomic barrier in global memory, giving
    multi-SM parallelism while keeping everything in one kernel launch.
    """
    pid_batch = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_m = tl.arange(0, TILE_BLOCK)
    offs_k = tl.arange(0, TILE_BLOCK)
    offs_n = tl.arange(0, TILE_BLOCK)

    # Row / col range for this program's tile.
    rm = pid_i * TILE_BLOCK + offs_m
    rn = pid_j * TILE_BLOCK + offs_n
    mask = (rm[:, None] < M) & (rn[None, :] < M)

    # Base pointers for this batch element.
    a_base = A_ptr + pid_batch * batch_stride
    out_base = out_ptr + pid_batch * batch_stride
    scratch_stride = M * M

    # total_progs = total programs per batch element = TILES * TILES.
    # Each batch element has its own barrier slot (barrier_ptr + pid_batch * 64).
    n_total = TILES * TILES

    # -----------------------------------------------------------------
    # Step 0 — copy input A tiles to scratch[0] (z) and scratch[2] (result)
    # -----------------------------------------------------------------
    a_tile = tl.load(
        a_base + rm[:, None] * M + rn[None, :], mask=mask, other=0.0,
    )
    tl.store(
        scratch_ptr + 0 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile, mask=mask,
    )
    tl.store(
        scratch_ptr + 2 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile, mask=mask,
    )

    # Ping-pong indices for the scratch buffer (4 slots per batch).
    #   z_buf:   0 or 1   — current power of two
    #   r_buf:   2 or 3   — current result
    z_buf = 0
    r_buf = 2
    has_result = False
    n_remaining = n
    barrier_base = barrier_ptr + pid_batch * 64

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                has_result = True; r_buf = 2
            else:
                dst_r = 5 - r_buf
                _compute_tiled_matmul(
                    scratch_ptr + r_buf * scratch_stride,
                    scratch_ptr + z_buf * scratch_stride,
                    scratch_ptr + dst_r * scratch_stride,
                    M, rm, rn, offs_m, offs_k, offs_n,
                    mask, TILE_BLOCK, TILES,
                )
                r_buf = dst_r
        n_remaining >>= 1
        if n_remaining > 0:
            dst_z = 1 - z_buf
            _compute_tiled_matmul(
                scratch_ptr + z_buf * scratch_stride,
                scratch_ptr + z_buf * scratch_stride,
                scratch_ptr + dst_z * scratch_stride,
                M, rm, rn, offs_m, offs_k, offs_n,
                mask, TILE_BLOCK, TILES,
            )
            z_buf = dst_z

        # ---- Grid-level barrier (release/acquire semantics) ----
        my_count = tl.atomic_add(barrier_base, 1, sem="release")
        barrier_round = (my_count // n_total) + 1
        target = barrier_round * n_total
        # Spin with acquire semantics for faster visibility
        while tl.atomic_add(barrier_base, 0, sem="acquire") < target:
            pass

    # ---- Store final result ----
    tl.store(
        out_base + rm[:, None] * M + rn[None, :],
        tl.load(
            scratch_ptr + r_buf * scratch_stride + rm[:, None] * M + rn[None, :],
            mask=mask, other=0.0,
        ),
        mask=mask,
    )


@triton.jit
def _compute_tiled_matmul(
    A_base, B_base, C_base, M,
    rm, rn, offs_m, offs_k, offs_n, mask_c,
    TILE_BLOCK: tl.constexpr, TILES: tl.constexpr,
):
    """Compute one tile of C = A @ B, storing result to C_base."""
    acc = tl.zeros((TILE_BLOCK, TILE_BLOCK), dtype=tl.float32)
    for tk in range(TILES):
        rk = tk * TILE_BLOCK + offs_k
        mask_a = (rm[:, None] < M) & (rk[None, :] < M)
        mask_b = (rk[:, None] < M) & (rn[None, :] < M)
        a_tile = tl.load(
            A_base + rm[:, None] * M + rk[None, :], mask=mask_a, other=0.0,
        )
        b_tile = tl.load(
            B_base + rk[:, None] * M + rn[None, :], mask=mask_b, other=0.0,
        )
        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32), allow_tf32=False)
    acc = tl.where(mask_c, acc, 0.0)
    tl.store(C_base + rm[:, None] * M + rn[None, :], acc, mask=mask_c)


# ===========================================================================
# Buffer cache — reuse scratch / barrier tensors across calls
# ===========================================================================

_buffer_cache = {}

def _get_scratch_buffers(batch_size: int, m: int, dtype, device):
    """Return (scratch, barrier) tensors, reusing cached allocations."""
    key = (batch_size, m, dtype, device)
    if key in _buffer_cache:
        scratch, barrier = _buffer_cache[key]
        barrier.zero_()
        return scratch, barrier
    scratch = torch.empty(4 * batch_size, m, m, dtype=dtype, device=device)
    barrier = torch.zeros(batch_size * 64, dtype=torch.int32, device=device)
    _buffer_cache[key] = (scratch, barrier)
    return scratch, barrier


# ===========================================================================
# Thresholds for dispatch
# ===========================================================================

SINGLE_TILE_MAX = 32   # single-program fused kernel (fastest for M <= 32)
TILED_MAX = 64          # multi-program tiled kernel  (33 <= M <= 64)


# ===========================================================================
# CUDA graph helpers for the fallback path
# ===========================================================================

_graph_cache = {}


def _build_binary_chain(n: int):
    """Return a list of (op, src_a, src_b, dst) tuples for binary exponentiation.

    Buffers: 0 = A (input), 1 = z (current power of two), 2 = result.
    Each matmul step writes to a distinct buffer.
    """
    chain = []
    # Scratch buffer assignment:
    #   slot 0: input A (read-only after initial copy)
    #   slot 1: z   (ping-pong with slot 3)
    #   slot 2: result (ping-pong with slot 4)
    #   Total: 5 buffers
    z_buf = 1
    r_buf = 2
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                # result = z  (copy)
                chain.append(("copy", z_buf, 0, r_buf))
                has_result = True
            else:
                # result = result @ z  →  other result slot
                dst = 4 if r_buf == 2 else 2
                chain.append(("mm", r_buf, z_buf, dst))
                r_buf = dst
        n_remaining >>= 1
        if n_remaining > 0:
            # z = z @ z  →  other z slot
            dst = 3 if z_buf == 1 else 1
            chain.append(("mm", z_buf, z_buf, dst))
            z_buf = dst

    return chain, r_buf  # which buffer holds the final result


def _get_or_create_graph(
    A_flat: torch.Tensor, n: int, out_flat: torch.Tensor,
    is_batched: bool,
):
    """Return a (graph, buffers) tuple for the given (M, n, batch) key."""
    M = A_flat.shape[-1]
    dtype = A_flat.dtype
    device = A_flat.device
    batch_size = A_flat.shape[0]
    key = (M, n, batch_size, dtype, is_batched)

    if key in _graph_cache:
        return _graph_cache[key]

    # ---- Build the operation chain ----
    chain, result_buf = _build_binary_chain(n)

    # ---- Allocate static buffers (5 slots) ----
    buf_shape = (batch_size, M, M) if is_batched else (M, M)
    buffers = [torch.empty(buf_shape, dtype=dtype, device=device) for _ in range(5)]

    # ---- Capture graph ----
    g = torch.cuda.CUDAGraph()
    _mm = torch.bmm if is_batched else torch.mm

    # Disable TF32 during capture to match PyTorch's NoTF32Guard
    _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.cuda.graph(g):
            # Only capture the matmul chain — input/output copies
            # happen outside the graph so it works with any A / out.
            for op, src_a, src_b, dst in chain:
                if op == "copy":
                    buffers[dst].copy_(buffers[src_a])
                else:
                    _mm(buffers[src_a], buffers[src_b], out=buffers[dst])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _prev_tf32

    # Warmup once (both input slots need initial data).
    inp = A_flat if is_batched else A_flat.squeeze(0)
    buffers[0].copy_(inp)
    buffers[1].copy_(inp)
    g.replay()

    entry = (g, buffers, result_buf, is_batched)
    _graph_cache[key] = entry
    return entry


# ===========================================================================
# Host helpers
# ===========================================================================


def _eye_like(A: torch.Tensor) -> torch.Tensor:
    m = A.shape[-1]
    shape = A.shape
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    if len(shape) > 2:
        eye = eye.expand(shape[:-2] + (m, m)).clone()
    return eye


# ===========================================================================
# Main entry point
# ===========================================================================


def linalg_matrix_power(
    A: torch.Tensor, n: int, *, out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS LINALG_MATRIX_POWER")

    # ---- validation ----
    shape = A.shape
    if len(shape) < 2:
        raise ValueError(f"linalg.matrix_power: A must be at least 2-D, got shape {shape}")
    m, k = shape[-2], shape[-1]
    if m != k:
        raise ValueError(f"linalg.matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg.matrix_power: n must be int, got {type(n).__name__}")

    # ---- n == 0 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None: out.copy_(eye); return out
        return eye

    # ---- n == 1 (only use fused kernel for M <= TRITON_THRESHOLD) ----
    if n == 1:
        use_triton = m <= TRITON_THRESHOLD and A.device.type == flag_gems.device
        if not use_triton:
            if out is not None: out.copy_(A); return out
            return A.clone()

    # ---- negative n ----
    if n < 0:
        A = torch.linalg.inv(A); n = -n

    # ---- n == 2, 3: fast paths for large M ----
    if n == 2 and m > TRITON_THRESHOLD:
        r = torch.matmul(A, A)
        if out is not None: out.copy_(r); return out
        return r
    if n == 3 and m > TRITON_THRESHOLD:
        r = torch.matmul(torch.matmul(A, A), A)
        if out is not None: out.copy_(r); return out
        return r

    # ---- flatten batch ----
    if len(shape) > 2:
        A_flat = A.reshape(-1, m, m)
    else:
        A_flat = A.unsqueeze(0)
    batch_size = A_flat.shape[0]
    batch_stride = m * m

    if out is not None:
        out_flat = out.reshape(-1, m, m)
    else:
        out_flat = torch.empty(batch_size, m, m, dtype=A.dtype, device=A.device)

    # ---- dispatch ----
    if m <= SINGLE_TILE_MAX and A.device.type == flag_gems.device:
        # Tier 1: single-program fused (M <= 32).  tl.dot in sweet spot.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat, out_flat, m, n, batch_stride, BLOCK=BLOCK,
        )

    elif m <= TILED_MAX and A.device.type == flag_gems.device:
        # Tier 2: single-tile (33 <= M <= 64).
        # Grid-sync barrier overhead (~5 us/barrier × 3) exceeds the
        # single-SM tl.dot(64,64) time for 4-tile grids.  CUDA graph
        # memcpy overhead (~5 us × 3 copies) also dominates for M≤64.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat, out_flat, m, n, batch_stride, BLOCK=BLOCK,
        )

    elif A.device.type == flag_gems.device and m <= 256:
        # Tier 3: grid-level sync fused (65 <= M <= 256).
        TILES = triton.cdiv(m, TILE)
        scratch, barrier = _get_scratch_buffers(batch_size, m, A.dtype, A.device)
        _grid_sync_kernel[(batch_size, TILES, TILES)](
            A_flat, out_flat, scratch, barrier,
            m, n, batch_stride, TILE_BLOCK=TILE, TILES=TILES,
        )

    else:
        # Fallback: CUDA graph or batched torch.mm (M > 64 or CPU).
        is_batched = batch_size > 1
        if A.device.type == flag_gems.device and n >= 2:
            # CUDA graph path: matmul chain captured once, replayed per call.
            g, buffers, result_buf, _is_batched = _get_or_create_graph(
                A_flat, n, out_flat, is_batched,
            )
            inp = A_flat if is_batched else A_flat.squeeze(0)
            buffers[0].copy_(inp)
            buffers[1].copy_(inp)  # z starts as A
            g.replay()
            if is_batched:
                out_flat.copy_(buffers[result_buf])
            else:
                out_flat.squeeze_(0).copy_(buffers[result_buf])
        else:
            # CPU fallback (or n < 2, already handled above).
            _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                z = A_flat if is_batched else A_flat.squeeze(0)
                result = None
                while n > 0:
                    bit = n % 2; n //= 2
                    if bit == 1:
                        result = z.clone() if result is None else torch.mm(result, z)
                    if n > 0:
                        z = torch.mm(z, z)
                if result is not None:
                    out_flat = result if is_batched else result.unsqueeze(0)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = _prev_tf32

    # ---- reshape back ----
    if len(shape) > 2:
        out_flat = out_flat.reshape(shape)
    else:
        out_flat = out_flat.squeeze(0)

    if out is not None: return out
    return out_flat
