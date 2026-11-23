"""GCU400-specific implementation of cummaxmin_backward (scatter-add semantics).

Semantics: ``grad_input.scatter_add_(dim, indices, grad_output)``, accumulating repeated
indices, with the accumulation done in float32 and then cast back to the input dtype.

Two measurements on GCU determine the shape of this implementation:
- ``torch.zeros`` memset runs at only ~2 GB/s (a 4096x4096 67MB buffer needs ~31ms), while a
  Triton kernel writing the same buffer takes ~0.1ms, so zeroing always uses ``_zero_kernel``;
- ``tl.atomic_add`` is very slow on this hardware (~30ms for 4096x4096, 100x a plain store of
  the same size), so large reduction axes switch to the atomic-free "prefix sum + segment boundary" path.

Derivation of the atomic-free path (``_cmb_prefix_kernel``): when ``indices`` is non-decreasing
along the reduction axis, each target position t maps to one contiguous segment [s(t), e(t)] of
that axis, so ``out[t] = S[e(t)] - S[s(t)-1]`` where S is the prefix sum along the axis. Segment
starts and ends come from comparing adjacent indices, and each t is written exactly once (no
write conflict):
- the segment start writes ``S[r]-g[r]`` into the start buffer;
- the segment end writes ``S[r]`` into the output;
- one element-wise ``out -= start`` finishes the result.
If a row's indices turn out not to be non-decreasing, that row falls back to the atomic path
inside the same kernel, and the semantics stay correct.
"""

import torch
import triton
import triton.language as tl

__all__ = ["cummaxmin_backward"]

# Upper limit of grid.x on GCU
_MAX_GRID_X = 65535
# Best BLOCK for elementwise kernels (zeroing / final subtraction): measured 32768 takes only 0.11ms for 268MB,
# clearly better than 1024 (3.4ms) and 8192 (0.43ms), so these two kernels use a large BLOCK to fill a single wave;
# For small buffers, round numel up to a power of 2 to avoid using only a few of the 32768 lanes.
_ELEM_BLOCK = 32768
# Dual thresholds for enabling the atomic-free "prefix sum + segment boundaries" path (from an empirical sweep):
# Atomic overhead only becomes the dominant cost when the reduction axis is long enough and there are enough total
# elements; on small shapes, the atomic path
# is faster because it avoids one indices read and one kernel. From RS=1024/numel=1M, the prefix path leads by 6.8x,
# and the two are on par at RS=512/numel=128K, so 512 + 256K is used as the switch point.
_PREFIX_RS_THRESHOLD = 512
_PREFIX_NUMEL_THRESHOLD = 262144
# Threshold for using torch.zeros for small buffers: measured 64x64 (4096 elements) with torch.zeros is actually slower
# than
# torch.empty + a Triton zeroing kernel by 20% (0.093ms vs 0.076ms), so always use the Triton kernel.
_SMALL_NUMEL = 0


@triton.jit
def _zero_kernel(OUT, N, BLOCK: tl.constexpr):
    """Zeroes the first N elements of OUT (grid-stride, avoiding a grid.x overflow)."""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        tl.store(OUT + pos, 0.0, mask=pos < N)


@triton.jit
def _sub_kernel(OUT, SUB, N, BLOCK: tl.constexpr):
    """OUT -= SUB (element-wise, grid-strided, in place)."""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        m = pos < N
        tl.store(
            OUT + pos, tl.load(OUT + pos, mask=m) - tl.load(SUB + pos, mask=m), mask=m
        )


@triton.jit
def _sub_cast_kernel(OUT, SUB, DST, N, BLOCK: tl.constexpr):
    """DST = OUT - SUB written directly in the input dtype: one pass for both the closing subtraction and the dtype write-back."""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        m = pos < N
        val = tl.load(OUT + pos, mask=m) - tl.load(SUB + pos, mask=m)
        tl.store(DST + pos, val.to(DST.dtype.element_ty), mask=m)


@triton.jit
def _cmb_kernel(G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    """Atomic path (reduction axis innermost).

    grid.x is capped at 65535, so a fixed grid plus a grid-stride walk covers the outer rows.
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // IS
        inner = row % IS
        base = outer * (RS * IS) + inner
        for bs in range(0, RS, BLOCK):
            pos = bs + ar
            if MASKED:
                m = pos < RS
                g = tl.load(G + base + pos * IS, mask=m, other=0.0)
                t = tl.load(IDX + base + pos * IS, mask=m, other=0)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + base + pos * IS)
                t = tl.load(IDX + base + pos * IS)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32))


@triton.jit
def _cmb_inner_kernel(
    G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    """Atomic path (reduction axis in the middle): vectorized along the contiguous inner axis.

    ``row`` indexes (outer, r); each lane scatters ``grad[outer, r, j]`` to
    ``grad_input[outer, t, j]``, where ``t`` comes from the forward indices and inner column j is unchanged.
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // RS
        outer_base = outer * (RS * IS)
        src_base = outer_base + (row % RS) * IS
        for bs in range(0, IS, BLOCK):
            j = bs + ar
            if MASKED:
                m = j < IS
                g = tl.load(G + src_base + j, mask=m, other=0.0)
                t = tl.load(IDX + src_base + j, mask=m, other=0)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + src_base + j)
                t = tl.load(IDX + src_base + j)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32))


@triton.jit
def _cmb_prefix_kernel(G, IDX, OUT, START, NROW, RS, BLOCK: tl.constexpr):
    """Atomic-free path (reduction axis innermost, valid when indices are non-decreasing along it).

    Each program handles a whole row: it first verifies that the row indices are non-decreasing,
    and if so takes the prefix sum + segment boundary writes, otherwise the row falls back to
    atomic scatter. Each t is written at most once on the atomic-free path, so no atomics are needed.
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        base = row * RS

        # First pass: verify that indices are non-decreasing (no comparison at pos==0)
        bad = 0
        for bs in range(0, RS, BLOCK):
            pos = bs + ar
            m = pos < RS
            cur = tl.load(IDX + base + pos, mask=m, other=0)
            prv = tl.load(IDX + base + pos - 1, mask=(pos > 0) & m, other=0)
            bad += tl.sum(((cur < prv) & (pos > 0) & m).to(tl.int32), 0)

        if bad == 0:
            # Second pass: prefix sum + segment-start/segment-end writes (no write conflicts)
            carry = 0.0
            for bs in range(0, RS, BLOCK):
                pos = bs + ar
                m = pos < RS
                g = tl.load(G + base + pos, mask=m, other=0.0).to(tl.float32)
                cur = tl.load(IDX + base + pos, mask=m, other=0)
                prv = tl.load(IDX + base + pos - 1, mask=(pos > 0) & m, other=0)
                nxt = tl.load(IDX + base + pos + 1, mask=(pos + 1 < RS) & m, other=0)
                t = cur.to(tl.int32)
                s = tl.cumsum(g, 0) + carry
                is_start = (pos == 0) | (cur != prv)
                is_end = (pos == RS - 1) | (cur != nxt)
                tl.store(START + base + t, s - g, mask=m & is_start)
                tl.store(OUT + base + t, s, mask=m & is_end)
                carry += tl.sum(g, 0)
        else:
            # Non-decreasing check failed: fall back to the atomic path for this row
            for bs in range(0, RS, BLOCK):
                pos = bs + ar
                m = pos < RS
                g = tl.load(G + base + pos, mask=m, other=0.0)
                t = tl.load(IDX + base + pos, mask=m, other=0)
                tl.atomic_add(OUT + base + t, g.to(tl.float32), mask=m)


def _zero_fill(numel, dtype, device):
    """Allocates and zeroes a 1-D buffer.

    On small buffers ``torch.zeros`` memset matches a Triton kernel while saving one launch, so
    ``torch.zeros`` is used directly when size does not exceed ``_SMALL_NUMEL``.
    """
    if numel <= _SMALL_NUMEL:
        return torch.zeros(numel, dtype=dtype, device=device)
    buf = torch.empty(numel, dtype=dtype, device=device)
    BLOCK = min(_ELEM_BLOCK, max(1024, triton.next_power_of_2(numel)))
    grid = (min(triton.cdiv(numel, BLOCK), _MAX_GRID_X),)
    _zero_kernel[grid](buf, numel, BLOCK=BLOCK, num_warps=8)
    return buf


def cummaxmin_backward(grad_output, input, indices, dim):
    ndim = grad_output.ndim
    if dim < 0:
        dim += ndim
    shape = list(grad_output.shape)
    RS = shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    if RS == 0:
        return torch.zeros(shape, dtype=grad_output.dtype, device=grad_output.device)

    # Make non-contiguous inputs contiguous first: all kernels address data in [outer, reduction, inner] row-major order
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    numel = outer_size * RS * inner_size
    use_prefix = (
        inner_size == 1
        and RS >= _PREFIX_RS_THRESHOLD
        and numel >= _PREFIX_NUMEL_THRESHOLD
    )
    if use_prefix:
        # The output and segment-start buffer share one allocation; zeroing runs only one kernel
        buf = _zero_fill(2 * numel, torch.float32, grad_output.device)
        out = buf[:numel].view(shape)
        start = buf[numel:]
        _cmb_prefix_kernel[(min(outer_size, _MAX_GRID_X),)](
            grad_output,
            indices,
            out,
            start,
            outer_size,
            RS,
            BLOCK=min(triton.next_power_of_2(RS), 2048),
            num_warps=8,
        )
        BLOCK = min(_ELEM_BLOCK, max(1024, triton.next_power_of_2(numel)))
        grid = (min(triton.cdiv(numel, BLOCK), _MAX_GRID_X),)
        if grad_output.dtype is torch.float32:
            _sub_kernel[grid](out, start, numel, BLOCK=BLOCK, num_warps=8)
        else:
            # Subtract and write back to the target dtype in one pass: avoids one fp32 write and one cast kernel
            dst = torch.empty(shape, dtype=grad_output.dtype, device=out.device)
            _sub_cast_kernel[grid](out, start, dst, numel, BLOCK=BLOCK, num_warps=8)
            return dst
    else:
        out = _zero_fill(numel, torch.float32, grad_output.device).view(shape)
        if inner_size == 1:
            BLOCK = min(triton.next_power_of_2(RS), 1024)
            _cmb_kernel[(min(outer_size, _MAX_GRID_X),)](
                grad_output,
                indices,
                out,
                outer_size,
                RS,
                inner_size,
                BLOCK=BLOCK,
                MASKED=(BLOCK != RS),
                num_warps=8,
            )
        else:
            BLOCK = min(triton.next_power_of_2(inner_size), 1024)
            nrow = outer_size * RS
            _cmb_inner_kernel[(min(nrow, _MAX_GRID_X),)](
                grad_output,
                indices,
                out,
                nrow,
                RS,
                inner_size,
                BLOCK=BLOCK,
                MASKED=(BLOCK != inner_size),
                num_warps=8,
            )
    if grad_output.dtype is torch.float32:
        return out
    return out.to(grad_output.dtype)
