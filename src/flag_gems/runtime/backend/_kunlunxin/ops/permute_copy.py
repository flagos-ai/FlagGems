import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def _pc_flat_copy_kernel(src_ptr, dst_ptr, n_words, BLOCK: tl.constexpr,
                         NEED_MASK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = off < n_words
        v = tl.load(src_ptr + off, mask=m)
        tl.store(dst_ptr + off, v, mask=m)
    else:
        v = tl.load(src_ptr + off)
        tl.store(dst_ptr + off, v)


@triton.jit
def _pc_gather_kernel(in_ptr, out_ptr, R, C,
                      N: tl.constexpr, A: tl.constexpr, B: tl.constexpr,
                      ST0: tl.constexpr, ST1: tl.constexpr, ST2: tl.constexpr,
                      ST3: tl.constexpr,
                      ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr,
                      NEED_MASK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = pid_c * COL_BLOCK + tl.arange(0, COL_BLOCK)
    r = rows
    if N == 1:
        off = tl.zeros([ROW_BLOCK], dtype=tl.int32)
        col_off = cols * ST0
    elif N == 2:
        off = r * ST0
        col_off = cols * ST1
    elif N == 3:
        d0 = r // A
        d1 = r % A
        off = d0 * ST0 + d1 * ST1
        col_off = cols * ST2
    else:
        d0 = r // (A * B)
        d1 = (r // B) % A
        d2 = r % B
        off = d0 * ST0 + d1 * ST1 + d2 * ST2
        col_off = cols * ST3
    m = (rows < R)[:, None] & (cols < C)[None, :]
    if NEED_MASK:
        v = tl.load(in_ptr + off[:, None] + col_off[None, :], mask=m,
                    other=0.0)
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], v, mask=m)
    else:
        v = tl.load(in_ptr + off[:, None] + col_off[None, :])
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], v)


def _pc_flat_copy(view, out, n, elemsz):
    """Byte-wise flat copy with 4-byte word coalescing when alignment allows."""
    src = view.reshape(-1)
    if n == 0:
        return
    if elemsz in (2, 4):
        # move as many 4-byte words as possible; keep sources contiguous
        words = (n * elemsz) // 4
        head_el = words * (4 // elemsz)
        if words > 0:
            src4 = src[:head_el].contiguous().view(torch.int32)
            dst4 = out.reshape(-1)[:head_el].view(torch.int32)
            BLOCK = 8192 if words <= 1048576 else 32768
            _pc_flat_copy_kernel[(triton.cdiv(words, BLOCK),)](
                src4, dst4, words, BLOCK, words % BLOCK != 0)
        tail_el = n - head_el
        if tail_el > 0:
            out_flat = out.reshape(-1)
            _pc_flat_copy_kernel[(1,)](src[-tail_el:], out_flat[-tail_el:],
                                       tail_el, 512, True)
    elif elemsz == 1:
        BLOCK = 8192 if n <= 1048576 else 32768
        _pc_flat_copy_kernel[(triton.cdiv(n, BLOCK),)](
            src, out, n, BLOCK, n % BLOCK != 0)
    else:  # 8-byte elements: direct 8-byte copy
        BLOCK = 2048 if n <= 262144 else 8192
        _pc_flat_copy_kernel[(triton.cdiv(n, BLOCK),)](
            src, out, n, BLOCK, n % BLOCK != 0)


def _pc_gather(view, x, out, n):
    N = view.dim()
    C = view.shape[-1]
    R = n // C
    sz = list(view.shape) + [1] * (4 - N)
    st = list(view.stride()) + [1] * (4 - N)
    RB, CB = 4, 128
    NEED_MASK = (R % RB != 0) or (C % CB != 0)
    grid = (triton.cdiv(R, RB), triton.cdiv(C, CB))
    _pc_gather_kernel[grid](x, out, R, C, N, sz[1], sz[2],
                            st[0], st[1], st[2], st[3],
                            RB, CB, NEED_MASK)


@triton.jit
def _pc_rowtable_kernel(in_ptr, out_ptr, row_ptr, R, C, COL_STRIDE,
                        ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = pid_c * COL_BLOCK + tl.arange(0, COL_BLOCK)
    rb = tl.load(row_ptr + rows)
    m = (rows < R)[:, None] & (cols < C)[None, :]
    addr = rb[:, None] + cols[None, :] * COL_STRIDE
    v = tl.load(in_ptr + addr, mask=m)
    tl.store(out_ptr + rows[:, None] * C + cols[None, :], v, mask=m)


def _pc_row_gather(view, x, out, n):
    """Rank > 4 fallback: per-row input base table + linear column stride."""
    C = view.shape[-1]
    R = n // C
    # row base = in-offset of element (row, 0). Decode on host with torch.
    coords = torch.meshgrid(
        *[torch.arange(s, dtype=torch.int32, device=x.device)
          for s in view.shape[:-1]], indexing="ij")
    row_base = sum(c.reshape(-1) * s for c, s in zip(coords, view.stride()[:-1]))
    row_base = row_base.contiguous()
    RB, CB = 16, 64
    grid = (triton.cdiv(R, RB), triton.cdiv(C, CB))
    _pc_rowtable_kernel[grid](x, out, row_base, R, C,
                              view.stride()[-1], RB, CB)


def permute_copy(x: torch.Tensor, dims):
    """Wrapper for aten::permute_copy: return a copy of x with permuted dims."""
    logger.debug("GEMS_KUNLUNXIN PERMUTE_COPY")
    ndim = x.ndim
    if ndim == 0:
        return x.clone()

    dims = [d if d >= 0 else d + ndim for d in dims]
    out_shape = [x.shape[d] for d in dims]

    if x.numel() == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    src = x.contiguous() if not x.is_contiguous() else x
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    view = src.permute(dims)
    n = view.numel()
    if view.is_contiguous():
        # identity-style permutation: the permuted view is already contiguous,
        # so this is a pure flat copy. Move 4-byte words to keep the memory
        # bound copy at full bandwidth.
        _pc_flat_copy(view, out, n, src.element_size())
    elif view.dim() <= 4:
        # strided permutation: decode each output row's input offset with
        # constexpr dims (no per-element div/mod) and gather COL_BLOCK-wide
        # tiles. Chosen on XPU (probe sweep: 4x128/w1 fastest & correct).
        _pc_gather(view, src, out, n)
    else:
        # rank > 4: per-row table fallback (correct for any permutation).
        _pc_row_gather(view, src, out, n)
    return out